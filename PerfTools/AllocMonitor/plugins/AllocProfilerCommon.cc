#include <version>
#if (__cpp_lib_stacktrace >= 202011L) && (__cpp_lib_formatters >= 202302L) && \
    (__cpp_lib_ranges_enumerate >= 202302L) && (__cpp_lib_print >= 202207L)

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AllocProfilerCommon.h"

namespace cms::perftools::allocMon::profiler {
  void CollectingStatisticsStrategy::printStatistics(edm::LogSystem& log,
                                                     std::size_t numAllocTraces,
                                                     std::size_t numDeallocTraces) const {
    log.format("\nStatistics:\nAllocations recorded {} in {}\nDeallocations recorded {} in {}\n",
               numAllocations_,
               allocationsTime_,
               numDeallocations_,
               deallocationsTime_);
    log.format("Allocation traces total {} aggregated {}; aggregation {} sorting {} formatting {}\n",
               numAllocTraces,
               allocStats_.uniqueAggregated,
               allocStats_.t_aggregation,
               allocStats_.t_sorting,
               allocStats_.t_format);
    log.format(
        "AtMaxActual sorting {} formatting {}\n", allocStats_.t_atMaxActualSorting, allocStats_.t_atMaxActualFormat);
    log.format("Added filtering {} formatting {}\n", allocStats_.t_addedFiltering, allocStats_.t_addedFormat);
    log.format("Deallocation traces total {} aggregated {}; aggregation {} sorting {} formatting {}\n",
               numDeallocTraces,
               deallocStats_.uniqueAggregated,
               deallocStats_.t_aggregation,
               deallocStats_.t_sorting,
               deallocStats_.t_format);
    log.format("Churn aggregated traces {} inner container max size {}; aggregation {} sorting {} formatting {}",
               churnStats_.uniqueAggregated,
               churnStats_.maxInnerSize,
               churnStats_.t_aggregation,
               churnStats_.t_sorting,
               churnStats_.t_format);
  }

  StackNodeData::StackNodeData(std::string filePattern, ReportConfiguration config)
      : filePattern_(std::move(filePattern)),
        statistics_(config.printStatistics_ ? static_cast<std::unique_ptr<StatisticsStrategy>>(
                                                  std::make_unique<CollectingStatisticsStrategy>())
                                            : std::make_unique<NoOpStatisticsStrategy>()),
        config_(config) {}

  void StackNodeData::recordAllocation(std::stacktrace trace, AllocationRecord record, void const* ptr) {
    auto timer = statistics_->recordAllocation();

    UniqueTraceIndex traceIndex;
    auto hash = std::hash<std::stacktrace>()(trace);
    auto found = hashToAllocTraceIndex_.find(hash);
    if (found != hashToAllocTraceIndex_.end()) {
      traceIndex = found->second;
      assert(trace == uniqueAllocTraces_[traceIndex].trace_);
      uniqueAllocTraces_[traceIndex].recordAllocation(record);
    } else {
      // intentionally narrowing
      traceIndex = static_cast<UniqueTraceIndex>(uniqueAllocTraces_.size());
      uniqueAllocTraces_.emplace_back(record, std::move(trace));
      hashToAllocTraceIndex_.emplace(hash, traceIndex);
    }
    addressToAllocation_.emplace(ptr, std::tuple{record, traceIndex});

    presentActual_ += record.actual_;
    if (presentActual_ > maxActual_) {
      for (auto& trace : uniqueAllocTraces_) {
        trace.setAtMaxActual();
      }
      maxActual_ = presentActual_;
    }
  }

  void StackNodeData::print(edm::LogSystem& log, std::string_view measurementName) const {
    // Compute longest list of common trace entries fron the top across al recorded traces (both alloc and dealloc) so
    // that a single consistent measurement context is stripped from every section of the report.
    auto allAllocTraces =
        uniqueAllocTraces_ | std::views::transform([](auto const& at) -> std::stacktrace const& { return at.trace_; });
    auto allDeallocTraces = uniqueDeallocTraces_ |
                            std::views::transform([](auto const& dt) -> std::stacktrace const& { return dt.trace_; });
    std::size_t const commonTopEntries = computeCommonTopEntries(allAllocTraces, allDeallocTraces);

    // Print the dynamically computed common context so the user knows what call chain all reported traces are
    // relative to.  Pick the first available trace as the representative; the common suffix frames are identical
    // across all traces by definition.
    auto const* refTrace = !uniqueAllocTraces_.empty()     ? &uniqueAllocTraces_.front().trace_
                           : !uniqueDeallocTraces_.empty() ? &uniqueDeallocTraces_.front().trace_
                                                           : nullptr;
    std::string commonTraceContext;
    if (commonTopEntries > 0 && refTrace != nullptr) {
      auto const skipFromBottom =
          (commonTopEntries < refTrace->size()) ? static_cast<int>(refTrace->size() - commonTopEntries) : 0;
      commonTraceContext = formatTrace(*refTrace, skipFromBottom, 0);
    }
    if (filePattern_.empty() && !commonTraceContext.empty()) {
      log.format(" Reported stack traces are based on\n{}", commonTraceContext);
    }

    printAllocations(log, measurementName, commonTopEntries, commonTraceContext);
    if (config_.deallocationReport_) {
      printDeallocations(log, measurementName, commonTopEntries, commonTraceContext);
    }
    if (config_.churnReport_) {
      printChurn(log, measurementName, commonTopEntries, commonTraceContext);
    }

    statistics_->printStatistics(log, uniqueAllocTraces_.size(), uniqueDeallocTraces_.size());
  }

  void StackNodeData::subtractDeallocationFromAllocations(std::optional<UniqueTraceIndex> deallocIndex,
                                                          void const* ptr,
                                                          MonitorStackNode* prevNode) {
    auto found = addressToAllocation_.find(ptr);
    if (found == addressToAllocation_.end()) {
      // The memory was allocated before the measurement region
      // If there is an earlier node in the stack, try to subtract it from there
      if (prevNode) {
        prevNode->get().recordDeallocationFromNestedMeasurement(ptr, prevNode->previousNode());
      }
      // If there is no earlier node in the stack, ignore the deallocation
      return;
    }
    // subtraction
    auto const& allocRecord = std::get<AllocationRecord>(found->second);
    auto const allocIndex = std::get<UniqueTraceIndex>(found->second);
    uniqueAllocTraces_[allocIndex].recordDeallocation(allocRecord);

    // Account the presentActual on the node where the memory was allocated
    // This can lead to some possibly weird cases when a nested node deallocates memory
    presentActual_ -= allocRecord.actual_;

    // churn analysis (only if deallocation is from the same measurement and not from a nested one)
    if (deallocIndex) {
      analyzeDeallocationForChurn(*deallocIndex, allocRecord, allocIndex);
    }

    addressToAllocation_.erase(found);
  }

  void StackNodeData::analyzeDeallocationForChurn(UniqueTraceIndex deallocIndex,
                                                  AllocationRecord const& record,
                                                  UniqueTraceIndex allocIndex) {
    // allocIndex points to an element in uniqueAllocTraces_ and is thus at most the size of uniqueAllocTraces_
    if (allocIndex >= churnRecords_.size()) {
      churnRecords_.resize(uniqueAllocTraces_.size());
    }
    auto& churns = churnRecords_[allocIndex];
    auto found =
        std::ranges::find_if(churns, [deallocIndex](auto& elem) { return elem.deallocIndex_ == deallocIndex; });
    if (found == churns.end()) {
      churns.push_back(ChurnRecord{.deallocIndex_ = deallocIndex, .total_ = record});
    } else {
      found->total_.add(record);
    }
  }

  StackNodeData::UniqueTraceIndex StackNodeData::addDeallocation(std::stacktrace trace, DeallocationRecord record) {
    auto hash = std::hash<std::stacktrace>()(trace);
    auto found = hashToDeallocTraceIndex_.find(hash);
    UniqueTraceIndex traceIndex;
    if (found != hashToDeallocTraceIndex_.end()) {
      traceIndex = found->second;
      assert(trace == uniqueDeallocTraces_[traceIndex].trace_);
      uniqueDeallocTraces_[traceIndex].recordDeallocation(record);
    } else {
      traceIndex = uniqueDeallocTraces_.size();
      uniqueDeallocTraces_.emplace_back(record, std::move(trace));
      hashToDeallocTraceIndex_.emplace(hash, traceIndex);
    }
    return traceIndex;
  }

  void StackNodeData::printAllocations(edm::LogSystem& log,
                                       std::string_view measurementName,
                                       std::size_t commonTopEntries,
                                       std::string_view commonTraceContext) const {
    using AggregationMap = std::unordered_map<std::string, AllocationTrace::Records>;

    AggregationMap aggregatedAllocs;
    {
      auto timer = statistics_->timeAllocationsAggregation();
      for (auto const& record : uniqueAllocTraces_) {
        auto const skipTop = (commonTopEntries > 0 && commonTopEntries < record.trace_.size())
                                 ? static_cast<int>(commonTopEntries) - 1
                                 : 0;
        aggregatedAllocs[formatTrace(record.trace_, 0, skipTop)].add(record.records_);
      }
      statistics_->recordAllocationsUniqueAggregated(aggregatedAllocs.size());
    }

    std::vector<std::unordered_map<std::string, AllocationTrace::Records>::const_iterator> orderAllocs;
    {
      auto timer = statistics_->timeAllocationsSorting();
      orderAllocs = makeSortedIterators(aggregatedAllocs, [](auto const& it) { return it->second.total_.actual_; });
    }

    {
      auto timer = statistics_->timeAllocationsFormatting();
      log.format("\nAll allocations");
      writeFileOrMessage("alloc", measurementName, commonTraceContext, log, [&](std::ostream& os) {
        for (auto const& itTrace : orderAllocs) {
          auto const& records = itTrace->second;
          std::print(os,
                     "count {} requested {} actual {}\n{}",
                     records.total_.count_,
                     records.total_.requested_,
                     records.total_.actual_,
                     itTrace->first);
        }
      });
    }

    // AtMaxActual
    {
      auto timer = statistics_->timeAllocationsAtMaxActualSorting();
      std::ranges::sort(orderAllocs, std::greater{}, [](auto const& it) { return it->second.atMaxActual_.actual_; });
    }

    {
      auto timer = statistics_->timeAllocationsAtMaxActualFormatting();
      log.format("\nAllocated memory at the max actual moment");
      writeFileOrMessage("atMaxActual", measurementName, commonTraceContext, log, [&](std::ostream& os) {
        for (auto const& itTrace : orderAllocs) {
          auto const& records = itTrace->second;
          // Stack traces of allocations done after the "max actual"
          // moment can show zero values, filter them out.
          if (records.atMaxActual_.count_ != 0) {
            std::print(os,
                       "count {} requested {} actual {}\n{}",
                       records.atMaxActual_.count_,
                       records.atMaxActual_.requested_,
                       records.atMaxActual_.actual_,
                       itTrace->first);
          }
        }
      });
    }

    // Added
    {
      auto timer = statistics_->timeAllocationsAddedFiltering();
      std::erase_if(orderAllocs, [](auto const& it) { return it->second.added_.count_ == 0; });
      std::ranges::sort(orderAllocs, std::greater{}, [](auto const& it) { return it->second.added_.actual_; });
    }

    {
      auto timer = statistics_->timeAllocationsAddedFormatting();
      log.format("\nAdded memory");
      writeFileOrMessage("added", measurementName, commonTraceContext, log, [&](std::ostream& os) {
        for (auto const& itTrace : orderAllocs) {
          auto const& records = itTrace->second;
          std::print(os,
                     "count {} requested {} actual {}\n{}",
                     records.added_.count_,
                     records.added_.requested_,
                     records.added_.actual_,
                     itTrace->first);
          // The keys of aggregatedAllocs are large, so destruct them as soon as possible. The erase() does not
          // invalidate the iterators to other elements, so it's safe to erase while iterating through orderAllocs.
          aggregatedAllocs.erase(itTrace);
        }
      });
    }
  }

  void StackNodeData::printDeallocations(edm::LogSystem& log,
                                         std::string_view measurementName,
                                         std::size_t commonTopEntries,
                                         std::string_view commonTraceContext) const {
    using AggregationMap = std::unordered_map<std::string, DeallocationRecord>;
    AggregationMap aggregatedDeallocs;

    {
      auto timer = statistics_->timeDeallocationAggregation();
      for (auto const& record : uniqueDeallocTraces_) {
        auto const skipTop = (commonTopEntries > 0 && commonTopEntries < record.trace_.size())
                                 ? static_cast<int>(commonTopEntries) - 1
                                 : 0;
        aggregatedDeallocs[formatTrace(record.trace_, 0, skipTop)].add(record.total_);
      }
      statistics_->recordDeallocationsUniqueAggregated(aggregatedDeallocs.size());
    }

    std::vector<std::unordered_map<std::string, DeallocationRecord>::const_iterator> orderDeallocs;
    {
      auto timer = statistics_->timeDeallocationsSorting();
      orderDeallocs = makeSortedIterators(aggregatedDeallocs, [](auto const& it) { return it->second.actual_; });
    }

    {
      auto timer = statistics_->timeDeallocationsFormatting();
      log.format("\nAll deallocations");
      writeFileOrMessage("dealloc", measurementName, commonTraceContext, log, [&](std::ostream& os) {
        for (auto const& itTrace : orderDeallocs) {
          auto const& record = itTrace->second;
          std::print(os, "count {} actual {}\n{}", record.count_, record.actual_, itTrace->first);
          aggregatedDeallocs.erase(itTrace);
        }
      });
    }
  }

  void StackNodeData::printChurn(edm::LogSystem& log,
                                 std::string_view measurementName,
                                 std::size_t commonTopEntries,
                                 std::string_view commonTraceContext) const {
    using AggregationMap = std::unordered_map<std::string, AllocationRecord>;
    AggregationMap aggregatedChurn;
    AggregationMap aggregatedChurnAllocs;

    {
      auto timer = statistics_->timeChurnAggregation();
      auto traceToString = [](auto const& traceEntry) { return traceEntry.description(); };
      // zip stops at the shorter range, naturally handling churnRecords_ being possibly smaller than
      // uniqueAllocTraces_
      for (auto const& [allocRecord, churnVec] : std::views::zip(uniqueAllocTraces_, churnRecords_)) {
        auto const& allocTrace = allocRecord.trace_;
        // Strip the common topmost frames that are shared across all traces in the measurement.
        // Subtract 1 so the measurement-starting function itself remains visible in the output.
        auto const allocKeep = (commonTopEntries > 0 && commonTopEntries < allocTrace.size())
                                   ? allocTrace.size() - (commonTopEntries - 1)
                                   : allocTrace.size();
        auto const allocTraceStrings = allocTrace | std::views::take(allocKeep) | std::views::transform(traceToString) |
                                       std::ranges::to<std::vector<std::string>>();

        statistics_->recordChurnMaxInnerSize(churnVec.size());
        for (auto const& record : churnVec) {
          // Stopping to the lowest common stacktrace_entry between allocation and deallocation
          {
            auto const& deallocTrace = uniqueDeallocTraces_[record.deallocIndex_].trace_;
            auto const deallocKeep = (commonTopEntries > 0 && commonTopEntries < deallocTrace.size())
                                         ? deallocTrace.size() - (commonTopEntries - 1)
                                         : deallocTrace.size();
            auto const deallocTraceStrings = deallocTrace | std::views::take(deallocKeep) |
                                             std::views::transform(traceToString) |
                                             std::ranges::to<std::vector<std::string>>();

            // iterate from the top (outermost) downward to find where the traces first diverge
            auto allocEntries = allocTraceStrings | std::views::reverse;
            auto deallocEntries = deallocTraceStrings | std::views::reverse;

            auto [it_alloc, it_dealloc] = std::ranges::mismatch(allocEntries, deallocEntries);
            // If it_alloc and it_dealloc point to different
            // functions, go one level up (toward outer frames) to
            // find the common function.
            //
            // If it_alloc and it_dealloc point to the same function
            // (can happen at least with realloc()), it_alloc would
            // point to allocEntries.end(), and to get the lowest
            // entry need to decrease the iterator as well.
            assert(it_alloc != allocEntries.begin());
            --it_alloc;

            // Index back into the original stacktrace_entry range so the formatter
            // includes source-location info (e.g. "at :0") in the output.
            auto const idx = static_cast<std::size_t>(std::prev(it_alloc.base()) - allocTraceStrings.begin());
            auto str_trace =
                formatTrace(std::ranges::subrange(allocTrace.cbegin() + idx, allocTrace.cbegin() + allocKeep));
            aggregatedChurn[str_trace].add(record.total_);
          }
          // Churn allocation stack traces
          {
            auto str_trace = formatTrace(std::ranges::subrange(allocTrace.cbegin(), allocTrace.cbegin() + allocKeep));
            aggregatedChurnAllocs[str_trace].add(record.total_);
          }
        }
      }
      statistics_->recordChurnUniqueAggregated(aggregatedChurn.size());
    }

    std::vector<std::unordered_map<std::string, AllocationRecord>::const_iterator> orderChurn, orderChurnAllocs;
    {
      auto timer = statistics_->timeChurnSorting();
      orderChurn = makeSortedIterators(aggregatedChurn, [](auto const& it) { return it->second.count_; });
      orderChurnAllocs = makeSortedIterators(aggregatedChurnAllocs, [](auto const& it) { return it->second.count_; });
    }

    {
      auto timer = statistics_->timeChurnFormatting();
      log.format("\nMemory allocation+deallocation churn");
      writeFileOrMessage("churn", measurementName, commonTraceContext, log, [&](std::ostream& os) {
        for (auto const& it : orderChurn) {
          auto const& record = it->second;
          std::print(
              os, "count {} requested {} actual {}\n{}", record.count_, record.requested_, record.actual_, it->first);
          aggregatedChurn.erase(it);
        }
      });
      log.format("\nMemory allocation+deallocation churn allocations");
      writeFileOrMessage("churnalloc", measurementName, commonTraceContext, log, [&](std::ostream& os) {
        for (auto const& it : orderChurnAllocs) {
          auto const& record = it->second;
          std::print(
              os, "count {} requested {} actual {}\n{}", record.count_, record.requested_, record.actual_, it->first);
          aggregatedChurnAllocs.erase(it);
        }
      });
    }
  }

  std::size_t StackNodeData::intersectCommonTopEntries(std::stacktrace const& ref,
                                                       std::vector<std::string>& refDescriptionCache,
                                                       std::size_t commonLen,
                                                       std::stacktrace const& tr) {
    std::size_t k = 0;
    while (k < commonLen && k < tr.size()) {
      auto const refIdx = ref.size() - 1 - k;
      auto const trIdx = tr.size() - 1 - k;

      if (ref[refIdx] != tr[trIdx]) {
        // Frames don't match - check if descriptions match (e.g. due to inlining).
        auto& cachedDesc = refDescriptionCache[refIdx];
        if (cachedDesc.empty()) {
          cachedDesc = ref[refIdx].description();
        }
        if (cachedDesc != tr[trIdx].description()) {
          break;  // Neither operator== nor descriptions match
        }
      }
      ++k;  // Single increment: either frames matched, or descriptions matched
    }
    return k;
  }

}  // namespace cms::perftools::allocMon::profiler
#endif
