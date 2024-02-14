#include <memory>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/DTTriggerPhase2/interface/TrapezoidalGrouping.h"

using namespace edm;
using namespace std;
using namespace cmsdt;
using namespace dtamgrouping;
// ============================================================================
// Constructors and destructor
// ============================================================================
TrapezoidalGrouping::TrapezoidalGrouping(const ParameterSet &pset, edm::ConsumesCollector &iC)
    : MotherGrouping(pset, iC), debug_(pset.getUntrackedParameter<bool>("debug")), currentBaseChannel_(-1) {
  // Obtention of parameters
  if (debug_)
    LogDebug("TrapezoidalGrouping") << "TrapezoidalGrouping: constructor";

  // Initialisation of channelIn array
  chInDummy_.push_back(DTPrimitive());
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
      channelIn_[lay][ch] = {chInDummy_};
      channelIn_[lay][ch].clear();
    }
  }
}

TrapezoidalGrouping::~TrapezoidalGrouping() {
  if (debug_)
    LogDebug("TrapezoidalGrouping") << "TrapezoidalGrouping: destructor";
}

// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void TrapezoidalGrouping::initialise(const edm::EventSetup &iEventSetup) {
  if (debug_)
    LogDebug("TrapezoidalGrouping") << "TrapezoidalGrouping::initialiase";
}

void TrapezoidalGrouping::run(Event &iEvent,
                              const EventSetup &iEventSetup,
                              const DTDigiCollection &digis,
                              MuonPathPtrs &mpaths) {
  //   This function returns the analyzable mpath collection back to the the main function
  //   so it can be fitted. This is in fact doing the so-called grouping.
  for (int supLayer = 0; supLayer < NUM_SUPERLAYERS; supLayer++) {  // for each SL:
    if (debug_)
      LogDebug("TrapezoidalGrouping") << "TrapezoidalGrouping::run Reading SL" << supLayer;
    setInChannels(&digis, supLayer);

    for (auto &hit : all_hits) {
      int layer_to_pivot = hit.layerId();
      int channel_to_pivot = hit.channelId();
      DTPrimitives hits_in_trapezoid;
      std::vector<DTPrimitives> hit_mpaths;
      std::vector<int> hit_tasks;
      for (size_t itask = 0; itask < task_list.size(); itask++) {
        // when pivoting over an internal layer, there are two cases
        // where the second layer is duplicated
        // 12 (0, 5) <-> 14 (0, 7)
        // 15 (1, 6) <-> 17 (1, 8)
        // we leave it hard-coded here, could be moved somewhere else
        if (layer_to_pivot == 1 || layer_to_pivot == 2) {
          if (itask == 14 || itask == 17)
            continue;
        }

        auto task = task_list[itask];

        std::vector<DTPrimitives> task_mpaths;
        std::stack<std::pair<DTPrimitives, int>> mpath_cells_per_task;
        mpath_cells_per_task.push(std::make_pair(DTPrimitives({hit}), 0));

        while (!mpath_cells_per_task.empty()) {
          auto mpath_cells = std::move(mpath_cells_per_task.top());
          std::vector<DTPrimitives> tmp_mpaths = {mpath_cells.first};
          auto task_index = mpath_cells.second;
          auto cell = task[task_index];
          auto vertical_shift = trapezoid_vertical_mapping[layer_to_pivot][cell];
          auto horizontal_shift = trapezoid_horizontal_mapping[layer_to_pivot][cell];
          if (channel_to_pivot + horizontal_shift >= 0 && channel_to_pivot + horizontal_shift < NUM_CH_PER_LAYER) {
            tmp_mpaths = group_hits(hit,
                                    tmp_mpaths,
                                    channelIn_[layer_to_pivot + vertical_shift][channel_to_pivot + horizontal_shift],
                                    hits_in_trapezoid);
          }
          mpath_cells_per_task.pop();
          for (const auto &tmp_mpath : tmp_mpaths) {
            mpath_cells_per_task.push(std::make_pair(tmp_mpath, task_index + 1));
          }
          while (!mpath_cells_per_task.empty()) {
            if (mpath_cells_per_task.top().second == (int)task.size()) {
              task_mpaths.push_back(mpath_cells_per_task.top().first);
              mpath_cells_per_task.pop();
            } else
              break;
          }
        }
        for (auto &task_mpath : task_mpaths) {
          hit_mpaths.push_back(task_mpath);
          hit_tasks.push_back(itask);
        }
      }
      if (hits_in_trapezoid.size() <= PATHFINDER_INPUT_HITS_LIMIT) {
        for (size_t ipath = 0; ipath < hit_mpaths.size(); ipath++) {
          auto ptrPrimitive = hit_mpaths[ipath];
          auto itask = hit_tasks[ipath];

          // In any case, if we have less than 3 hits, we don't output the mpath
          if (ptrPrimitive.size() < 3)
            continue;

          // check if the task has a missing layer associated
          // if it does, we add a dummy hit in the missing layer
          // if it does not, we check that we actually have 4 present hits;
          // if not, we skip the mpath.
          if (MISSING_LAYER_LAYOUTS_PER_TASK[layer_to_pivot][itask] != -1) {
            auto dtpAux = DTPrimitive();
            dtpAux.setTDCTimeStamp(-1);
            dtpAux.setChannelId(-1);
            dtpAux.setLayerId(MISSING_LAYER_LAYOUTS_PER_TASK[layer_to_pivot][itask]);  //  L=0,1,2,3
            dtpAux.setSuperLayerId(hit.superLayerId());
            dtpAux.setCameraId(-1);
            ptrPrimitive.push_back(dtpAux);
          } else {  // we have no missing hits, it must be a 4-hit TP.
            if (ptrPrimitive.size() < 4)
              continue;
          }

          // sort the hits by layer, so they are included ordered in the MuonPath object
          std::stable_sort(ptrPrimitive.begin(), ptrPrimitive.end(), hitLayerSort);

          auto ptrMuonPath = std::make_shared<MuonPath>(ptrPrimitive);
          ptrMuonPath->setCellHorizontalLayout(CELL_HORIZONTAL_LAYOUTS_PER_TASK[layer_to_pivot][itask]);
          ptrMuonPath->setMissingLayer(MISSING_LAYER_LAYOUTS_PER_TASK[layer_to_pivot][itask]);
          mpaths.push_back(std::move(ptrMuonPath));
        }
      }
    }
  }
  if (debug_)
    LogDebug("TrapezoidalGrouping") << "[TrapezoidalGrouping::run] end";
}

void TrapezoidalGrouping::finish() { return; };

// ============================================================================
// Other methods
// ============================================================================

void TrapezoidalGrouping::setInChannels(const DTDigiCollection *digis, int sl) {
  //   before setting channels we need to clear
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
      channelIn_[lay][ch].clear();
    }
  }
  all_hits.clear();

  // now fill with those primitives that make sense:
  for (const auto &dtLayerId_It : *digis) {
    const DTLayerId dtLId = dtLayerId_It.first;

    if (dtLId.superlayer() != sl + 1)
      continue;  //skip digis not in SL...

    for (DTDigiCollection::const_iterator digiIt = (dtLayerId_It.second).first; digiIt != (dtLayerId_It.second).second;
         ++digiIt) {
      int layer = dtLId.layer() - 1;
      int wire = (*digiIt).wire() - 1;
      int digiTIME = (*digiIt).time();
      int digiTIMEPhase2 = digiTIME;

      if (debug_)
        LogDebug("TrapezoidalGrouping") << "[TrapezoidalGrouping::setInChannels] SL" << sl << " L" << layer << " : "
                                        << wire << " " << digiTIMEPhase2;
      auto dtpAux = DTPrimitive();
      dtpAux.setTDCTimeStamp(digiTIMEPhase2);
      dtpAux.setChannelId(wire);
      dtpAux.setLayerId(layer);    //  L=0,1,2,3
      dtpAux.setSuperLayerId(sl);  // SL=0,1,2
      dtpAux.setCameraId(dtLId.rawId());
      channelIn_[layer][wire].push_back(dtpAux);
      all_hits.push_back(dtpAux);
    }
  }

  // sort everything by the time of the hits, so it has the same behaviour as the fw
  for (int lay = 0; lay < NUM_LAYERS; lay++) {
    for (int ch = 0; ch < NUM_CH_PER_LAYER; ch++) {
      std::stable_sort(channelIn_[lay][ch].begin(), channelIn_[lay][ch].end(), hitTimeSort);
    }
  }
  std::stable_sort(all_hits.begin(), all_hits.end(), hitTimeSort);
}

std::vector<DTPrimitives> TrapezoidalGrouping::group_hits(DTPrimitive pivot_hit,
                                                          std::vector<DTPrimitives> input_paths,
                                                          DTPrimitives hits_per_cell,
                                                          DTPrimitives &hits_in_trapezoid) {
  std::vector<DTPrimitives> output_paths;
  for (auto &hit : hits_per_cell) {
    int hit_bx = hit.tdcTimeStamp() / LHC_CLK_FREQ;
    int pivot_hit_bx = pivot_hit.tdcTimeStamp() / LHC_CLK_FREQ;
    if (hitTimeSort(pivot_hit, hit) || (pivot_hit_bx / BX_PER_FRAME) - (hit_bx / BX_PER_FRAME) > MAX_FRAME_DIF)
      continue;
    // limit the number of hits in the trapezoid to PATHFINDER_INPUT_HITS_LIMIT
    if (std::find(hits_in_trapezoid.begin(), hits_in_trapezoid.end(), hit) == hits_in_trapezoid.end())
      hits_in_trapezoid.push_back(hit);

    if (hits_in_trapezoid.size() > PATHFINDER_INPUT_HITS_LIMIT) {
      std::vector<DTPrimitives> empty_paths;
      return empty_paths;
    }

    for (auto &input_path : input_paths) {
      auto tmp_path = input_path;
      tmp_path.push_back(hit);
      output_paths.push_back(tmp_path);
    }
  }
  if (output_paths.empty())
    return input_paths;
  else
    return output_paths;
}
