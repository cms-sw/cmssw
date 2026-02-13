#include "FWCore/ParameterSet/interface/DescriptionCloner.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  namespace {
    constexpr char const kTrackedPrefix = '+';
    constexpr char const kUntrackedPrefix = '-';
    void insertInto(std::string_view iFullPathName,
                    std::shared_ptr<DescriptionCloner::EntryTypeBase> entry,
                    std::vector<DescriptionCloner::WhichEntry>& entries) {
      auto pos = iFullPathName.find('.');
      if (pos == std::string_view::npos) {
        //this is a leaf entry
        DescriptionCloner::WhichEntry whichEntry;
        whichEntry.label = std::string(iFullPathName);
        whichEntry.entry = std::move(entry);
        auto it = std::lower_bound(entries.begin(),
                                   entries.end(),
                                   whichEntry,
                                   [](DescriptionCloner::WhichEntry const& a, DescriptionCloner::WhichEntry const& b) {
                                     return a.label < b.label;
                                   });
        entries.emplace(it, std::move(whichEntry));
      } else {
        //this is a PSet
        std::string_view label = iFullPathName.substr(0, pos);
        std::string_view remainingPath = iFullPathName.substr(pos + 1);
        //find or create the WhichEntry for this label
        DescriptionCloner::WhichEntry* foundEntry = nullptr;
        auto it = std::lower_bound(
            entries.begin(), entries.end(), label, [](DescriptionCloner::WhichEntry const& e, std::string_view label) {
              return e.label < label;
            });
        if (it == entries.end() or it->label != label) {
          DescriptionCloner::WhichEntry whichEntry;
          whichEntry.label = std::string(label);
          whichEntry.entry = std::vector<DescriptionCloner::WhichEntry>{};
          auto index = it - entries.begin();
          entries.emplace(it, std::move(whichEntry));
          foundEntry = &entries[index];
        } else {
          foundEntry = &(*it);
        }
        //recurse
        auto& subEntries = std::get<std::vector<DescriptionCloner::WhichEntry>>(foundEntry->entry);
        insertInto(remainingPath, std::move(entry), subEntries);
      }
    }
    void setTrackiness(std::string const& iPath,
                       std::vector<DescriptionCloner::WhichEntry>& whichEntry,
                       const ParameterSetDescription& defaultDesc) {
      for (auto& entry : whichEntry) {
        auto label = entry.label;
        assert(label.empty() == false);
        std::string fullPath = iPath.empty() ? "" : iPath + ".";
        if (label[0] == kTrackedPrefix) {
          label = label.substr(1);
          entry.isTracked = true;
          entry.label = label;
          fullPath += label;
        } else if (label[0] == kUntrackedPrefix) {
          label = label.substr(1);
          entry.isTracked = false;
          entry.label = label;
          fullPath += label;
        } else {
          fullPath += label;
          auto trackiness = defaultDesc.trackiness(fullPath);
          if (trackiness == cfi::Trackiness::kNotAllowed) {
            throw edm::Exception(edm::errors::Configuration)
                << "DescriptionCloner::determineTrackinessFromDefaultDescription\n"
                << "The parameter \"" << fullPath << "\" does not exist in the default description.\n";
          }
          if (trackiness == cfi::Trackiness::kUnknown) {
            throw edm::Exception(edm::errors::Configuration)
                << "DescriptionCloner::determineTrackinessFromDefaultDescription\n"
                << "The parameter \"" << fullPath << "\" has unknown trackiness in the default description.\n";
          }
          entry.isTracked = (trackiness == cfi::Trackiness::kTracked);
        }
        if (std::holds_alternative<std::vector<DescriptionCloner::WhichEntry>>(entry.entry)) {
          auto& subEntries = std::get<std::vector<DescriptionCloner::WhichEntry>>(entry.entry);
          setTrackiness(fullPath, subEntries, defaultDesc);
        }
      }
    }

    ParameterSetDescription createDifferenceFromEntries(const std::vector<DescriptionCloner::WhichEntry>& entries) {
      ParameterSetDescription diffDesc;
      for (auto const& whichEntry : entries) {
        if (std::holds_alternative<std::vector<DescriptionCloner::WhichEntry>>(whichEntry.entry)) {
          //this is a PSet
          auto const& subEntry = std::get<std::vector<DescriptionCloner::WhichEntry>>(whichEntry.entry);
          ParameterSetDescription subDesc = createDifferenceFromEntries(subEntry);
          if (whichEntry.isTracked) {
            diffDesc.add(whichEntry.label, subDesc);
          } else {
            diffDesc.addUntracked(whichEntry.label, subDesc);
          }
        } else if (std::holds_alternative<std::shared_ptr<DescriptionCloner::EntryTypeBase>>(whichEntry.entry)) {
          auto const& entryPtr = std::get<std::shared_ptr<DescriptionCloner::EntryTypeBase>>(whichEntry.entry);
          entryPtr->addTo(diffDesc, whichEntry.label, whichEntry.isTracked);
        }
      }
      return diffDesc;
    }

    class NoneDescriptionNode : public edm::ParameterDescriptionNode {
    public:
      explicit NoneDescriptionNode(std::string iLabel) : label_(std::move(iLabel)) {}
      ~NoneDescriptionNode() override = default;

      ParameterDescriptionNode* clone() const final { return new NoneDescriptionNode(label_); }

    private:
      std::string label_;
      void checkAndGetLabelsAndTypes_(std::set<std::string>& usedLabels,
                                      std::set<ParameterTypes>& parameterTypes,
                                      std::set<ParameterTypes>& wildcardTypes) const final {}

      void validate_(ParameterSet& pset, std::set<std::string>& validatedLabels, Modifier modifier) const final {
        //do nothing, this node is used to indicate that a parameter should be omitted from the generated cfi file
      }

      void writeCfi_(std::ostream& os,
                     Modifier modifier,
                     bool& startWithComma,
                     int indentation,
                     CfiOptions&,
                     bool& wroteSomething) const final {
        wroteSomething = true;
        if (startWithComma)
          os << ",";
        startWithComma = true;
        os << "\n";
        printSpaces(os, indentation);

        os << label_ << " = None";
      }

      void print_(std::ostream& os, Modifier /*modifier*/, bool /*writeToCfi*/, edm::DocFormatHelper& dfh) const final {
        if (dfh.pass() == 0) {
          dfh.setAtLeast1(label_.size());
        } else {
          if (dfh.brief()) {
            std::ios::fmtflags oldFlags = os.flags();

            dfh.indent(os);
            os << std::left << std::setw(dfh.column1()) << label_ << " omitted\n";
            os.flags(oldFlags);
          } else {
            dfh.indent(os);
            os << label_ << "\n";
            dfh.indent2(os);
            os << "This parameter is omitted from the generated cfi file.\n";
          }
        }
      }

      cfi::Trackiness trackiness_(std::string_view path) const final { return cfi::Trackiness::kNotAllowed; }

      bool exists_(ParameterSet const& pset) const final { return false; }

      bool partiallyExists_(ParameterSet const& pset) const final { return false; }

      int howManyXORSubNodesExist_(ParameterSet const& pset) const final { return 0; }
    };
    struct EntryTypeOmit : public edm::DescriptionCloner::EntryTypeBase {
      void addTo(edm::ParameterSetDescription& iDesc, std::string_view iLabel, bool isTracked) const final {
        iDesc.addNode(std::make_unique<NoneDescriptionNode>(std::string(iLabel)));
      }
    };

  }  // namespace

  void DescriptionCloner::insert(std::string_view fullPathName, std::shared_ptr<EntryTypeBase> entry) {
    insertInto(fullPathName, std::move(entry), entries_);
  }

  void DescriptionCloner::omit(std::string_view fullPathName) {
    insert(fullPathName, std::make_shared<EntryTypeOmit>());
  }

  ParameterSetDescription DescriptionCloner::createDifference() const { return createDifferenceFromEntries(entries_); }

  void DescriptionCloner::determineTrackinessFromDefaultDescription(const ParameterSetDescription& defaultDesc) {
    setTrackiness("", entries_, defaultDesc);
  }

}  // namespace edm
