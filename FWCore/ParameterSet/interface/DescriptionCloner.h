#ifndef FWCore_ParameterSet_DescriptionCloner_h
#define FWCore_ParameterSet_DescriptionCloner_h
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include <string_view>
#include <variant>
#include <memory>

namespace edm {
  class DescriptionCloner {
  public:
    DescriptionCloner() = default;
    ~DescriptionCloner() = default;

    // Handles injecting to proper node type into the ParameterSetDescripion
    struct EntryTypeBase {
      virtual ~EntryTypeBase() = default;
      virtual void addTo(edm::ParameterSetDescription&, std::string_view iLabel, bool isTracked) const = 0;
    };
    template <typename T>
    struct EntryType : public EntryTypeBase {
      EntryType(T const& iDefault) : defaultValue{iDefault} {}
      void addTo(edm::ParameterSetDescription& iDesc, std::string_view iLabel, bool isTracked) const final {
        if (isTracked) {
          iDesc.add<T>(std::string(iLabel), defaultValue);
        } else {
          iDesc.addUntracked<T>(std::string(iLabel), defaultValue);
        }
      }
      T defaultValue;
    };

    /** The vector<WhichEntry> in `entry` is used to represent a PSet with that label and trackiness. The values in
     * that vector represent the parameters within that PSet. The EntryTypeBase is used to represent the actual 
     * parameter to be modified which has that label and trackiness.
    */
    struct WhichEntry {
      std::string label;
      std::variant<std::vector<WhichEntry>, std::shared_ptr<EntryTypeBase>> entry;
      bool isTracked;
    };

    template <typename T>
    void set(std::string_view fullPathName, T const& value) {
      insert(fullPathName, std::make_shared<EntryType<T>>(value));
    }

    //Should only be called by ConfigurationDescriptions
    void determineTrackinessFromDefaultDescription(const ParameterSetDescription& defaultDesc);
    ParameterSetDescription createDifference() const;

  private:
    void insert(std::string_view fullPathName, std::shared_ptr<EntryTypeBase> entry);
    std::vector<WhichEntry> entries_;
  };
}  // namespace edm
#endif
