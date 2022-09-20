#ifndef DataFormats_PatCandidates_PackedTriggerPrescales_h
#define DataFormats_PatCandidates_PackedTriggerPrescales_h

#include <cstring>
#include <type_traits>

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Common/interface/Ref.h"

namespace pat {

  class PackedTriggerPrescales {
  public:
    PackedTriggerPrescales() : triggerNames_(nullptr) {}
    PackedTriggerPrescales(const edm::Handle<edm::TriggerResults> &handle);
    ~PackedTriggerPrescales() = default;

    // get prescale by index
    //  method templated to force correct choice of output type
    //  (as part of deprecating integer types for trigger prescales)
    template <typename T = int>
    T getPrescaleForIndex(int index) const;

    // get prescale by name or name prefix (if setTriggerNames was called)
    //  method templated to force correct choice of output type
    //  (as part of deprecating integer types for trigger prescales)
    template <typename T = int>
    T getPrescaleForName(const std::string &name, bool prefixOnly = false) const;

    // return the TriggerResults associated with this
    const edm::TriggerResults &triggerResults() const { return *edm::getProduct<edm::TriggerResults>(triggerResults_); }

    // use this method first if you want to be able to access the prescales by name
    // you can get the TriggerNames from the TriggerResults and the Event (edm or fwlite)
    void setTriggerNames(const edm::TriggerNames &names) { triggerNames_ = &names; }

    // set that the trigger of given index has a given prescale
    void addPrescaledTrigger(int index, double prescale);

  protected:
    std::vector<double> prescaleValues_;
    edm::RefCore triggerResults_;
    const edm::TriggerNames *triggerNames_;
  };

  template <typename T>
  T PackedTriggerPrescales::getPrescaleForIndex(int index) const {
    static_assert(std::is_same_v<T, double>,
                  "\n\n\tPlease use getPrescaleForIndex<double> "
                  "(other types for trigger prescales are not supported anymore by PackedTriggerPrescales)");
    if (unsigned(index) >= triggerResults().size())
      throw cms::Exception("InvalidReference", "Index out of bounds");
    return prescaleValues_[index];
  }

  template <typename T>
  T PackedTriggerPrescales::getPrescaleForName(const std::string &name, bool prefixOnly) const {
    static_assert(std::is_same_v<T, double>,
                  "\n\n\tPlease use getPrescaleForName<double> "
                  "(other types for trigger prescales are not supported anymore by PackedTriggerPrescales)");
    if (triggerNames_ == nullptr)
      throw cms::Exception("LogicError", "getPrescaleForName called without having called setTriggerNames first");
    if (prefixOnly) {
      auto const &names = triggerNames_->triggerNames();
      if (name.empty())
        throw cms::Exception("EmptyName",
                             "getPrescaleForName called with invalid arguments (name is empty, prefixOnly=true");
      size_t siz = name.length() - 1;
      while (siz > 0 && (name[siz] == '*' || name[siz] == '\0'))
        siz--;
      for (unsigned int i = 0, n = names.size(); i < n; ++i) {
        if (strncmp(name.c_str(), names[i].c_str(), siz) == 0) {
          return getPrescaleForIndex<T>(i);
        }
      }
      throw cms::Exception("InvalidReference", "Index out of bounds");
    } else {
      int index = triggerNames_->triggerIndex(name);
      return getPrescaleForIndex<T>(index);
    }
  }

}  // namespace pat

#endif  // DataFormats_PatCandidates_PackedTriggerPrescales_h
