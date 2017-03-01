#include "DataFormats/PatCandidates/interface/PackedTriggerPrescales.h"
#include "DataFormats/Common/interface/RefProd.h"
#include <cstring>

pat::PackedTriggerPrescales::PackedTriggerPrescales(const edm::Handle<edm::TriggerResults> & handle) :
    prescaleValues_(), 
    triggerResults_(edm::RefProd<edm::TriggerResults>(handle).refCore()),
    triggerNames_(0)
{
    prescaleValues_.resize(handle->size(),0);
}

int pat::PackedTriggerPrescales::getPrescaleForIndex(int index) const {
    if (unsigned(index) >= triggerResults().size()) throw cms::Exception("InvalidReference", "Index out of bounds");
    return prescaleValues_[index];
}

int pat::PackedTriggerPrescales::getPrescaleForName(const std::string & name, bool prefixOnly) const {
    if (triggerNames_ == 0) throw cms::Exception("LogicError", "getPrescaleForName called without having called setTriggerNames first");
    if (prefixOnly) {
        const std::vector<std::string> &names = triggerNames_->triggerNames();
        size_t siz = name.length()-1;
        while (siz > 0 && (name[siz] == '*' || name[siz] == '\0')) siz--;
        for (unsigned int i = 0, n = names.size(); i < n; ++i) {
            if (strncmp(name.c_str(), names[i].c_str(), siz) == 0) {
                return getPrescaleForIndex(i);
            }
        }
        throw cms::Exception("InvalidReference", "Index out of bounds");
    } else {
        int index = triggerNames_->triggerIndex(name);
        return getPrescaleForIndex(index);
    }
}

void pat::PackedTriggerPrescales::addPrescaledTrigger(int index, int prescale) {
    if (unsigned(index) >= triggerResults().size()) throw cms::Exception("InvalidReference", "Index out of bounds");
    prescaleValues_[index] = prescale;
}

