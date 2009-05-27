#include "DataFormats/PatCandidates/interface/Lepton.h"

/// default constructor

using namespace pat;

Lepton::Lepton() :
  PATObject() {
}


//============ BEGIN ISOLATION BLOCK =====
/// Returns the isolation variable for a specifc key (or pseudo-key like CaloIso), or -1.0 if not available
float Lepton::isolation(IsolationKeys key) const { 
  if (key >= 0) {
    //if (key >= isolations_.size()) throw cms::Excepton("Missing Data") << "Isolation corresponding to key " << key << " was not stored for this particle.";
    if (size_t(key) >= isolations_.size()) return -1.0;
    return isolations_[key];
  } else switch (key) {
    case CaloIso:  
      //if (isolations_.size() <= HCalIso) throw cms::Excepton("Missing Data") << "CalIsoo Isolation was not stored for this particle.";
      if (isolations_.size() <= HCalIso) return -1.0; 
      return isolations_[ECalIso] + isolations_[HCalIso];
    default:
      return -1.0;
      //throw cms::Excepton("Missing Data") << "Isolation corresponding to key " << key << " was not stored for this particle.";
    }
}

/// Sets the isolation variable for a specifc key.
/// Note that you can't set isolation for a pseudo-key like CaloIso
void Lepton::setIsolation(IsolationKeys key, float value) {
  if (key >= 0) {
    if (size_t(key) >= isolations_.size()) isolations_.resize(key+1, -1.0);
    isolations_[key] = value;
  } else {
    throw cms::Exception("Illegal Argument") << 
      "The key for which you're setting isolation does not correspond " <<
      "to an individual isolation but to the sum of more independent isolations " <<
      "(e.g. Calo = ECal + HCal), so you can't SET the value, just GET it.\n" <<
      "Please set up each component independly.\n";
  }
}


//============ BEGIN ISODEPOSIT BLOCK =====
/// Returns the IsoDeposit associated with some key, or a null pointer if it is not available
const IsoDeposit * Lepton::isoDeposit(IsolationKeys key) const {
  for (IsoDepositPairs::const_iterator it = isoDeposits_.begin(), ed = isoDeposits_.end(); 
       it != ed; ++it) 
    {
      if (it->first == key) return & it->second;
    }
  return 0;
} 

/// Sets the IsoDeposit associated with some key; if it is already existent, it is overwritten.
void Lepton::setIsoDeposit(IsolationKeys key, const IsoDeposit &dep) {
  IsoDepositPairs::iterator it = isoDeposits_.begin(), ed = isoDeposits_.end();
  for (; it != ed; ++it) {
    if (it->first == key) { it->second = dep; return; }
  }
  isoDeposits_.push_back(std::make_pair(key,dep));
} 
