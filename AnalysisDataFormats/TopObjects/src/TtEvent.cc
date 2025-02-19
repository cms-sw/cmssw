#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <cstring>

// find corresponding hypotheses based on JetLepComb
int
TtEvent::correspondingHypo(const HypoClassKey& key1, const unsigned& hyp1, const HypoClassKey& key2) const
{
  for(unsigned hyp2 = 0; hyp2 < this->numberOfAvailableHypos(key2); ++hyp2) {
    if( this->jetLeptonCombination(key1, hyp1) == this->jetLeptonCombination(key2, hyp2) )
      return hyp2;
  }
  return -1; // if no corresponding hypothesis was found
}

// return the corresponding enum value from a string
TtEvent::HypoClassKey
TtEvent::hypoClassKeyFromString(const std::string& label) const 
{
   static HypoClassKeyStringToEnum hypoClassKeyStringToEnumMap[] = {
      { "kGeom",              kGeom             },
      { "kWMassMaxSumPt",     kWMassMaxSumPt    },
      { "kMaxSumPtWMass",     kMaxSumPtWMass    },
      { "kGenMatch",          kGenMatch         },
      { "kMVADisc",           kMVADisc          },
      { "kKinFit",            kKinFit           },
      { "kKinSolution",       kKinSolution      },
      { "kWMassDeltaTopMass", kWMassDeltaTopMass},
      { "kHitFit",            kHitFit           },
      { 0, (HypoClassKey)-1 }
   };

   bool found = false;
   HypoClassKey value = (HypoClassKey)-1;
   for(int i = 0; hypoClassKeyStringToEnumMap[i].label && (!found); ++i){
     if(!strcmp(label.c_str(), hypoClassKeyStringToEnumMap[i].label)){
       found = true;
       value = hypoClassKeyStringToEnumMap[i].value;
     }
   }

   // in case of unrecognized selection type
   if(!found){
     throw cms::Exception("TtEventError") << label << " is not a recognized HypoClassKey";
   }
   return value;
}

// print pt, eta, phi, mass of a given candidate into an existing LogInfo
void
TtEvent::printParticle(edm::LogInfo &log, const char* name, const reco::Candidate* cand) const
{
  if(!cand) {
    log << std::setw(15) << name << ": not available!\n";
    return;
  }
  log << std::setprecision(3) << setiosflags(std::ios::fixed | std::ios::showpoint);
  log << std::setw(15) << name         << ": "
      << std::setw( 7) << cand->pt()   << "; "
      << std::setw( 7) << cand->eta()  << "; "
      << std::setw( 7) << cand->phi()  << "; "
      << resetiosflags(std::ios::fixed | std::ios::showpoint) << setiosflags(std::ios::scientific)
      << std::setw(10) << cand->mass() << "\n";
  log << resetiosflags(std::ios::scientific);
}
