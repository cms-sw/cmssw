#include "AnalysisDataFormats/TopObjects/interface/TtEvent.h"

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
      { "kGeom",          kGeom          },
      { "kWMassMaxSumPt", kWMassMaxSumPt },
      { "kMaxSumPtWMass", kMaxSumPtWMass },
      { "kGenMatch",      kGenMatch      },
      { "kMVADisc",       kMVADisc       },
      { "kKinFit",        kKinFit        },
      { "kKinSolution",   kKinSolution   },
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
