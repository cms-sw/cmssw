#ifndef HLT1CaloJetEnergy_h
#define HLT1CaloJetEnergy_h

/** \class HLT1CaloJetEnergy
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single jet requirement with an Energy threshold (not Et!)
 *  Based on HLTSinglet
 *
 *  $Date: 2011/02/11 20:55:23 $
 *  $Revision: 1.5 $
 *
 *  \author Jim Brooke
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

namespace edm {
   class ConfigurationDescriptions;
}

//
// class declaration
//

class HLT1CaloJetEnergy : public HLTFilter {

   public:

      explicit HLT1CaloJetEnergy(const edm::ParameterSet&);
      ~HLT1CaloJetEnergy();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      bool saveTags_;           // whether to save this tag
      double min_E_;           // energy threshold in GeV 
      double max_Eta_;         // maximum eta
      int min_N_;              // minimum number

};

#endif //HLT1CaloJetEnergy_h
