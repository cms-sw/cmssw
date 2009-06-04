#ifndef HLT1CaloJetEnergy_h
#define HLT1CaloJetEnergy_h

/** \class HLT1CaloJetEnergy
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a
 *  single jet requirement with an Energy threshold (not Et!)
 *  Based on HLTSinglet
 *
 *  $Date: 2009/03/25 12:51:17 $
 *  $Revision: 1.3 $
 *
 *  \author Jim Brooke
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include<vector>

//
// class declaration
//

class HLT1CaloJetEnergy : public HLTFilter {

   public:

      explicit HLT1CaloJetEnergy(const edm::ParameterSet&);
      ~HLT1CaloJetEnergy();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product
      bool saveTag_;           // whether to save this tag
      double min_E_;           // energy threshold in GeV 
      double max_Eta_;         // maximum eta
      int min_N_;              // minimum number

};

#endif //HLT1CaloJetEnergy_h
