#ifndef HLTEgammaEtFilterPairs_h
#define HLTEgammaEtFilterPairs_h

/** \class HLTEgammaEtFilterPairs
 *
 *  \author Alessio Ghezzi 
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
//

class HLTEgammaEtFilterPairs : public HLTFilter {

   public:
      explicit HLTEgammaEtFilterPairs(const edm::ParameterSet&);
      ~HLTEgammaEtFilterPairs();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag inputTag_; // input tag identifying product contains egammas
      double etcutEB1_;           // Et threshold in GeV 
      double etcutEB2_;           // Et threshold in GeV 
      double etcutEE1_;           // Et threshold in GeV 
      double etcutEE2_;           // Et threshold in GeV 
      bool   store_;
      bool   relaxed_;
      edm::InputTag L1IsoCollTag_; 
      edm::InputTag L1NonIsoCollTag_; 
};

#endif //HLTEgammaEtFilterPairs_h
