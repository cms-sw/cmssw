#ifndef HLTFiltCand_h
#define HLTFiltCand_h

/** \class HLTFiltCand
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on candidates, requiring a g/e/m/j tuple above
 *  pt cuts
 *
 *  $Date: 2007/04/13 15:57:57 $
 *  $Revision: 1.19 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class declaration
//

class HLTFiltCand : public HLTFilter {

   public:
      explicit HLTFiltCand(const edm::ParameterSet&);
      ~HLTFiltCand();
      virtual bool filter(edm::Event&, const edm::EventSetup&);

   private:
      edm::InputTag photTag_;  // input tag identifying product containing photons
      edm::InputTag elecTag_;  // input tag identifying product containing electrons
      edm::InputTag muonTag_;  // input tag identifying product containing muons
      edm::InputTag tausTag_;  // input tag identifying product containing taus
      edm::InputTag jetsTag_;  // input tag identifying product containing jets
      edm::InputTag metsTag_;  // input tag identifying product containing METs
      edm::InputTag trckTag_;  // input tag identifying product containing Tracks
      edm::InputTag ecalTag_;  // input tag identifying product containing SuperClusters

      double min_Pt_;          // min pt cut
};

#endif //HLTFiltCand_h
