#ifndef HLTFiltCand_h
#define HLTFiltCand_h

/** \class HLTFiltCand
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on candidates, requiring a g/e/m/j tuple above
 *  pt cuts
 *
 *  $Date: 2006/06/25 19:03:02 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTFilter.h"

//
// class decleration
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
      edm::InputTag jetsTag_;  // input tag identifying product containing jets

      double phot_pt_;       // pt cut for photon
      double elec_pt_;       // pt cut for electron
      double muon_pt_;       // pt cut for muons
      double jets_pt_;       // pt cut for jets
};

#endif //HLTFiltCand_h
