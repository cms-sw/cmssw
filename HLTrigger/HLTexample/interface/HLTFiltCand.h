#ifndef HLTFiltCand_h
#define HLTFiltCand_h

/** \class HLTFiltCand
 *
 *  
 *  This class is an HLTFilter (-> EDFilter) implementing a very basic
 *  HLT trigger acting on candidates, requiring a g/e/m/j tuple above
 *  pt cuts
 *
 *  $Date: 2006/08/14 15:26:43 $
 *  $Revision: 1.10 $
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
      edm::InputTag tausTag_;  // input tag identifying product containing taus
      edm::InputTag jetsTag_;  // input tag identifying product containing jets
      edm::InputTag metsTag_;  // input tag identifying product containing METs

      double phot_pt_;       // pt cut for photon
      double elec_pt_;       // pt cut for electron
      double muon_pt_;       // pt cut for muons
      double taus_pt_;       // pt cut for taus
      double jets_pt_;       // pt cut for jets
      double mets_pt_;       // pt cut for MET
};

#endif //HLTFiltCand_h
