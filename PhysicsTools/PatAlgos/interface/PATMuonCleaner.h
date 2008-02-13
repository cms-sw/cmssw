#ifndef PhysicsTools_PatAlgos_PATMuonCleaner_h
#define PhysicsTools_PatAlgos_PATMuonCleaner_h
//
// $Id: PATMuonCleaner.h,v 1.4 2008/02/07 15:48:53 fronga Exp $
//

/**
  \class    PATMuonCleaner PATMuonCleaner.h "PhysicsTools/PatAlgos/interface/PATMuonCleaner.h"
  \brief    Produces a clean list of muons, and associated back-references to the original muon collection

   The PATMuonCleaner produces a list of clean muons with associated back-references to the original muon collection.

   The muon selection is based on reconstruction, custom selection or (in the future) muon
   identification algorithms. It is steered by the configuration parameters:

\code
PSet selection = {
   string type = "none | globalMuons | muId | custom" // muId not implemented yet
   [ // If custom, give cut values
     double dPbyPmax = ...
     double chi2max  = ...
     int    nHitsMin = ...
   ]
}
\endcode
 
  \author   Giovanni Petrucciani (from PATMuonProducer by Steven Lowette, Roger Wolf)
  \version  $Id: PATMuonCleaner.h,v 1.4 2008/02/07 15:48:53 fronga Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/interface/CleanerHelper.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include "PhysicsTools/PatUtils/interface/MuonSelector.h"

#include <string>

namespace pat {

  class PATMuonCleaner : public edm::EDProducer {

    public:

      explicit PATMuonCleaner(const edm::ParameterSet & iConfig);
      ~PATMuonCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);

    private:
      // configurables
      edm::InputTag muonSrc_;
      pat::helper::CleanerHelper< reco::Muon, 
                                  reco::Muon,
                                  reco::MuonCollection, 
                                  GreaterByPt<reco::Muon> > helper_;

      edm::ParameterSet selectionCfg_;       ///< Defines everything about the selection
      std::auto_ptr<MuonSelector> selector_; ///< Actually performs the selection
  };


}

#endif
