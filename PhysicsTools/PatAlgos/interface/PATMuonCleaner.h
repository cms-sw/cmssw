#ifndef PhysicsTools_PatAlgos_PATMuonCleaner_h
#define PhysicsTools_PatAlgos_PATMuonCleaner_h
//
// $Id: PATMuonCleaner.h,v 1.3 2008/01/16 16:04:37 gpetrucc Exp $
//

/**
  \class    PATMuonCleaner PATMuonCleaner.h "PhysicsTools/PatAlgos/interface/PATMuonCleaner.h"
  \brief    Produces a clean list of muons, and associated back-references to the original muon collection

   The PATMuonCleaner produces a list of clean muons with associated back-references to the original muon collection.

   Muon selection is performed using the standard muon ID helper functions.
   It can be configured in the following way:

   PSet selection = {
   string type = "none | globalMuons | muId | custom" // muId not implemented yet
   [ // If custom, give cut values
     double dPbyPmax = ...
     double chi2max  = ...
     int    nHitsMin = ...
   ]
   }
 
  \author   Giovanni Petrucciani (from PATMuonProducer by Steven Lowette, Roger Wolf, 
  \version  $Id: PATMuonCleaner.h,v 1.3 2008/01/16 16:04:37 gpetrucc Exp $
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
