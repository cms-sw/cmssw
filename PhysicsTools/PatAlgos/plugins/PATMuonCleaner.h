#ifndef PhysicsTools_PatAlgos_PATMuonCleaner_h
#define PhysicsTools_PatAlgos_PATMuonCleaner_h
//
// $Id: PATMuonCleaner.h,v 1.2 2008/03/11 11:01:08 llista Exp $
//

/**
  \class    pat::PATMuonCleaner PATMuonCleaner.h "PhysicsTools/PatAlgos/interface/PATMuonCleaner.h"
  \brief    Produces a clean list of muons

  The PATMuonCleaner produces a list of clean muons with associated back-references 
  to the original muon collection.

  The muon selection is based on reconstruction, custom selection or muon
  identification algorithms (to be implemented). It is steered by the configuration 
  parameters:

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
 
  The actual selection is performed by the MuonSelector.

  \author   Giovanni Petrucciani (from PATMuonProducer by Steven Lowette, Roger Wolf)
  \version  $Id: PATMuonCleaner.h,v 1.2 2008/03/11 11:01:08 llista Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

#include "PhysicsTools/PatUtils/interface/MuonSelector.h"

#include <string>

namespace pat {

  class PATMuonCleaner : public edm::EDProducer {

    public:

      explicit PATMuonCleaner(const edm::ParameterSet & iConfig);
      ~PATMuonCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void endJob();

    private:
      // configurables
      edm::InputTag muonSrc_;
      pat::helper::CleanerHelper< reco::Muon, 
                                  reco::Muon,
                                  reco::MuonCollection, 
                                  GreaterByPt<reco::Muon> > helper_;

      edm::ParameterSet selectionCfg_; ///< Defines everything about the selection
      MuonSelector      selector_;     ///< Actually performs the selection

  };

}

namespace reco {
  namespace modules {
    /// Helper struct to convert from ParameterSet to MuonSelection
    template<> 
    struct ParameterAdapter<pat::MuonSelector> { 
      static pat::MuonSelector make(const edm::ParameterSet & cfg) {
        pat::MuonSelection config_;
        const std::string& selectionType = cfg.getParameter<std::string>("type");
        config_.selectionType = selectionType;
        if ( selectionType == "custom" )
          {
            config_.dPbyPmax  = cfg.getParameter<double>("dPbyPmax");
            config_.chi2max  = cfg.getParameter<double>("chi2max");
            config_.nHitsMin = cfg.getParameter<int>("nHitsMin");
          }
        return pat::MuonSelector( config_ );
      }
    };
  }
}

#endif
