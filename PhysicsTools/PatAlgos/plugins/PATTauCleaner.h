//
// $Id: PATTauCleaner.h,v 1.4 2008/10/16 13:15:38 veelken Exp $
//

#ifndef PhysicsTools_PatAlgos_PATTauCleaner_h
#define PhysicsTools_PatAlgos_PATTauCleaner_h

/**
  \class    pat::PATTauCleaner PATTauCleaner.h "PhysicsTools/PatAlgos/interface/PATTauCleaner.h"
  \brief    Produces pat::Tau's

   The PATTauCleaner produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
  \version  $Id: PATTauCleaner.h,v 1.4 2008/10/16 13:15:38 veelken Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
//#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminator.h"

#include "PhysicsTools/Utilities/interface/PtComparator.h"

#include <string>


namespace pat {


  template<typename TauIn, typename TauOut, typename TauTag>
  class PATTauCleaner : public edm::EDProducer {

    public:

      explicit PATTauCleaner(const edm::ParameterSet & iConfig);
      ~PATTauCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void endJob();

    private:
      // configurables
      edm::InputTag tauSrc_;
      edm::InputTag tauDiscSrc_;
      // main helper
      pat::helper::CleanerHelper<TauIn,
                                 TauOut,
                                 std::vector<TauOut>,
                                 GreaterByPt<TauOut> > helper_;

      // deltaR overlap helper
      pat::helper::OverlapHelper overlapHelper_;
  };

  typedef PATTauCleaner<reco::PFTau,reco::PFTau,reco::PFTauDiscriminator>       PATPFTauCleaner;
  typedef PATTauCleaner<reco::CaloTau,reco::CaloTau,reco::CaloTauDiscriminator> PATCaloTauCleaner;

}

#endif
