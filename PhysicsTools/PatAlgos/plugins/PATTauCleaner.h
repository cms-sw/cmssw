//
// $Id: PATTauCleaner.h,v 1.2 2008/03/11 10:50:21 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATTauCleaner_h
#define PhysicsTools_PatAlgos_PATTauCleaner_h

/**
  \class    pat::PATTauCleaner PATTauCleaner.h "PhysicsTools/PatAlgos/interface/PATTauCleaner.h"
  \brief    Produces pat::Tau's

   The PATTauCleaner produces analysis-level pat::Tau's starting from
   a collection of objects of TauType.

  \author   Steven Lowette, Christophe Delaere
  \version  $Id: PATTauCleaner.h,v 1.2 2008/03/11 10:50:21 llista Exp $
*/


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"
#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminatorByIsolation.h"
#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"

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

  typedef PATTauCleaner<reco::PFTau,reco::PFTau,reco::PFTauDiscriminatorByIsolation>       PATPFTauCleaner;
  typedef PATTauCleaner<reco::CaloTau,reco::CaloTau,reco::CaloTauDiscriminatorByIsolation> PATCaloTauCleaner;
  typedef PATTauCleaner<reco::PFTau,reco::BaseTau,reco::PFTauDiscriminatorByIsolation>     PATPF2BaseTauCleaner;
  typedef PATTauCleaner<reco::CaloTau,reco::BaseTau,reco::CaloTauDiscriminatorByIsolation> PATCalo2BaseTauCleaner;


}

#endif
