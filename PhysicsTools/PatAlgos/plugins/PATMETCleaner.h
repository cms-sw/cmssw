//
// $Id: PATMETCleaner.h,v 1.2 2008/03/11 11:02:43 llista Exp $
//

#ifndef PhysicsTools_PatAlgos_PATMETCleaner_h
#define PhysicsTools_PatAlgos_PATMETCleaner_h

/**
  \class    pat::PATMETCleaner PATMETCleaner.h "PhysicsTools/PatAlgos/interface/PATMETCleaner.h"
  \brief    Produces pat::MET's

   The PATMETCleaner produces analysis-level pat::MET's starting from
   a collection of objects of METType.

  \author   Steven Lowette, Jeremy Andrea
  \version  $Id: PATMETCleaner.h,v 1.2 2008/03/11 11:02:43 llista Exp $
*/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "PhysicsTools/PatAlgos/plugins/CleanerHelper.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMET.h"

#include "PhysicsTools/Utilities/interface/EtComparator.h"


namespace pat {

  // templated. One day we might have PFMET...
  template<typename METIn, typename METOut>
  class PATMETCleaner : public edm::EDProducer {

    public:

      explicit PATMETCleaner(const edm::ParameterSet & iConfig);
      ~PATMETCleaner();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
      virtual void endJob();

    private:
      // configurables
      edm::InputTag            metSrc_;
      // helper
      pat::helper::CleanerHelper<METIn,
                                 METOut,
                                 std::vector<METOut>,
                                 GreaterByEt<METOut> > helper_;

  };

  // now I'm typedeffing eveything, but I don't think we really need all them
  typedef PATMETCleaner<reco::MET,reco::MET>         PATBaseMETCleaner;
  typedef PATMETCleaner<reco::CaloMET,reco::CaloMET> PATCaloMETCleaner;
  typedef PATMETCleaner<reco::CaloMET,reco::MET>     PATCalo2BaseMETCleaner; // nor this

}

#endif
