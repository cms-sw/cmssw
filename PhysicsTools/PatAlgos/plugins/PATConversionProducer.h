//
//

#ifndef PhysicsTools_PatAlgos_PATConversionProducer_h
#define PhysicsTools_PatAlgos_PATConversionProducer_h

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "PhysicsTools/PatAlgos/interface/MultiIsolator.h"
#include "PhysicsTools/PatAlgos/interface/EfficiencyLoader.h"
#include "PhysicsTools/PatAlgos/interface/KinResolutionsLoader.h"

#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/PatCandidates/interface/UserData.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <string>


namespace pat {


  class PATConversionProducer : public edm::global::EDProducer<> {

    public:

      explicit PATConversionProducer(const edm::ParameterSet & iConfig);
      ~PATConversionProducer();

      virtual void produce(edm::StreamID, edm::Event & iEvent, const edm::EventSetup& iSetup) const override;

    private:

      // configurables
      const edm::EDGetTokenT<edm::View<reco::GsfElectron> > electronToken_;
      const edm::EDGetTokenT<reco::BeamSpot>                bsToken_;
      const edm::EDGetTokenT<reco::ConversionCollection>    conversionsToken_;

  };


}

#endif
