//
// $Id: PATConversionProducer.h,v 1.2 2013/02/27 23:26:56 wmtan Exp $
//

#ifndef PhysicsTools_PatAlgos_PATConversionProducer_h
#define PhysicsTools_PatAlgos_PATConversionProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
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


  class PATConversionProducer : public edm::EDProducer {

    public:

      explicit PATConversionProducer(const edm::ParameterSet & iConfig);
      ~PATConversionProducer();

      virtual void produce(edm::Event & iEvent, const edm::EventSetup& iSetup) override;

    private:

      // configurables
      edm::InputTag electronSrc_;

  };


}

#endif
