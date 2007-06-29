#ifndef CkfTrackCandidateMaker_h
#define CkfTrackCandidateMaker_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/CkfPattern/interface/TrackerTrajectoryBuilder.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/DetLayers/interface/NavigationSetter.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "RecoTracker/CkfPattern/interface/RedundantSeedCleaner.h"
#include "RecoTracker/CkfPattern/interface/CkfTrackCandidateMakerBase.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

class TransientInitialStateEstimator;

namespace cms
{
  class CkfTrackCandidateMaker : public CkfTrackCandidateMakerBase, public edm::EDProducer
  {
  public:

    explicit CkfTrackCandidateMaker(const edm::ParameterSet& conf):
      CkfTrackCandidateMakerBase(conf){
      produces<TrackCandidateCollection>();
    }

    virtual ~CkfTrackCandidateMaker(){;}

    virtual void beginJob (edm::EventSetup const & es){beginJobBase(es);}

    virtual void produce(edm::Event& e, const edm::EventSetup& es){produceBase(e,es);}
  };
}

#endif
