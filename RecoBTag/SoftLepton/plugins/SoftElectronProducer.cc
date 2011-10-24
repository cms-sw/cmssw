
#include <vector>
#include <iostream>
#include <boost/regex.hpp>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"

#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"

#include "ElectronIdMLP.h"
#include "SoftElectronProducer.h"

//------------------------------------------------------------------------------

using namespace std;
using namespace edm;

//------------------------------------------------------------------------------

SoftElectronProducer::SoftElectronProducer(const edm::ParameterSet &iConf) :
  theConf(iConf), theTrackAssociator(0), theElecNN(0)
{
  theTrackTag             = theConf.getParameter<InputTag>("TrackTag");

  theHBHERecHitTag        = theConf.getParameter<InputTag>("HBHERecHitTag");
  theBasicClusterTag      = theConf.getParameter<InputTag>("BasicClusterTag");
 // theBasicClusterShapeTag = theConf.getParameter<InputTag>("BasicClusterShapeTag");

  theHOverEConeSize = theConf.getParameter<double>("HOverEConeSize");

  barrelRecHitCollection_ = theConf.getParameter<edm::InputTag>("BarrelRecHitCollection");
  endcapRecHitCollection_ = theConf.getParameter<edm::InputTag>("EndcapRecHitCollection");

  // TrackAssociator and its parameters
  theTrackAssociator = new TrackDetectorAssociator();
  theTrackAssociator->useDefaultPropagator();
  edm::ParameterSet parameters = iConf.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  theTrackAssociatorParameters = new TrackAssociatorParameters( parameters );

  theDiscriminatorCut = theConf.getParameter<double>("DiscriminatorCut");

  theElecNN = new ElectronIdMLP;

  // register the product
  produces<reco::ElectronCollection>();
}

//------------------------------------------------------------------------------

SoftElectronProducer::~SoftElectronProducer()
{
  if (theElecNN)                    delete theElecNN;
  if (theTrackAssociator)           delete theTrackAssociator;
  if (theTrackAssociatorParameters) delete theTrackAssociatorParameters;
}

//------------------------------------------------------------------------------

void SoftElectronProducer::produce(edm::Event &iEvent,
                                      const edm::EventSetup &iSetup)
{

  auto_ptr<reco::ElectronCollection> candidates(new reco::ElectronCollection());

  Handle<reco::TrackCollection> handleTrack;
  reco::TrackCollection::const_iterator itTrack;

  Handle<reco::BasicClusterCollection> handleCluster;
  reco::BasicClusterCollection::const_iterator itCluster;

  //Handle<reco::ClusterShapeCollection> handleShape;
  //reco::ClusterShapeCollection::const_iterator itShape;

  Handle<HBHERecHitCollection> handleRecHit;
  CaloRecHitMetaCollectionV::const_iterator itRecHit;

  ESHandle<CaloGeometry> handleCaloGeom;
  

  double hcalEnergy, clusEnergy, trkP;
  double x[2], y[2], z[2];
  double covEtaEta, covEtaPhi, covPhiPhi, emFraction, deltaE;
  double eMax, e2x2, e3x3, e5x5, v1, v2, v3, v4;
  double value, dist, distMin;

  const reco::BasicCluster *matchedCluster;
  //const reco::ClusterShape *matchedShape;
  const reco::Track *track;

  // get basic clusters
  iEvent.getByLabel(theBasicClusterTag, handleCluster);

  // get basic cluster shapes
  //iEvent.getByLabel(theBasicClusterShapeTag, handleShape);

  // get rec. hits
  iEvent.getByLabel(theHBHERecHitTag, handleRecHit);
  HBHERecHitMetaCollection metaRecHit(*handleRecHit);

  //only barrel is used, giving twice the same inputag..


  EcalClusterLazyTools ecalTool(iEvent, iSetup, barrelRecHitCollection_, endcapRecHitCollection_ );

  // get calorimeter geometry
  iSetup.get<CaloGeometryRecord>().get(handleCaloGeom);

  CaloConeSelector selectorRecHit(theHOverEConeSize, handleCaloGeom.product(), DetId::Hcal);

  // get tracks
  iEvent.getByLabel(theTrackTag, handleTrack);

  FreeTrajectoryState tmpFTS;
  TrackDetMatchInfo info;
  unsigned int counterTrack;

  // loop over tracks
  for(itTrack = handleTrack->begin(), counterTrack = 0;
      itTrack != handleTrack->end();
      ++itTrack, ++counterTrack)
  {
    track = &(*itTrack);

    try {
      tmpFTS = theTrackAssociator->getFreeTrajectoryState(iSetup, *track);
      info = theTrackAssociator->associate(iEvent, iSetup, tmpFTS, *theTrackAssociatorParameters);
    } catch (cms::Exception e) {
      // extrapolation failed, skip this track
      std::cerr << "Caught exception during track extrapolation: " << e.what() << ". Skipping track" << endl;
      continue;
    }

    x[0] = info.trkGlobPosAtEcal.x();
    y[0] = info.trkGlobPosAtEcal.y();
    z[0] = info.trkGlobPosAtEcal.z();

    // analyse only tracks passing quality cuts
    if(track->numberOfValidHits() >= 8 && track->pt() > 2.0 &&
       abs(track->eta()) < 1.2 && info.isGoodEcal)
    {
      distMin = 1.0e6;
      matchedCluster = 0;
//      matchedShape = 0;

      // loop over basic clusters
      for(itCluster = handleCluster->begin(); itCluster != handleCluster->end();++itCluster)
      {
        x[1] = itCluster->x();
        y[1] = itCluster->y();
        z[1] = itCluster->z();

        dist = hypot(x[0] - x[1], y[0] - y[1]);
        dist = hypot(dist, z[0] - z[1]);

        if(dist < distMin)
        {
          distMin = dist;
          matchedCluster = &(*itCluster);
        }
      }

      // identify electrons based on cluster properties
      if(matchedCluster  && distMin < 10.0)
      {
        GlobalPoint position(matchedCluster->x(), matchedCluster->y(), matchedCluster->z());
        auto_ptr<CaloRecHitMetaCollectionV> chosen = selectorRecHit.select(position, metaRecHit);
        hcalEnergy = 0.0;
        for(itRecHit = chosen->begin(); itRecHit != chosen->end(); ++itRecHit)
        {
          hcalEnergy += itRecHit->energy();
        }

        clusEnergy = matchedCluster->energy();
        trkP = track->p();

        deltaE = (clusEnergy - trkP)/(clusEnergy + trkP);
        emFraction =  clusEnergy/(clusEnergy + hcalEnergy);

        eMax = ecalTool.eMax(*matchedCluster);
        e2x2 = ecalTool.e2x2(*matchedCluster);
        e3x3 = ecalTool.e3x3(*matchedCluster);
        e5x5 = ecalTool.e5x5(*matchedCluster);
        v1 = eMax/e3x3;
        v2 = eMax/e2x2;
        v3 = e2x2/e5x5;
        v4 = ((e5x5 - eMax) < 0.001) ? 1.0 : (e3x3 - eMax)/(e5x5 - eMax);
        std::vector<float> cov = ecalTool.covariances(*matchedCluster); 
        covEtaEta = cov[0];
        covEtaPhi = cov[1];
        covPhiPhi = cov[2];

        value = theElecNN->value(0, covEtaEta, covEtaPhi, covPhiPhi,
                                 v1, v2, v3, v4, emFraction, deltaE);
        if (value > theDiscriminatorCut)
        {
          const reco::Particle::LorentzVector  p4(0.0, 0.0, 0.0, clusEnergy);
          const reco::Particle::Point vtx(0.0, 0.0, 0.0);
          reco::Electron newCandidate(0, p4, vtx);
          reco::TrackRef refTrack(handleTrack, counterTrack);
          newCandidate.setTrack(refTrack);
          candidates->push_back(newCandidate);
        }
      }
    }
  }
  
  // put the product in the event
  iEvent.put(candidates);

}
