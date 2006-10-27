// -*- C++ -*-
//
// Package:    EgammaElectronAlgos
// Class:      PixelMatchElectronAlgo.
// 
/**\class PixelMatchElectronAlgo EgammaElectronAlgos/PixelMatchElectronAlgo

 Description: top algorithm producing TrackCandidate and Electron objects from supercluster
              driven pixel seeded Ckf tracking

*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Thu july 6 13:22:06 CEST 2006
// $Id: PixelMatchElectronAlgo.cc,v 1.18 2006/10/27 15:04:31 uberthon Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"

//#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"

#include "Geometry/Vector/interface/GlobalPoint.h"
#include "Geometry/Vector/interface/GlobalVector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include <TMath.h>
#include <sstream>

#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

using namespace edm;
using namespace std;
using namespace reco;
//using namespace math; // conflicts with DataFormat/Math/interface/Point3D.h!!!!

PixelMatchElectronAlgo::PixelMatchElectronAlgo(double maxEOverPBarrel, double maxEOverPEndcaps, 
                                               double hOverEConeSize, double maxHOverE, 
                                               double maxDeltaEta, double maxDeltaPhi):  
 maxEOverPBarrel_(maxEOverPBarrel), maxEOverPEndcaps_(maxEOverPEndcaps), 
 hOverEConeSize_(hOverEConeSize), maxHOverE_(maxHOverE), 
 maxDeltaEta_(maxDeltaEta), maxDeltaPhi_(maxDeltaPhi), 
 theCkfTrajectoryBuilder(0), theTrajectoryCleaner(0),
 theInitialStateEstimator(0), theNavigationSchool(0) {}

PixelMatchElectronAlgo::~PixelMatchElectronAlgo() {

  delete theInitialStateEstimator;
  delete theNavigationSchool;
  delete theTrajectoryCleaner; 
    
}

void PixelMatchElectronAlgo::setupES(const edm::EventSetup& es, const edm::ParameterSet &conf) {

  //services
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
  es.get<IdealMagneticFieldRecord>().get(theMagField);

  // get calo geometry
   es.get<IdealGeometryRecord>().get(theCaloGeom);
  
  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
  theInitialStateEstimator       = new TransientInitialStateEstimator( es,tise_params);

  theNavigationSchool   = new SimpleNavigationSchool(&(*theGeomSearchTracker),&(*theMagField));

  // set the correct navigation
  NavigationSetter setter( *theNavigationSchool);

  //  theCkfTrajectoryBuilder = new CkfTrajectoryBuilder(conf,es,theMeasurementTracker);
  theTrajectoryCleaner = new TrajectoryCleanerBySharedHits();    
  std::string trajectoryBuilderName = conf.getParameter<std::string>("TrajectoryBuilder");
  edm::ESHandle<TrackerTrajectoryBuilder> theTrajectoryBuilderHandle;
  es.get<CkfComponentsRecord>().get(trajectoryBuilderName,theTrajectoryBuilderHandle);
  theCkfTrajectoryBuilder = theTrajectoryBuilderHandle.product();    

  trackBarrelLabel_ = conf.getParameter<string>("TrackBarrelLabel");
  trackBarrelInstanceName_ = conf.getParameter<string>("TrackBarrelProducer");
  trackEndcapLabel_ = conf.getParameter<string>("TrackEndcapLabel");
  trackEndcapInstanceName_ = conf.getParameter<string>("TrackEndcapProducer");
  assBarrelLabel_ = conf.getParameter<string>("SCLBarrelLabel");
  assBarrelInstanceName_ = conf.getParameter<string>("SCLBarrelProducer");
  assEndcapLabel_ = conf.getParameter<string>("SCLEndcapLabel");
  assEndcapInstanceName_ = conf.getParameter<string>("SCLEndcapProducer");
}

void  PixelMatchElectronAlgo::run(Event& e, PixelMatchGsfElectronCollection & outEle) {

  // get the input 
  edm::Handle<TrackCollection> tracksBarrelH;
  edm::Handle<TrackCollection> tracksEndcapH;
  // to check existance
  edm::Handle<HBHERecHitCollection> hbhe;
  if (hOverEConeSize_ > 0.) e.getByType(hbhe);
  HBHERecHitMetaCollection mhbhe(*hbhe);
  e.getByLabel(trackBarrelLabel_,trackBarrelInstanceName_,tracksBarrelH);
  e.getByLabel(trackEndcapLabel_,trackEndcapInstanceName_,tracksEndcapH);
  edm::Handle<SeedSuperClusterAssociationCollection> barrelH;
  edm::Handle<SeedSuperClusterAssociationCollection> endcapH;
  e.getByLabel(assBarrelLabel_,assBarrelInstanceName_,barrelH);
  e.getByLabel(assEndcapLabel_,assEndcapInstanceName_,endcapH);
  edm::LogInfo("") 
    <<"\n\n Treating "<<e.id()<<", Number of seeds Barrel:"
    <<barrelH.product()->size()<<" Number of seeds Endcap:"<<endcapH.product()->size();
  
  // create electrons from tracks in 2 steps: barrel + endcap
  const SeedSuperClusterAssociationCollection  *sclAss=&(*barrelH);
  process(tracksBarrelH,sclAss,mhbhe,outEle);
  sclAss=&(*endcapH);
  process(tracksEndcapH,sclAss,mhbhe,outEle);

  std::ostringstream str;

  str << "========== PixelMatchElectronAlgo Info ==========";
  str << "Event " << e.id();
  str << "Number of final electron tracks: " << tracksBarrelH.product()->size()+ tracksEndcapH.product()->size();
  str << "Number of final electrons: " << outEle.size();
  for (vector<PixelMatchGsfElectron>::const_iterator it = outEle.begin(); it != outEle.end(); it++) {
    str << "New electron with charge, pt, eta, phi : "  << it->charge() << " , " 
        << it->pt() << " , " << it->eta() << " , " << it->phi();
  }
 
  str << "=================================================";
  LogDebug("PixelMatchElectronAlgo") << str.str();
  return;
}

void PixelMatchElectronAlgo::process(edm::Handle<TrackCollection> tracksH, const SeedSuperClusterAssociationCollection *sclAss,
                                     HBHERecHitMetaCollection mhbhe, PixelMatchGsfElectronCollection & outEle) {
  const TrackCollection *tracks=tracksH.product();
  for (unsigned int i=0;i<tracks->size();++i) {
    const Track & t=(*tracks)[i];
    // look for corresponding seed
    //temporary as long as there is no way to have a pointer to the seed from the track
    edm::Ref<TrajectorySeedCollection> seed;
    bool found = false;
    for( SeedSuperClusterAssociationCollection::const_iterator it= sclAss->begin(); it != sclAss->end(); ++it) {
      seed=(*it).key;
      if (equal(seed,t)) {
	found=true;
	break;
      }
    }
    
    if (!found) {
      LogWarning("") <<" No seed corresponding to track was found!!";
      continue;
    }
    const SuperCluster theClus=*((*sclAss)[seed]);
    if (preSelection(theClus,t,mhbhe)) {
      LogInfo("")<<"Constructed new electron with energy  "<< (*sclAss)[seed]->energy();
      TSCPBuilderNoMaterial tscpBuilder;
      TrajectoryStateTransform tsTransform;
      FreeTrajectoryState fts_scl = tsTransform.outerFreeState(t,theMagField.product());
      TrajectoryStateClosestToPoint tscp_scl = tscpBuilder(fts_scl, GlobalPoint(theClus.position().x(),theClus.position().y(),theClus.position().z()));
      FreeTrajectoryState fts_seed = tsTransform.outerFreeState(t,theMagField.product());
      TrajectoryStateClosestToPoint tscp_seed = tscpBuilder(fts_seed,GlobalPoint(theClus.seed()->position().x(),theClus.seed()->position().y(),theClus.seed()->position().z()));
      edm::Ref<TrackCollection> trackRef(tracksH,i);
      const GlobalPoint pscl=tscp_scl.position();
      const GlobalVector mscl=tscp_scl.momentum();
      const GlobalPoint pseed=tscp_seed.position();
      const GlobalVector mseed=tscp_seed.momentum();
      PixelMatchGsfElectron ele((*sclAss)[seed],trackRef,pscl,mscl,pseed,mseed);
      //      PixelMatchGsfElectron ele((*sclAss)[seed],trackRef,tscp_scl.position(),tscp_scl.momentum(),tscp_seed.position(),tscp_seed.momentum());
      outEle.push_back(ele);
    }
  }  // loop over tracks
}

bool PixelMatchElectronAlgo::preSelection(const SuperCluster& clus, const Track& track, HBHERecHitMetaCollection mhbhe) 
{
  LogInfo("")<< "========== preSelection ==========";
 
  // extrapolate track inner momentum to nominal vertex
  TSCPBuilderNoMaterial tscpBuilder;
  TrajectoryStateTransform tsTransform;
  FreeTrajectoryState ftsin = tsTransform.innerFreeState(track,theMagField.product());
  TrajectoryStateClosestToPoint tscpin = tscpBuilder(ftsin, Global3DPoint(0,0,0) );
  // extrapolate track inner momentum to supercluster position
  FreeTrajectoryState ftscalo = tsTransform.innerFreeState(track,theMagField.product());
  TrajectoryStateClosestToPoint tscpcalo = tscpBuilder(ftscalo, Global3DPoint(clus.x(),clus.y(),clus.z()) );  
  // E/p criteria
  LogInfo("") << "E/p : " << clus.energy()/tscpin.momentum().mag();
  // temporary, exact identification of barrel and endcap case would be better
  if ((fabs(clus.eta()) < 1.479) && (clus.energy()/tscpin.momentum().mag() > maxEOverPBarrel_)) return false;
  if ((fabs(clus.eta()) >= 1.479) && (clus.energy()/tscpin.momentum().mag() > maxEOverPEndcaps_)) return false;
  LogInfo("") << "E/p criteria is satisfied ";
  // delta eta criteria
  double etaclu = clus.eta();
  double etatrk = tscpcalo.position().eta();
  double deta = etaclu-etatrk;
  LogInfo("") << "delta eta : " << deta;
  if (fabs(deta) > maxDeltaEta_) return false;
  LogInfo("") << "Delta eta criteria is satisfied ";
  // delta phi criteria
  double phiclu = clus.phi();
  double phitrk = tscpcalo.position().phi();
  double dphi = phiclu-phitrk;
  LogInfo("") << "delta phi : " << dphi;
  if (fabs(dphi) > maxDeltaPhi_) return false;
  LogInfo("") << "Delta phi criteria is satisfied ";
  // had/em criteria if non zero cone size given
  if (hOverEConeSize_ > 0.) {
    //cout << "calo position is eta-phi = " << clus.eta() << " " << clus.phi() << endl;
    CaloConeSelector sel(hOverEConeSize_, theCaloGeom.product(), DetId::Hcal);
    GlobalPoint pclu(clus.x(),clus.y(),clus.z());
    double hcalEnergy = 0.;
    std::auto_ptr<CaloRecHitMetaCollectionV> chosen=sel.select(pclu,mhbhe);
    for (CaloRecHitMetaCollectionV::const_iterator i=chosen->begin(); i!=chosen->end(); i++) {
      //std::cout << HcalDetId(i->detid()) << " : " << (*i) << std::endl;
      hcalEnergy += i->energy();
    }
    LogInfo("") << "H/E : " << hcalEnergy/clus.energy();
    if (hcalEnergy/clus.energy() > maxHOverE_) return false;
    LogInfo("") << "H/E criteria is satisfied ";
  }
  LogInfo("") << "electron has passed preselection criteria ";
  LogInfo("") << "=================================================";
  return true;  
}  
//**************************************************************************
// all the following  is temporary, to be replaced by a method Track::seed()
//**************************************************************************
bool PixelMatchElectronAlgo::equal(edm::Ref<TrajectorySeedCollection> ts, const Track& t) {
  // we have 2 valid rechits from the seed
  // which we have to find in the track
  // curiously, they are not the first ones...
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef recHitContainer::const_iterator const_iterator;
  typedef std::pair<const_iterator,const_iterator> range;
  range r=ts->recHits();
  int foundHits=0;
  for (TrackingRecHitCollection::const_iterator rhits=r.first; rhits!=r.second; rhits++) {
    if ((*rhits).isValid()) {
      for (unsigned int j=0;j<t.recHitsSize();j++) {
	TrackingRecHitRef rh =t.recHit(j);
	if (rh->isValid()) {
	  if (compareHits((*rhits),(*rh))) {
	    foundHits++;
	    break;
	  }
	}
      }
    }
  }
  if (foundHits==2) return true;

  return false;
}

bool PixelMatchElectronAlgo::compareHits(const TrackingRecHit& rh1, const TrackingRecHit & rh2) const {
  //FIXME: Teddy's class for comparison??
       const float eps=.002;
       return ((TMath::Abs(rh1.localPosition().x()-rh2.localPosition().x())<eps)
		&& (TMath::Abs(rh1.localPosition().y()-rh2.localPosition().y())<eps)
	       &&(TMath::Abs(rh1.localPosition().z()-rh2.localPosition().z())<eps));
     }
  
