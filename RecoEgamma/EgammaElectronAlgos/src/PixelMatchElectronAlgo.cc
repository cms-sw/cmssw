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
// $Id: PixelMatchElectronAlgo.cc,v 1.22 2006/12/06 16:18:34 uberthon Exp $
//
//
#include "RecoEgamma/EgammaElectronAlgos/interface/PixelMatchElectronAlgo.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


#include "TrackingTools/PatternTools/interface/Trajectory.h"
//#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleanerBySharedHits.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TransverseImpactPointExtrapolator.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
//#include "TrackingTools/PatternTools/interface/TrajectoryFitter.h"

//#include "RecoTracker/CkfPattern/interface/TransientInitialStateEstimator.h"
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
  maxDeltaEta_(maxDeltaEta), maxDeltaPhi_(maxDeltaPhi)
{}

PixelMatchElectronAlgo::~PixelMatchElectronAlgo() {}

void PixelMatchElectronAlgo::setupES(const edm::EventSetup& es, const edm::ParameterSet &conf) {

  //services
  es.get<TrackerRecoGeometryRecord>().get( theGeomSearchTracker );
  es.get<IdealMagneticFieldRecord>().get(theMagField);

  // get calo geometry
   es.get<IdealGeometryRecord>().get(theCaloGeom);
  
  // get nested parameter set for the TransientInitialStateEstimator
  ParameterSet tise_params = conf.getParameter<ParameterSet>("TransientInitialStateEstimatorParameters") ;
  trackBarrelLabel_ = conf.getParameter<string>("TrackBarrelLabel");
  trackBarrelInstanceName_ = conf.getParameter<string>("TrackBarrelProducer");
  trackEndcapLabel_ = conf.getParameter<string>("TrackEndcapLabel");
  trackEndcapInstanceName_ = conf.getParameter<string>("TrackEndcapProducer");
  assBarrelLabel_ = conf.getParameter<string>("SCLBarrelLabel");
  assBarrelInstanceName_ = conf.getParameter<string>("SCLBarrelProducer");
  assEndcapLabel_ = conf.getParameter<string>("SCLEndcapLabel");
  assEndcapInstanceName_ = conf.getParameter<string>("SCLEndcapProducer");
  assBarrelTrTSLabel_ = conf.getParameter<string>("AssocTrTSBarrelLabel");
  assBarrelTrTSInstanceName_ = conf.getParameter<string>("AssocTrTBarrelProducer");
  assEndcapTrTSLabel_ = conf.getParameter<string>("AssocTrTEndcapLabel");
  assEndcapTrTSInstanceName_ = conf.getParameter<string>("AssocTrTEndcapProducer");
}

void  PixelMatchElectronAlgo::run(Event& e, PixelMatchGsfElectronCollection & outEle) {

  // get the input 
  edm::Handle<GsfTrackCollection> tracksBarrelH;
  edm::Handle<GsfTrackCollection> tracksEndcapH;
  // to check existance
  edm::Handle<HBHERecHitCollection> hbhe;
  HBHERecHitMetaCollection *mhbhe=0;
  if (hOverEConeSize_ > 0.) {
    e.getByType(hbhe);  
    mhbhe=  &HBHERecHitMetaCollection(*hbhe);  //FIXME, generates warning
  }
  e.getByLabel(trackBarrelLabel_,trackBarrelInstanceName_,tracksBarrelH);
  e.getByLabel(trackEndcapLabel_,trackEndcapInstanceName_,tracksEndcapH);

  edm::Handle<SeedSuperClusterAssociationCollection> barrelH;
  edm::Handle<SeedSuperClusterAssociationCollection> endcapH;
  e.getByLabel(assBarrelLabel_,assBarrelInstanceName_,barrelH);
  e.getByLabel(assEndcapLabel_,assEndcapInstanceName_,endcapH);

  edm::Handle<GsfTrackSeedAssociationCollection> barrelTSAssocH;
  edm::Handle<GsfTrackSeedAssociationCollection> endcapTSAssocH;
  e.getByLabel(assBarrelTrTSLabel_,assBarrelTrTSInstanceName_,barrelTSAssocH);
  e.getByLabel(assEndcapTrTSLabel_,assEndcapTrTSInstanceName_,endcapTSAssocH);
  edm::LogInfo("") 
    <<"\n\n Treating "<<e.id()<<", Number of seeds Barrel:"
    <<barrelH.product()->size()<<" Number of seeds Endcap:"<<endcapH.product()->size();
  
  // create electrons from tracks in 2 steps: barrel + endcap
  const SeedSuperClusterAssociationCollection  *sclAss=&(*barrelH);
  process(tracksBarrelH,sclAss,barrelTSAssocH.product(),mhbhe,outEle);
  sclAss=&(*endcapH);
  process(tracksEndcapH,sclAss,endcapTSAssocH.product(),mhbhe,outEle);

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

void PixelMatchElectronAlgo::process(edm::Handle<GsfTrackCollection> tracksH, const SeedSuperClusterAssociationCollection *sclAss, const GsfTrackSeedAssociationCollection *tsAss,
                                     HBHERecHitMetaCollection *mhbhe, PixelMatchGsfElectronCollection & outEle) {
  const GsfTrackCollection *tracks=tracksH.product();
  for (unsigned int i=0;i<tracks->size();++i) {
    const GsfTrack & t=(*tracks)[i];
    const GsfTrackRef trackRef = edm::Ref<GsfTrackCollection>(tracksH,i);
    edm::Ref<TrajectorySeedCollection> seed = (*tsAss)[trackRef];
    const SuperCluster theClus=*((*sclAss)[seed]);

    // calculate HoE
    double HoE;
    if (mhbhe) {
      CaloConeSelector sel(hOverEConeSize_, theCaloGeom.product(), DetId::Hcal);
      GlobalPoint pclu(theClus.x(),theClus.y(),theClus.z());
      double hcalEnergy = 0.;
      std::auto_ptr<CaloRecHitMetaCollectionV> chosen=sel.select(pclu,*mhbhe);
      for (CaloRecHitMetaCollectionV::const_iterator i=chosen->begin(); i!=chosen->end(); i++) {
	//std::cout << HcalDetId(i->detid()) << " : " << (*i) << std::endl;
	hcalEnergy += i->energy();
      }
      HoE = hcalEnergy/theClus.energy();
      LogDebug("") << "H/E : " << HoE;
    } else HoE=0;

    if (preSelection(theClus,t,HoE)) {
      LogInfo("")<<"Constructed new electron with energy  "<< (*sclAss)[seed]->energy();
      TSCPBuilderNoMaterial tscpBuilder;
      TrajectoryStateTransform tsTransform;
      FreeTrajectoryState fts_scl = tsTransform.outerFreeState(t,theMagField.product());
      TrajectoryStateClosestToPoint tscp_scl = tscpBuilder(fts_scl, GlobalPoint(theClus.position().x(),theClus.position().y(),theClus.position().z()));
      FreeTrajectoryState fts_seed = tsTransform.outerFreeState(t,theMagField.product());
      TrajectoryStateClosestToPoint tscp_seed = tscpBuilder(fts_seed,GlobalPoint(theClus.seed()->position().x(),theClus.seed()->position().y(),theClus.seed()->position().z()));
      edm::Ref<GsfTrackCollection> trackRef(tracksH,i);
      const GlobalPoint pscl=tscp_scl.position();
      const GlobalVector mscl=tscp_scl.momentum();
      const GlobalPoint pseed=tscp_seed.position();
      const GlobalVector mseed=tscp_seed.momentum();
      PixelMatchGsfElectron ele((*sclAss)[seed],trackRef,pscl,mscl,pseed,mseed,HoE);
      outEle.push_back(ele);
    }
  }  // loop over tracks
}

bool PixelMatchElectronAlgo::preSelection(const SuperCluster& clus, const GsfTrack& track, double HoE) 
{
  LogDebug("")<< "========== preSelection ==========";
 
  // extrapolate track inner momentum to nominal vertex
  TSCPBuilderNoMaterial tscpBuilder;
  TrajectoryStateTransform tsTransform;
  FreeTrajectoryState ftsin = tsTransform.innerFreeState(track,theMagField.product());
  TrajectoryStateClosestToPoint tscpin = tscpBuilder(ftsin, Global3DPoint(0,0,0) );
  // extrapolate track inner momentum to supercluster position
  FreeTrajectoryState ftscalo = tsTransform.innerFreeState(track,theMagField.product());
  TrajectoryStateClosestToPoint tscpcalo = tscpBuilder(ftscalo, Global3DPoint(clus.x(),clus.y(),clus.z()) );  
  // E/p criteria
  LogDebug("") << "E/p : " << clus.energy()/tscpin.momentum().mag();
  // temporary, exact identification of barrel and endcap case would be better
  if ((fabs(clus.eta()) < 1.479) && (clus.energy()/tscpin.momentum().mag() > maxEOverPBarrel_)) return false;
  if ((fabs(clus.eta()) >= 1.479) && (clus.energy()/tscpin.momentum().mag() > maxEOverPEndcaps_)) return false;
  LogDebug("") << "E/p criteria is satisfied ";
  // delta eta criteria
  double etaclu = clus.eta();
  double etatrk = tscpcalo.position().eta();
  double deta = etaclu-etatrk;
  LogDebug("") << "delta eta : " << deta;
  if (fabs(deta) > maxDeltaEta_) return false;
  LogDebug("") << "Delta eta criteria is satisfied ";
  // delta phi criteria
  double phiclu = clus.phi();
  double phitrk = tscpcalo.position().phi();
  double dphi = phiclu-phitrk;
  LogDebug("") << "delta phi : " << dphi;
  if (fabs(dphi) > maxDeltaPhi_) return false;
  LogDebug("") << "Delta phi criteria is satisfied ";
  // had/em criteria if hcal rechits available
//   if (mhbhe) {
//     //cout << "calo position is eta-phi = " << clus.eta() << " " << clus.phi() << endl;
//     CaloConeSelector sel(hOverEConeSize_, theCaloGeom.product(), DetId::Hcal);
//     GlobalPoint pclu(clus.x(),clus.y(),clus.z());
//     double hcalEnergy = 0.;
//     std::auto_ptr<CaloRecHitMetaCollectionV> chosen=sel.select(pclu,*mhbhe);
//     for (CaloRecHitMetaCollectionV::const_iterator i=chosen->begin(); i!=chosen->end(); i++) {
//       //std::cout << HcalDetId(i->detid()) << " : " << (*i) << std::endl;
//       hcalEnergy += i->energy();
//     }
  if (HoE > maxHOverE_) return false; //FIXME: passe dans tous les cas?
  LogDebug("") << "H/E criteria is satisfied ";

  LogDebug("") << "electron has passed preselection criteria ";
  LogDebug("") << "=================================================";
  return true;  
}  
