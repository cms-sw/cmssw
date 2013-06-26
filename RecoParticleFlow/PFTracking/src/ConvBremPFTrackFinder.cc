#include "RecoParticleFlow/PFTracking/interface/ConvBremPFTrackFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"


#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoParticleFlow/PFProducer/interface/Utils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h" 
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h" 
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"
#include "TMath.h"
#include "RecoParticleFlow/PFClusterTools/interface/LinkByRecHit.h"

using namespace edm;
using namespace std;
using namespace reco;

ConvBremPFTrackFinder::ConvBremPFTrackFinder(const TransientTrackBuilder& builder,
					     double mvaBremConvCut,
					     string mvaWeightFileConvBrem):
  builder_(builder),
  mvaBremConvCut_(mvaBremConvCut),
  mvaWeightFileConvBrem_(mvaWeightFileConvBrem)
{
  tmvaReader_ = new TMVA::Reader("!Color:Silent");
  tmvaReader_->AddVariable("secR",&secR);
  tmvaReader_->AddVariable("sTIP",&sTIP);
  tmvaReader_->AddVariable("nHITS1",&nHITS1);
  tmvaReader_->AddVariable("secPin",&secPin);
  tmvaReader_->AddVariable("Epout",&Epout);
  tmvaReader_->AddVariable("detaBremKF",&detaBremKF);
  tmvaReader_->AddVariable("ptRatioGsfKF",&ptRatioGsfKF);
  tmvaReader_->BookMVA("BDT",mvaWeightFileConvBrem.c_str());

  pfcalib_ = new PFEnergyCalibration();

}
ConvBremPFTrackFinder::~ConvBremPFTrackFinder(){delete tmvaReader_; delete pfcalib_; }

void
ConvBremPFTrackFinder::runConvBremFinder(const Handle<PFRecTrackCollection>& thePfRecTrackCol,
					 const Handle<VertexCollection>& primaryVertex,
					 const edm::Handle<reco::PFDisplacedTrackerVertexCollection>& pfNuclears,
					 const edm::Handle<reco::PFConversionCollection >& pfConversions,
					 const edm::Handle<reco::PFV0Collection >& pfV0,
					 bool useNuclear,
					 bool useConversions,
					 bool useV0,
					 const reco::PFClusterCollection & theEClus,
					 const reco::GsfPFRecTrack& gsfpfrectk)
{
  

  found_ = false;
  bool debug = false;
  bool debugRef = false;
  
  if(debug)
    cout << "runConvBremFinder:: Entering " << endl;
  
  
  
  reco::GsfTrackRef refGsf =  gsfpfrectk.gsfTrackRef();
  reco::PFRecTrackRef pfTrackRef = gsfpfrectk.kfPFRecTrackRef();
  vector<PFBrem> primPFBrem = gsfpfrectk.PFRecBrem();
 
  
  const PFRecTrackCollection& PfRTkColl = *(thePfRecTrackCol.product());
  reco::PFRecTrackCollection::const_iterator pft=PfRTkColl.begin();
  reco::PFRecTrackCollection::const_iterator pftend=PfRTkColl.end();
  //PFEnergyCalibration pfcalib_;




  vector<PFRecTrackRef> AllPFRecTracks;
  AllPFRecTracks.clear();
  unsigned int ipft = 0;
  

  for(;pft!=pftend;++pft,ipft++){
    // do not consider the kf track already associated to the seed
    if(pfTrackRef.isNonnull())
      if(pfTrackRef->trackRef() == pft->trackRef()) continue;
    
    PFRecTrackRef pfRecTrRef(thePfRecTrackCol,ipft);  
    TrackRef trackRef = pfRecTrRef->trackRef();
    reco::TrackBaseRef selTrackBaseRef(trackRef);
    
    if(debug)
      cout << "runConvBremFinder:: pushing_back High Purity " << pft->trackRef()->pt()  
	   << " eta,phi " << pft->trackRef()->eta() << ", " <<   pft->trackRef()->phi() 
	   <<  " Memory Address Ref  " << &*trackRef << " Memory Address BaseRef  " << &*selTrackBaseRef << endl;  
    AllPFRecTracks.push_back(pfRecTrRef);
  }
  

  if(useConversions) {
    const PFConversionCollection& PfConvColl = *(pfConversions.product());
    for(unsigned i=0;i<PfConvColl.size(); i++) {
      reco::PFConversionRef convRef(pfConversions,i);
      
      unsigned int trackSize=(convRef->pfTracks()).size();
      if ( convRef->pfTracks().size() < 2) continue;
      for(unsigned iTk=0;iTk<trackSize; iTk++) {
	PFRecTrackRef compPFTkRef = convRef->pfTracks()[iTk];
	reco::TrackBaseRef newTrackBaseRef(compPFTkRef->trackRef());
	// do not consider the kf track already associated to the seed
	if(pfTrackRef.isNonnull()) {
	  reco::TrackBaseRef primaryTrackBaseRef(pfTrackRef->trackRef());
	  if(primaryTrackBaseRef == newTrackBaseRef) continue;
	}
	bool notFound = true;
 	for(unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
	  reco::TrackBaseRef selTrackBaseRef(AllPFRecTracks[iPF]->trackRef());

	  if(debugRef) 
	    cout << "## Track 1 HP pt " << AllPFRecTracks[iPF]->trackRef()->pt() << " eta, phi " << AllPFRecTracks[iPF]->trackRef()->eta() << ", " << AllPFRecTracks[iPF]->trackRef()->phi() 
		 << " Memory Address Ref  " << &(*AllPFRecTracks[iPF]->trackRef()) << " Memory Address BaseRef  " << &*selTrackBaseRef << endl;
	  if(debugRef) 
	    cout << "** Track 2 CONV pt " << compPFTkRef->trackRef()->pt() << " eta, phi " <<  compPFTkRef->trackRef()->eta() << ", " << compPFTkRef->trackRef()->phi() 
		 << " Memory Address Ref " << &*compPFTkRef->trackRef() <<  " Memory Address BaseRef " << &*newTrackBaseRef << endl;
	  //if(selTrackBaseRef == newTrackBaseRef ||  AllPFRecTracks[iPF]->trackRef()== compPFTkRef->trackRef()) {
	  if(AllPFRecTracks[iPF]->trackRef()== compPFTkRef->trackRef()) {
	    if(debugRef) 
	      cout << "  SAME BREM REF " << endl;
	    notFound = false;
	  }
	}
	if(notFound) {
	  if(debug)
	    cout << "runConvBremFinder:: pushing_back Conversions " << compPFTkRef->trackRef()->pt() 
		 << " eta,phi " << compPFTkRef->trackRef()->eta() << " phi " << compPFTkRef->trackRef()->phi() <<endl; 
	  AllPFRecTracks.push_back(compPFTkRef);
	}
      }
    }
  }

  if(useNuclear) {
    const PFDisplacedTrackerVertexCollection& PfNuclColl = *(pfNuclears.product());
    for(unsigned i=0;i<PfNuclColl.size(); i++) {
      const reco::PFDisplacedTrackerVertexRef dispacedVertexRef(pfNuclears, i );
      unsigned int trackSize= dispacedVertexRef->pfRecTracks().size();
      for(unsigned iTk=0;iTk < trackSize; iTk++) {
	reco::PFRecTrackRef newPFRecTrackRef = dispacedVertexRef->pfRecTracks()[iTk]; 
	reco::TrackBaseRef newTrackBaseRef(newPFRecTrackRef->trackRef());
	// do not consider the kf track already associated to the seed
	if(pfTrackRef.isNonnull()) {
	  reco::TrackBaseRef primaryTrackBaseRef(pfTrackRef->trackRef());
	  if(primaryTrackBaseRef == newTrackBaseRef) continue;
	}
	bool notFound = true;
	for(unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
	  reco::TrackBaseRef selTrackBaseRef(AllPFRecTracks[iPF]->trackRef());
	  if(selTrackBaseRef == newTrackBaseRef) notFound = false;
	}
	if(notFound) {
	  if(debug)
	    cout << "runConvBremFinder:: pushing_back displaced Vertex pt " << newPFRecTrackRef->trackRef()->pt()  
		 << " eta,phi " << newPFRecTrackRef->trackRef()->eta() << ", " <<   newPFRecTrackRef->trackRef()->phi() <<  endl; 
	  AllPFRecTracks.push_back(newPFRecTrackRef);
	}
      }
    }
  }

  if(useV0) {
    const PFV0Collection& PfV0Coll = *(pfV0.product());
    for(unsigned i=0;i<PfV0Coll.size(); i++) {
      reco::PFV0Ref v0Ref( pfV0, i );
      unsigned int trackSize=(v0Ref->pfTracks()).size();
      for(unsigned iTk=0;iTk<trackSize; iTk++) {
	reco::PFRecTrackRef newPFRecTrackRef = (v0Ref->pfTracks())[iTk]; 
	reco::TrackBaseRef newTrackBaseRef(newPFRecTrackRef->trackRef());
	// do not consider the kf track already associated to the seed
	if(pfTrackRef.isNonnull()) {
	  reco::TrackBaseRef primaryTrackBaseRef(pfTrackRef->trackRef());
	  if(primaryTrackBaseRef == newTrackBaseRef) continue;
	}
	bool notFound = true;
	for(unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
	  reco::TrackBaseRef selTrackBaseRef(AllPFRecTracks[iPF]->trackRef());
	  if(selTrackBaseRef == newTrackBaseRef) notFound = false;
	}
	if(notFound) {
	  if(debug)
	    cout << "runConvBremFinder:: pushing_back V0 " << newPFRecTrackRef->trackRef()->pt()  
		 << " eta,phi " << newPFRecTrackRef->trackRef()->eta() << ", " <<   newPFRecTrackRef->trackRef()->phi() << endl; 
	  AllPFRecTracks.push_back(newPFRecTrackRef);
	}
      }
    }
  }



  pfRecTrRef_vec_.clear();


  for(unsigned iPF = 0; iPF < AllPFRecTracks.size(); iPF++) {
  
 
    double dphi= fabs(AllPFRecTracks[iPF]->trackRef()->phi()-refGsf->phi()); 
    if (dphi>TMath::Pi()) dphi-= TMath::TwoPi();
    double deta=fabs(AllPFRecTracks[iPF]->trackRef()->eta()-refGsf->eta());
     
    // limiting the phase space (just for saving cpu-time)
    if( fabs(dphi)> 1.0  || fabs(deta) > 0.4) continue;
    
  
    double minDEtaBremKF = 1000.;
    double minDPhiBremKF = 1000.;
    double minDRBremKF = 1000.;
    double minDEtaBremKFPos = 1000.;
    double minDPhiBremKFPos = 1000.;
    double minDRBremKFPos = 1000.;
    reco:: TrackRef  trkRef = AllPFRecTracks[iPF]->trackRef();
 
    double secEta = trkRef->innerMomentum().eta();
    double secPhi = trkRef->innerMomentum().phi();
    
    for(unsigned ipbrem = 0; ipbrem < primPFBrem.size(); ipbrem++) {
      if(primPFBrem[ipbrem].indTrajPoint() == 99) continue;
      const reco::PFTrajectoryPoint& atPrimECAL 
	= primPFBrem[ipbrem].extrapolatedPoint( reco::PFTrajectoryPoint::ECALEntrance );
      if( ! atPrimECAL.isValid() ) continue;
      double bremEta = atPrimECAL.momentum().Eta();
      double bremPhi = atPrimECAL.momentum().Phi();

      
      double deta = fabs(bremEta - secEta);
      double dphi = fabs(bremPhi - secPhi);
      if (dphi>TMath::Pi()) dphi-= TMath::TwoPi();
      double DR = sqrt(deta*deta + dphi*dphi);
      
      
      double detaPos = fabs(bremEta - trkRef->innerPosition().eta());
      double dphiPos = fabs(bremPhi - trkRef->innerPosition().phi());
      if (dphiPos>TMath::Pi()) dphiPos-= TMath::TwoPi();
      double DRPos = sqrt(detaPos*detaPos + dphiPos*dphiPos);
      


      // find the closest track tangent
      if(DR < minDRBremKF) {
	
	minDRBremKF = DR;
	minDEtaBremKF = deta;
	minDPhiBremKF = fabs(dphi);
      }
      
      if(DRPos < minDRBremKFPos) {
	minDRBremKFPos = DR;
	minDEtaBremKFPos = detaPos;
	minDPhiBremKFPos = fabs(dphiPos);
      }
  
    }

    //gsfR
    float gsfR = sqrt(refGsf->innerPosition().x()*refGsf->innerPosition().x() + 
		      refGsf->innerPosition().y()*refGsf->innerPosition().y() );	
    
    
    // secR
    secR = sqrt(trkRef->innerPosition().x()*trkRef->innerPosition().x() + 
		trkRef->innerPosition().y()*trkRef->innerPosition().y() );   
    
  
    // apply loose selection (to be parallel) between the secondary track and brem-tangents.
    // Moreover if the secR is internal with respect to the GSF track by two pixel layers discard it.
    if( (minDPhiBremKF < 0.1 || minDPhiBremKFPos < 0.1) &&
	(minDEtaBremKF < 0.02 ||  minDEtaBremKFPos < 0.02)&&
	secR > (gsfR-8)) {
      

      if(debug)
	cout << "runConvBremFinder:: OK Find track and BREM close " 
	     << " MinDphi " << minDPhiBremKF << " MinDeta " << minDEtaBremKF  << endl;
      

      float MinDist = 100000.;
      float EE_calib = 0.; 
      PFRecTrack pfrectrack = *AllPFRecTracks[iPF];
      pfrectrack.calculatePositionREP();
      // Find and ECAL associated cluster
      for (PFClusterCollection::const_iterator clus = theEClus.begin();
	   clus != theEClus.end();
	   clus++ ) {
	// Removed unusd variable, left this in case it has side effects
	clus->position();
	double dist = -1.;
	PFCluster clust = *clus;
	clust.calculatePositionREP();
	dist =  pfrectrack.extrapolatedPoint( reco::PFTrajectoryPoint::ECALShowerMax ).isValid() ?
	  LinkByRecHit::testTrackAndClusterByRecHit(pfrectrack , clust ) : -1.;
	
	if(dist > 0.) {
	  bool applyCrackCorrections = false;
	  vector<double> ps1Ene(0);
	  vector<double> ps2Ene(0);
	  double ps1,ps2;
	  ps1=ps2=0.;
	  if(dist < MinDist) {
	    MinDist = dist;
	    EE_calib = pfcalib_->energyEm(*clus,ps1Ene,ps2Ene,ps1,ps2,applyCrackCorrections);
	  }
	}
      }
      if(MinDist > 0. && MinDist < 100000.) {

	// compute all the input variables for conv brem selection
	
	secPout = sqrt(trkRef->outerMomentum().x()*trkRef->outerMomentum().x() +
		       trkRef->outerMomentum().y()*trkRef->outerMomentum().y() +
		       trkRef->outerMomentum().z()*trkRef->outerMomentum().z());
	
	secPin = sqrt(trkRef->innerMomentum().x()*trkRef->innerMomentum().x() +
		      trkRef->innerMomentum().y()*trkRef->innerMomentum().y() +
		      trkRef->innerMomentum().z()*trkRef->innerMomentum().z());
	

	// maybe put innter momentum pt? 
	ptRatioGsfKF = trkRef->pt()/(refGsf->ptMode());
	
	Vertex dummy;
	const Vertex *pv = &dummy;
	edm::Ref<VertexCollection> pvRef;
	if (primaryVertex->size() != 0) {
	  pv = &*primaryVertex->begin();
	  // we always use the first vertex (at the moment)
	  pvRef = edm::Ref<VertexCollection>(primaryVertex, 0);
	} else { // create a dummy PV
	  Vertex::Error e;
	  e(0, 0) = 0.0015 * 0.0015;
	  e(1, 1) = 0.0015 * 0.0015;
	  e(2, 2) = 15. * 15.;
	  Vertex::Point p(0, 0, 0);
	  dummy = Vertex(p, e, 0, 0, 0);
	}
	
	
	// direction of the Gsf track
	GlobalVector direction(refGsf->innerMomentum().x(), 
			       refGsf->innerMomentum().y(), 
			       refGsf->innerMomentum().z());
	
	TransientTrack transientTrack = builder_.build(*trkRef);	 
	sTIP = IPTools::signedTransverseImpactParameter(transientTrack, direction, *pv).second.significance();
	

	Epout = EE_calib/secPout;
	
	// eta distance brem-secondary kf track
	detaBremKF = minDEtaBremKF;
	
	// Number of commont hits
	trackingRecHit_iterator  nhit=refGsf->recHitsBegin();
	trackingRecHit_iterator  nhit_end=refGsf->recHitsEnd();
	unsigned int tmp_sh = 0;
	//uint ish=0;
	
	for (;nhit!=nhit_end;++nhit){
	  if ((*nhit)->isValid()){
	    trackingRecHit_iterator  ihit=trkRef->recHitsBegin();
	    trackingRecHit_iterator  ihit_end=trkRef->recHitsEnd();
	    for (;ihit!=ihit_end;++ihit){
	      if ((*ihit)->isValid()) {
		// method 1
		if((*nhit)->sharesInput(&*(*ihit),TrackingRecHit::all))  tmp_sh++;
		
		// method 2 to switch in case of problem with rechit collections
		//  if(((*ihit)->geographicalId()==(*nhit)->geographicalId())&&
		//  (((*nhit)->localPosition()-(*ihit)->localPosition()).mag()<0.01)) ish++;
		
		
	      }
	    }
	  }
	}
	  
	nHITS1 = tmp_sh;
	
	double mvaValue = tmvaReader_->EvaluateMVA("BDT");
	
	if(debug) 
	  cout << " The imput variables for conv brem tracks identification " << endl
	       << " secR          " << secR << " gsfR " << gsfR  << endl
	       << " N shared hits " << nHITS1 << endl
	       << " sTIP          " << sTIP << endl
	       << " detaBremKF    " << detaBremKF << endl
	       << " E/pout        " << Epout << endl
	       << " pin           " << secPin << endl
	       << " ptRatioKFGsf  " << ptRatioGsfKF << endl
	       << " ***** MVA ***** " << mvaValue << endl;
	
	if(mvaValue > mvaBremConvCut_) {
	  found_ = true;
	  pfRecTrRef_vec_.push_back(AllPFRecTracks[iPF]);
	  
	}
      } // end MinDIST
    } // end selection kf - brem tangents
  } // loop on the kf tracks





    
}


