#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Calibration/HcalAlCaRecoProducers/interface/AlCaIsoTracksProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h" 
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h" 
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h" 
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"

#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "RecoTracker/TrackProducer/interface/TrackProducerBase.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"

#include "DataFormats/Provenance/interface/ProductID.h"

#include <boost/regex.hpp> 

double getDistInCM(double eta1, double phi1, double eta2, double phi2)
{
  double dR, Rec;
  double theta1=2*atan(exp(-eta1));
  double theta2=2*atan(exp(-eta2));
  if (fabs(eta1)<1.479) Rec=129;
  else Rec=275;
  //|vect| times tg of acos(scalar product)
  dR=fabs((Rec/sin(theta1))*tan(acos(sin(theta1)*sin(theta2)*(sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2))+cos(theta1)*cos(theta2))));
  return dR;
}

double getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + pow(eta1-eta2,2));
  return dr;
}

bool checkHLTMatch(edm::Event& iEvent, edm::InputTag hltEventTag_, edm::InputTag hltFilterTag_, double eta, double phi)
{
  bool match =false;
  double minDDD=1000;
  
  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByLabel(hltEventTag_,trEv);
  const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());
  
  trigger::Keys KEYS;
  const trigger::size_type nFilt(trEv->sizeFilters());
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
    {
      if (trEv->filterTag(iFilt)==hltFilterTag_) KEYS=trEv->filterKeys(iFilt);
    }
  trigger::size_type nReg=KEYS.size();
  for (trigger::size_type iReg=0; iReg<nReg; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
      double dHit=getDist(TObj.eta(),TObj.phi(),eta,phi); 
      if (dHit<minDDD) minDDD=dHit;
    }
  if (minDDD>0.4) match=false;
  else match=true;
     
  return match;


}



AlCaIsoTracksProducer::AlCaIsoTracksProducer(const edm::ParameterSet& iConfig)
{ 
  
  m_inputTrackLabel_ = iConfig.getParameter<edm::InputTag>("InputTracksLabel");

  hoLabel_ = iConfig.getParameter<edm::InputTag>("HOInput");

  ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ECALInputs");

  hbheLabel_= iConfig.getParameter<edm::InputTag>("HBHEInput");
  
  m_dvCut = iConfig.getParameter<double>("vtxCut");
  m_ddirCut = iConfig.getParameter<double>("RIsolAtECALSurface");
  useConeCorr_=iConfig.getParameter<bool>("UseLowPtConeCorrection");
  m_pCut = iConfig.getParameter<double>("MinTrackP");
  m_ptCut = iConfig.getParameter<double>("MinTrackPt");
  m_ecalCut = iConfig.getUntrackedParameter<double>("NeutralIsolCut",8.);

  taECALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorECALCone",0.5);
  taHCALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorHCALCone",1.0);

  skipNeutrals_=iConfig.getUntrackedParameter<bool>("SkipNeutralIsoCheck",false);

  isolE_ = iConfig.getParameter<double>("MaxNearbyTrackEnergy");
  etaMax_= iConfig.getParameter<double>("MaxTrackEta");
  cluRad_ = iConfig.getParameter<double>("ECALClusterRadius");
  ringOutRad_ = iConfig.getParameter<double>("ECALRingOuterRadius");
  ringInnRad_=iConfig.getParameter<double>("ECALRingInnerRadius");  

  checkHLTMatch_=iConfig.getParameter<bool>("CheckHLTMatch");
  hltEventTag_=iConfig.getParameter<edm::InputTag>("hltTriggerEventLabel");  
  hltFiltTag_=iConfig.getParameter<edm::InputTag>("hltL3FilterLabel");
  
  //////////
//
// Parameters for track associator   ===========================
//  
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  parameters_.loadParameters( parameters );
  trackAssociator_.useDefaultPropagator();
// ===============================================================

  //create also IsolatedPixelTrackCandidateCollection which contains isolation info and reference to primary track
  produces<reco::IsolatedPixelTrackCandidateCollection>("HcalIsolatedTrackCollection");

  produces<reco::TrackCollection>("IsoTrackTracksCollection");
  produces<reco::TrackExtraCollection>("IsoTrackExtraTracksCollection");

  produces<EcalRecHitCollection>("IsoTrackEcalRecHitCollection");
  produces<EcalRecHitCollection>("IsoTrackPSEcalRecHitCollection");
  
  produces<HBHERecHitCollection>("IsoTrackHBHERecHitCollection");
  produces<HORecHitCollection>("IsoTrackHORecHitCollection");

}


AlCaIsoTracksProducer::~AlCaIsoTracksProducer() { }


void AlCaIsoTracksProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  std::auto_ptr<reco::IsolatedPixelTrackCandidateCollection> outputHcalIsoTrackColl(new reco::IsolatedPixelTrackCandidateCollection);
  std::auto_ptr<reco::TrackCollection> outputTColl(new reco::TrackCollection);
  std::auto_ptr<reco::TrackExtraCollection> outputExTColl(new reco::TrackExtraCollection);
  
  reco::TrackExtraRefProd rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>("IsoTrackExtraTracksCollection"); 
  std::auto_ptr<EcalRecHitCollection> outputEColl(new EcalRecHitCollection);
  std::auto_ptr<EcalRecHitCollection> outputESColl(new EcalRecHitCollection);
  std::auto_ptr<HBHERecHitCollection> outputHColl(new HBHERecHitCollection);
  std::auto_ptr<HORecHitCollection> outputHOColl(new HORecHitCollection);
  
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  geo = pG.product();
  
  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(m_inputTrackLabel_,trackCollection);
  
  
  // temporary collection of EB+EE recHits
  std::auto_ptr<EcalRecHitCollection> tmpEcalRecHitCollection(new EcalRecHitCollection);
  
  std::vector<edm::InputTag>::const_iterator i;
  for (i=ecalLabels_.begin(); i!=ecalLabels_.end(); i++) 
    {
      edm::Handle<EcalRecHitCollection> ec;
      iEvent.getByLabel(*i,ec);
      for(EcalRecHitCollection::const_iterator recHit = (*ec).begin(); recHit != (*ec).end(); ++recHit)
	{
	  tmpEcalRecHitCollection->push_back(*recHit);
	}
    }      
  
  edm::Handle<HBHERecHitCollection> hbheRHcol;
  iEvent.getByLabel(hbheLabel_, hbheRHcol);

  const reco::TrackCollection tC = *(trackCollection.product());
  
  int itrk=0;
  int nisotr=0;
  edm::Ref<reco::TrackExtraCollection>::key_type  idx = 0;
  
  //   Parameters for TrackDetAssociator ================================
  // For Low momentum tracks need to look for larger cone for search ====
  // ====================================================================
  
  parameters_.useEcal = true;
  parameters_.useHcal = true;
  parameters_.useCalo = false;
  parameters_.useMuon = false;
  parameters_.dREcal = taECALCone_;
  parameters_.dRHcal = taHCALCone_;
  
  ///////////////////////////////
  
  ///vector of used hits:
  std::vector<int> usedHits;
  ///

  // main loop over input tracks
  for (reco::TrackCollection::const_iterator track=tC.begin(); track!=tC.end(); track++) {
    bool noChargedTracks = true;
    int itrk1=0;
    itrk++;
    double px = track->px();
    double py = track->py();
    double pz = track->pz();
    double ptrack = sqrt(px*px+py*py+pz*pz);
    
    if (ptrack < m_pCut || track->pt() < m_ptCut ) continue; 
	    
    TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track), parameters_);
    
    double etaecal=info.trkGlobPosAtEcal.eta();
    double phiecal=info.trkGlobPosAtEcal.phi();
    
//    double thetaecal = 2*atan(exp(-etaecal));

    //check matching to HLT object to make sure that ecal FEDs are present (optional)

    if (checkHLTMatch_&&!checkHLTMatch(iEvent, hltEventTag_, hltFiltTag_, etaecal, phiecal)) continue;
    
    if (fabs(etaecal)>etaMax_) continue;
    
    // check charged isolation from all other tracks
    double maxPNearby=-10;
    double sumPNearby=0;
    for (reco::TrackCollection::const_iterator track1=tC.begin(); track1!=tC.end(); track1++)
      {
	itrk1++;
	if (track == track1) continue;
	double ptrack1 = sqrt(track1->px()*track1->px()+track1->py()*track1->py()+track1->pz()*track1->pz());
	
	TrackDetMatchInfo info1 = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track1), parameters_);
	
	double etaecal1=info1.trkGlobPosAtEcal.eta();
	double phiecal1=info1.trkGlobPosAtEcal.phi();
   
        if (etaecal1==0&&phiecal1==0) continue;	

	double ecDist=getDistInCM(etaecal,phiecal,etaecal1,phiecal1);

	// increase required separation for low momentum tracks
	double factor=1.;
	double factor1=1.;

	if (useConeCorr_)
	   {
	     if(ptrack<10.)factor+=(10.-ptrack)/20.;
	     if(ptrack1<10.)factor1+=(10.-ptrack1)/20.;
	   }

	if( ecDist <  m_ddirCut*factor*factor1 ) 
	  {
	    //calculate maximum P and sum P near seed track
	    if (track1->p()>maxPNearby)
	      {
		maxPNearby=track1->p();
	      }
	    sumPNearby+=track1->p();
	    
	    //apply loose isolation criteria
	    if (track1->p()>isolE_) 
	      {
		noChargedTracks = false;
		break;
	      }
	    //////////////
	  }
	
      } //second track loop

    bool noNeutrals = false;
    
    // we have a good charge-isolated track, so check neutral isolation and write it out
	
    if(noChargedTracks) 
      {
	//find ecal cluster energy and write ecal recHits
	double ecClustR=0;
	double ecOutRingR=0;
	usedHits.clear();
	for (std::vector<EcalRecHit>::const_iterator ehit=tmpEcalRecHitCollection->begin(); ehit!=tmpEcalRecHitCollection->end(); ehit++) 
	  {
	    ////////////////////// FIND ECAL CLUSTER ENERGY
	    // R-scheme of ECAL CLUSTERIZATION
	    GlobalPoint posH = geo->getPosition((*ehit).detid());
	    double phihit = posH.phi();
	    double etahit = posH.eta();
	    
	    double dHit=getDist(etaecal,phiecal,etahit,phihit);
	    
	    double dHitCM=getDistInCM(etaecal,phiecal,etahit,phihit);
	    
	    if (dHitCM<cluRad_)
	      {
		ecClustR+=ehit->energy();
	      }
	    if (dHitCM>ringInnRad_&&dHitCM<ringOutRad_)
	      {
		ecOutRingR+=ehit->energy();
	      }
	    //////////////////////////////////
			    
	    // check whether hit was used or not, if not used push into usedHits
	    bool hitIsUsed=false;
	    int hitHashedIndex=-10000;
	    if (ehit->id().subdetId()==EcalBarrel)
	      {
		EBDetId did(ehit->id());
		hitHashedIndex=did.hashedIndex();
	      }
	    
	    if (ehit->id().subdetId()==EcalEndcap)
	      {
		EEDetId did(ehit->id());
		hitHashedIndex=did.hashedIndex();
	      }
	    for (uint32_t i=0; i<usedHits.size(); i++)
	      {
		if (usedHits[i]==hitHashedIndex) hitIsUsed=true;
	      }
	    if (hitIsUsed) continue; //skip if used
	    usedHits.push_back(hitHashedIndex);
	    /////////////////////////////////
	    
	    if(dHit<1.)  
	      {
		outputEColl->push_back(*ehit);
	      }   
	  }

	usedHits.clear();
	
	//check neutrals 
        if (ecOutRingR<m_ecalCut) noNeutrals=true;
	else noNeutrals=false;

	if (noNeutrals||skipNeutrals_)
	{

        // Take info on the track extras and keep it in the outercollection

        reco::TrackExtraRef myextra = (*track).extra();
        reco::TrackExtraRef teref= reco::TrackExtraRef ( rTrackExtras, idx ++ );

        outputTColl->push_back(*track);
        reco::Track & mytrack = outputTColl->back();

        mytrack.setExtra( teref );
        outputExTColl->push_back(*myextra);
        //      reco::TrackExtra & tx = outputExTColl->back();

        //Create IsolatedPixelTrackCandidate (will change naming in future release)
        reco::IsolatedPixelTrackCandidate newHITCandidate(reco::Candidate::LorentzVector(track->px(),track->py(),track->pz(),track->p()));
	newHITCandidate.SetSumPtPxl(sumPNearby);
	newHITCandidate.SetMaxPtPxl(maxPNearby);

	//set cluster energy deposition and ring energy deposition and push_back
	newHITCandidate.SetEnergyIn(ecClustR);
	newHITCandidate.SetEnergyOut(ecOutRingR);
	outputHcalIsoTrackColl->push_back(newHITCandidate);

	//save hcal recHits
	for (std::vector<HBHERecHit>::const_iterator hhit=hbheRHcol->begin(); hhit!=hbheRHcol->end(); hhit++) 
	  {
	    //check that this hit was not considered before and push it into usedHits
	    bool hitIsUsed=false;
	    int hitHashedIndex=hhit->id().hashed_index();
	    for (uint32_t i=0; i<usedHits.size(); i++)
	      {
		if (usedHits[i]==hitHashedIndex) hitIsUsed=true;
	      }
	    if (hitIsUsed) continue;
	    usedHits.push_back(hitHashedIndex);
	    ////////////
	    
	    GlobalPoint posH = geo->getPosition((*hhit).detid());
	    double phihit = posH.phi();
	    double etahit = posH.eta();
	    
	    double dHit=getDist(etaecal,phiecal,etahit,phihit);
	    
	    if(dHit<1.)  
	      {
		outputHColl->push_back(*hhit);
	      }
	    }
	  }
	
	nisotr++;
	
      } //if (noNeutrals....
  
  } // end of main track loop
  
  if(outputTColl->size() > 0)
    {
      //   Take HO collection
      edm::Handle<HORecHitCollection> ho;
      iEvent.getByLabel(hoLabel_,ho);
      
      const HORecHitCollection Hitho = *(ho.product());
      for(HORecHitCollection::const_iterator hoItr=Hitho.begin(); hoItr!=Hitho.end(); hoItr++)
	{
	  outputHOColl->push_back(*hoItr);
	}
           
      // Take Preshower collection      
      
      // get the ps geometry
      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
      
      // get the ps topology
      EcalPreshowerTopology psTopology(geoHandle);
      
      // process rechits
      edm::Handle<EcalRecHitCollection> pRecHits;
      iEvent.getByLabel("ecalPreshowerRecHit","EcalRecHitsES",pRecHits);
      const EcalRecHitCollection& psrechits = *(pRecHits.product());

      typedef EcalRecHitCollection::const_iterator IT;
      
      for(IT i=psrechits.begin(); i!=psrechits.end(); i++) 
	{
	  outputESColl->push_back( *i );
	}
    }  

  iEvent.put( outputHcalIsoTrackColl, "HcalIsolatedTrackCollection");
  iEvent.put( outputTColl, "IsoTrackTracksCollection");
  iEvent.put( outputExTColl, "IsoTrackExtraTracksCollection");
  iEvent.put( outputEColl, "IsoTrackEcalRecHitCollection");
  iEvent.put( outputESColl, "IsoTrackPSEcalRecHitCollection");
  iEvent.put( outputHColl, "IsoTrackHBHERecHitCollection");
  iEvent.put( outputHOColl, "IsoTrackHORecHitCollection");
}

void AlCaIsoTracksProducer::endJob(void) {}

