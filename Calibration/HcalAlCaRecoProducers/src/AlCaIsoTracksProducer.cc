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

#include "Calibration/HcalAlCaRecoProducers/plugins/ConeDefinition.h"

#include <boost/regex.hpp> 

double getDistInCM(double eta1, double phi1, double eta2, double phi2)
{
  double dR, Rec;
  double theta1=2*atan(exp(-eta1));
  double theta2=2*atan(exp(-eta2));
  if (fabs(eta1)<1.479) Rec=129; //radius of ECAL barrel
  else Rec=tan(theta1)*317; //distance from IP to ECAL endcap

  //|vect| times tg of acos(scalar product)
  double angle=acos((sin(theta1)*sin(theta2)*(sin(phi1)*sin(phi2)+cos(phi1)*cos(phi2))+cos(theta1)*cos(theta2)));
  if (angle<acos(-1)/2)
    {
      dR=fabs((Rec/sin(theta1))*tan(angle));
      return dR;
    }
  else return 1000;
}

double getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + std::pow(eta1-eta2,2));
  return dr;
}

bool checkHLTMatch(edm::Event& iEvent, edm::InputTag hltEventTag_, std::vector<std::string> hltFilterTag_, double eta, double phi, double hltMatchingCone_)
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
      for (unsigned l=0; l<hltFilterTag_.size(); l++)
	{
	  if ((trEv->filterTag(iFilt).label()).substr(0,27)==hltFilterTag_[l]) 
	    {
	      KEYS=trEv->filterKeys(iFilt);
	    }
	}
    }
  trigger::size_type nReg=KEYS.size();
  for (trigger::size_type iReg=0; iReg<nReg; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
      double dHit=getDist(TObj.eta(),TObj.phi(),eta,phi); 
      if (dHit<minDDD) minDDD=dHit;
    }
  if (minDDD>hltMatchingCone_) match=false;
  else match=true;
     
  return match;
}

std::pair<double,double> getL1triggerDirection(edm::Event& iEvent, edm::InputTag hltEventTag_, std::string l1FilterTag_)
{
  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByLabel(hltEventTag_,trEv);
  const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());
  
  trigger::Keys KEYS;
  const trigger::size_type nFilt(trEv->sizeFilters());
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++)
    {
      if ((trEv->filterTag(iFilt).label()).substr(0,14)==l1FilterTag_) KEYS=trEv->filterKeys(iFilt); 
    }
  trigger::size_type nReg=KEYS.size();
  double etaTrig=-10000;
  double phiTrig=-10000;
  double ptMax=0;
  for (trigger::size_type iReg=0; iReg<nReg; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
      if (TObj.pt()>ptMax)
	{
	  etaTrig=TObj.eta();
	  phiTrig=TObj.phi();
	  ptMax=TObj.pt();
	}
    }
  return std::pair<double,double>(etaTrig,phiTrig);
}


AlCaIsoTracksProducer::AlCaIsoTracksProducer(const edm::ParameterSet& iConfig)
{ 
  
  m_inputTrackLabel_ = iConfig.getParameter<edm::InputTag>("InputTracksLabel");

  hoLabel_ = iConfig.getParameter<edm::InputTag>("HOInput");

  ecalLabels_=iConfig.getParameter<std::vector<edm::InputTag> >("ECALInputs");

  hbheLabel_= iConfig.getParameter<edm::InputTag>("HBHEInput");
  
  m_dvCut = iConfig.getParameter<double>("vtxCut");
  m_ddirCut = iConfig.getParameter<double>("RIsolAtHCALSurface");
  useConeCorr_=iConfig.getParameter<bool>("UseLowPtConeCorrection");
  m_pCut = iConfig.getParameter<double>("MinTrackP");
  m_ptCut = iConfig.getParameter<double>("MinTrackPt");
  m_ecalCut = iConfig.getUntrackedParameter<double>("NeutralIsolCut",8.);

  taECALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorECALCone",0.5);
  taHCALCone_=iConfig.getUntrackedParameter<double>("TrackAssociatorHCALCone",1.0);

  skipNeutrals_=iConfig.getUntrackedParameter<bool>("SkipNeutralIsoCheck",false);

  nHitsMinCore_=iConfig.getParameter<int>("MinNumberOfHitsCoreTrack");
  nHitsMinIso_=iConfig.getParameter<int>("MinNumberOfHitsInConeTracks");

  isolE_ = iConfig.getParameter<double>("MaxNearbyTrackEnergy");
  etaMax_= iConfig.getParameter<double>("MaxTrackEta");
  cluRad_ = iConfig.getParameter<double>("ECALClusterRadius");
  ringOutRad_ = iConfig.getParameter<double>("ECALRingOuterRadius");
  ringInnRad_=iConfig.getParameter<double>("ECALRingInnerRadius");  

  useECALCluMatrix_=iConfig.getParameter<bool>("ClusterECALasMatrix");
  matrixSize_=iConfig.getParameter<int>("ECALMatrixFullSize");

  checkHLTMatch_=iConfig.getParameter<bool>("CheckHLTMatch");
  hltEventTag_=iConfig.getParameter<edm::InputTag>("hltTriggerEventLabel");  
  hltFiltTag_=iConfig.getParameter<std::vector<std::string> >("hltL3FilterLabels");
  hltMatchingCone_=iConfig.getParameter<double>("hltMatchingCone");
  l1FilterTag_=iConfig.getParameter<std::string>("l1FilterLabel");
  l1jetVetoCone_=iConfig.getParameter<double>("l1JetVetoCone");
  
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
  std::vector<HcalDetId> usedHitsHC;
  std::vector<int> usedHitsEC;
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
 
    if (track->hitPattern().numberOfValidHits() < nHitsMinCore_) continue;

    // check that track is not in the region of L1 jet
    double l1jDR=deltaR(track->eta(), track->phi(), getL1triggerDirection(iEvent,hltEventTag_,l1FilterTag_).first,getL1triggerDirection(iEvent,hltEventTag_,l1FilterTag_).second);
    if (l1jDR<l1jetVetoCone_) continue;
    ///
	    
    TrackDetMatchInfo info = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track), parameters_);
    
    GlobalPoint gPointHcal(info.trkGlobPosAtHcal.x(),info.trkGlobPosAtHcal.y(),info.trkGlobPosAtHcal.z());

    GlobalVector trackMomAtHcal = info.trkMomAtHcal;
    
    double etaecal=info.trkGlobPosAtEcal.eta();
    double phiecal=info.trkGlobPosAtEcal.phi();

    if (etaecal==0&&phiecal==0) continue;    

    //check matching to HLT object (optional)

    if (checkHLTMatch_&&!checkHLTMatch(iEvent, hltEventTag_, hltFiltTag_, etaecal, phiecal,hltMatchingCone_)) continue;
    
    if (fabs(track->eta())>etaMax_) continue;
    
    // check charged isolation from all other tracks
    double maxPNearby=-10;
    double sumPNearby=0;
    for (reco::TrackCollection::const_iterator track1=tC.begin(); track1!=tC.end(); track1++)
      {
	itrk1++;
	if (track == track1) continue;
        if (track->hitPattern().numberOfValidHits() < nHitsMinIso_) continue;
	double ptrack1 = sqrt(track1->px()*track1->px()+track1->py()*track1->py()+track1->pz()*track1->pz());
	
	TrackDetMatchInfo info1 = trackAssociator_.associate(iEvent, iSetup, trackAssociator_.getFreeTrajectoryState(iSetup, *track1), parameters_);
	
	GlobalPoint gPointHcal1(info1.trkGlobPosAtHcal.x(),info1.trkGlobPosAtHcal.y(),info1.trkGlobPosAtHcal.z());

	double etaecal1=info1.trkGlobPosAtEcal.eta();
	double phiecal1=info1.trkGlobPosAtEcal.phi();
   
        if (etaecal1==0&&phiecal1==0) continue;	

	double hcDist=getDistInPlaneTrackDir(gPointHcal, trackMomAtHcal, gPointHcal1);

//        double hcDist=getDistInCM(etaecal,phiecal,etaecal1,phiecal1);

	// increase required separation for low momentum tracks
	double factor=1.;
	double factor1=1.;

	if (useConeCorr_)
	   {
	     if(ptrack<10.)factor+=(10.-ptrack)/20.;
	     if(ptrack1<10.)factor1+=(10.-ptrack1)/20.;
	   }
	
	if( hcDist <  m_ddirCut*factor*factor1 ) 
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
	double ecClustN=0;
	double ecOutRingR=0;

	//get index of ECAL crystal hit by track
	std::vector<const EcalRecHit*> crossedECids=info.crossedEcalRecHits;
	int etaIDcenter=-10000;
	int phiIDcenter=-10000;
	double enMax=0;
	for (unsigned int i=0; i<crossedECids.size(); i++)
	  {
	    if ((*crossedECids[i]).id().subdetId()==EcalEndcap)
	      {
		EEDetId did(crossedECids[i]->id());
		if (crossedECids[i]->energy()>enMax)
		  {
		    enMax=crossedECids[i]->energy();
		    etaIDcenter=did.iy();
		    phiIDcenter=did.ix();
		  }
	      }
	    if ((*crossedECids[i]).id().subdetId()==EcalBarrel)
	      {
		EBDetId did(crossedECids[i]->id());
		if (crossedECids[i]->energy()>enMax)
		  {
		    enMax=crossedECids[i]->energy();
		    etaIDcenter=did.ieta();
		    phiIDcenter=did.iphi();
		  }
	      }
	  }
	for (std::vector<EcalRecHit>::const_iterator ehit=tmpEcalRecHitCollection->begin(); ehit!=tmpEcalRecHitCollection->end(); ehit++) 
	  {
	    ////////////////////// FIND ECAL CLUSTER ENERGY
	    // R scheme of ECAL CLUSTERIZATION
	    GlobalPoint posH = geo->getPosition((*ehit).detid());
	    double phihit = posH.phi();
	    double etahit = posH.eta();
	    
	    double dHit=deltaR(etaecal,phiecal,etahit,phihit);
	    
	    double dHitCM=getDistInPlaneTrackDir(gPointHcal, trackMomAtHcal, posH);
	    
	    if (dHitCM<cluRad_)
	      {
		ecClustR+=ehit->energy();
	      }
	    
	    if (dHitCM>ringInnRad_&&dHitCM<ringOutRad_)
	      {
		ecOutRingR+=ehit->energy();
	      }

	    //////////////////////////////////
	    //NxN scheme & check whether hit was used or not, if not used push into usedHits
	    bool hitIsUsed=false;
	    int hitHashedIndex=-10000;
	    if (ehit->id().subdetId()==EcalBarrel)
	      {
		EBDetId did(ehit->id());
		hitHashedIndex=did.hashedIndex();
		if (fabs(did.ieta()-etaIDcenter)<=matrixSize_/2&&fabs(did.iphi()-phiIDcenter)<=matrixSize_/2) ecClustN+=ehit->energy();
	      }
	    
	    if (ehit->id().subdetId()==EcalEndcap)
	      {
		EEDetId did(ehit->id());
		hitHashedIndex=did.hashedIndex();
		if (fabs(did.iy()-etaIDcenter)<=matrixSize_/2&&fabs(did.ix()-phiIDcenter)<=matrixSize_/2) ecClustN+=ehit->energy();
	      }
	    for (uint32_t i=0; i<usedHitsEC.size(); i++)
	      {
		if (usedHitsEC[i]==hitHashedIndex) hitIsUsed=true;
	      }

	    if (hitIsUsed) continue; //skip if used
	    usedHitsEC.push_back(hitHashedIndex);
	    /////////////////////////////////
	    
	    if(dHit<1.)  
	      {
		outputEColl->push_back(*ehit);
	      }   
	  }

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
	if (!useECALCluMatrix_) newHITCandidate.SetEnergyIn(ecClustR);
	else newHITCandidate.SetEnergyIn(ecClustN);
	newHITCandidate.SetEnergyOut(ecOutRingR);
	outputHcalIsoTrackColl->push_back(newHITCandidate);

	//save hcal recHits
	for (std::vector<HBHERecHit>::const_iterator hhit=hbheRHcol->begin(); hhit!=hbheRHcol->end(); hhit++) 
	  {
	    //check that this hit was not considered before and push it into usedHits
	    bool hitIsUsed=false;
	    for (uint32_t i=0; i<usedHitsHC.size(); i++)
	      {
		if (usedHitsHC[i]==hhit->id()) hitIsUsed=true;
	      }
	    if (hitIsUsed) continue;
	    usedHitsHC.push_back(hhit->id());
	    ////////////
	    
	    GlobalPoint posH = geo->getPosition((*hhit).detid());
	    double phihit = posH.phi();
	    double etahit = posH.eta();
	    
	    double dHit=deltaR(etaecal,phiecal,etahit,phihit);
	    
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

