// -*- C++ -*-
//
// Package:    Calibration/HcalCalibAlgos/plugins
// Class:      ValidationHcalIsoTrackAlCaReco
// 
/**\class ValidationHcalIsoTrackAlCaReco ValidationHcalIsoTrackAlCaReco.cc Calibration/HcalCalibAlgos/plugins/ValidationHcalIsoTrackAlCaReco.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Grigory SAFRONOV, Sergey PETRUSHANKO
//         Created:  Tue Oct  14 16:10:31 CEST 2008
// $Id: ValidationHcalIsoTrackAlCaReco.cc,v 1.4 2012/09/28 16:27:32 wdd Exp $
//
//


// system include files
#include <memory>

// user include files

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsTechTrigRcd.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"


// Sergey +

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

// Sergey -

#include "Calibration/HcalCalibAlgos/plugins/ValidationHcalIsoTrackAlCaReco.h"
#include <fstream>

#include "TH1F.h"


double ValidationHcalIsoTrackAlCaReco::getDist(double eta1, double phi1, double eta2, double phi2)
{
  double dphi = fabs(phi1 - phi2); 
  if(dphi>acos(-1)) dphi = 2*acos(-1)-dphi;
  double dr = sqrt(dphi*dphi + pow(eta1-eta2,2));
  return dr;
}

// Sergey +

double ValidationHcalIsoTrackAlCaReco::getDistInCM(double eta1, double phi1, double eta2, double phi2)
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

// Sergey -

std::pair<int,int> ValidationHcalIsoTrackAlCaReco::towerIndex(double eta, double phi) 
{
  int ieta=0;
  int iphi=0;
  for (int i=1; i<21; i++)
    {
      if (fabs(eta)<(i*0.087)&&fabs(eta)>(i-1)*0.087) ieta=int(fabs(eta)/eta)*i;
    }
  if (fabs(eta)>1.740&&fabs(eta)<1.830) ieta=int(fabs(eta)/eta)*21;
  if (fabs(eta)>1.830&&fabs(eta)<1.930) ieta=int(fabs(eta)/eta)*22;
  if (fabs(eta)>1.930&&fabs(eta)<2.043) ieta=int(fabs(eta)/eta)*23;

  double delta=phi+0.174532925;
  if (delta<0) delta=delta+2*acos(-1);
  if (fabs(eta)<1.740) 
    {
      for (int i=0; i<72; i++)
	{
	  if (delta<(i+1)*0.087266462&&delta>i*0.087266462) iphi=i;
	}
    }
  else 
    {
      for (int i=0; i<36; i++)
	{
	  if (delta<2*(i+1)*0.087266462&&delta>2*i*0.087266462) iphi=2*i;
	}
    }

  return std::pair<int,int>(ieta,iphi);
}


ValidationHcalIsoTrackAlCaReco::ValidationHcalIsoTrackAlCaReco(const edm::ParameterSet& iConfig) :
  simTracksTag_(iConfig.getParameter<edm::InputTag>("simTracksTag"))
{
  folderName_ = iConfig.getParameter<std::string>("folderName");
  saveToFile_=iConfig.getParameter<bool>("saveToFile");
  outRootFileName_=iConfig.getParameter<std::string>("outputRootFileName");
  hltEventTag_=iConfig.getParameter<edm::InputTag>("hltTriggerEventLabel");
  hltFilterTag_=iConfig.getParameter<edm::InputTag>("hltL3FilterLabel");
  arITrLabel_=iConfig.getParameter<edm::InputTag>("alcarecoIsoTracksLabel");
  recoTrLabel_=iConfig.getParameter<edm::InputTag>("recoTracksLabel");
  pThr_=iConfig.getUntrackedParameter<double>("pThrL3",0);
  heLow_=iConfig.getUntrackedParameter<double>("lowerHighEnergyCut",40);
  heLow_=iConfig.getUntrackedParameter<double>("upperHighEnergyCut",60);

  nTotal=0;
  nHLTL3accepts=0;
}

ValidationHcalIsoTrackAlCaReco::~ValidationHcalIsoTrackAlCaReco()
{}

void ValidationHcalIsoTrackAlCaReco::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  nTotal++;

  edm::Handle<trigger::TriggerEvent> trEv;
  iEvent.getByLabel(hltEventTag_,trEv);
  
  edm::Handle<reco::IsolatedPixelTrackCandidateCollection> recoIsoTracks;
  iEvent.getByLabel(arITrLabel_,recoIsoTracks);

  const trigger::TriggerObjectCollection& TOCol(trEv->getObjects());
  
  trigger::Keys KEYS;
  const trigger::size_type nFilt(trEv->sizeFilters());
  for (trigger::size_type iFilt=0; iFilt!=nFilt; iFilt++) 
    {
      if (trEv->filterTag(iFilt)==hltFilterTag_) 
	{
	  KEYS=trEv->filterKeys(iFilt);
	}
    }
  
  trigger::size_type nReg=KEYS.size();
  
  std::vector<double> trigEta;
  std::vector<double> trigPhi;
  bool trig=false;

  //checks with IsoTrack trigger results
  for (trigger::size_type iReg=0; iReg<nReg; iReg++)
    {
      const trigger::TriggerObject& TObj(TOCol[KEYS[iReg]]);
      if (TObj.p()<pThr_) continue;
      hl3eta->Fill(TObj.eta(),1);
      hl3AbsEta->Fill(fabs(TObj.eta()),1);
      hl3phi->Fill(TObj.phi(),1);

      if (recoIsoTracks->size()>0)
	{
	  double minRecoL3dist=1000;
	  reco::IsolatedPixelTrackCandidateCollection::const_iterator mrtr;
	  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator rtrit=recoIsoTracks->begin(); rtrit!=recoIsoTracks->end(); rtrit++)
	    {
	      double R=getDist(rtrit->eta(),rtrit->phi(),TObj.eta(),TObj.phi()); 
	      if (R<minRecoL3dist) 
		{
		  mrtr=rtrit;
		  minRecoL3dist=R;
		}
	    }
	  hOffL3TrackMatch->Fill(minRecoL3dist,1);
	  hOffL3TrackPtRat->Fill(TObj.pt()/mrtr->pt(),1);
	}
      
      hl3Pt->Fill(TObj.pt(),1);
      trig=true; 
      trigEta.push_back(TObj.eta());
      trigPhi.push_back(TObj.phi());
    }

  //general distributions
  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator itr=recoIsoTracks->begin(); itr!=recoIsoTracks->end(); itr++)
    {
      bool match=false;
      for (unsigned int l=0; l<trigEta.size(); l++)
	{
	  if (getDist(itr->eta(),itr->phi(),trigEta[l],trigPhi[l])<0.2) match=true;
	}
      if (match&&trig)
	{	
	  hOffEtaFP->Fill(itr->eta(),1);
	  hOffPhiFP->Fill(itr->phi(),1);
	}
      
      hOffEta->Fill(itr->eta(),1);
      hOffPhi->Fill(itr->phi(),1);

      hOffAbsEta->Fill(fabs(itr->eta()),1);
   
      hDeposEcalInner->Fill(itr->energyIn(),1);
      hDeposEcalOuter->Fill(itr->energyOut(),1);

      hTracksSumP->Fill(itr->sumPtPxl(),1);
      hTracksMaxP->Fill(itr->maxPtPxl(),1);

      if (fabs(itr->eta())<0.5) hOffP_0005->Fill(itr->p(),1);
      if (fabs(itr->eta())>0.5&&fabs(itr->eta())<1.0) hOffP_0510->Fill(itr->p(),1);
      if (fabs(itr->eta())>1.0&&fabs(itr->eta())<1.5) hOffP_1015->Fill(itr->p(),1);
      if (fabs(itr->eta())<1.5&&fabs(itr->eta())<2.0) hOffP_1520->Fill(itr->p(),1);

      hOffP->Fill(itr->p(),1);

      std::pair<int,int> TI=towerIndex(itr->eta(),itr->phi());
      hOccupancyFull->Fill(TI.first,TI.second,1);
      if (itr->p()>heLow_&&itr->p()<heUp_) hOccupancyHighEn->Fill(TI.first,TI.second,1);
    }
    
// Sergey +

   std::cout <<  std::endl << "  End / Start " << std::endl;

   edm::Handle<edm::SimTrackContainer> simTracks;
   iEvent.getByLabel<edm::SimTrackContainer>(simTracksTag_, simTracks);

  for (reco::IsolatedPixelTrackCandidateCollection::const_iterator bll=recoIsoTracks->begin(); bll!=recoIsoTracks->end(); bll++)
    {         

    std::cout<<"ISO Pt " << bll->pt() <<  " P     "  << bll->p() << " Eta "<< bll->eta() << " Phi "<< bll->phi()<< std::endl;   
    
    double distanceMin = 1.;
    double SimPtMatched = 1.;
    double SimPhiMatched = 1.;
    double SimEtaMatched = 1.;
    double SimDistMatched = 1.;
    double SimPMatched = 1.;
    double neuen = 0.;
    double neuenm = 0.;
    int neun = 0;

   for(edm::SimTrackContainer::const_iterator tracksCI = simTracks->begin(); 
       tracksCI != simTracks->end(); tracksCI++){
        
	int partIndex = tracksCI->genpartIndex();
        if (tracksCI->momentum().eta() > (bll->eta()-0.1) 
	   && tracksCI->momentum().eta() < (bll->eta()+0.1)
	   && tracksCI->momentum().phi() > (bll->phi()-0.1) 
	   && tracksCI->momentum().phi() < (bll->phi()+0.1) 
//	   && tracksCI->momentum().e() > (0.5*bll->p())
//	   && tracksCI->momentum().e() < (2.*bll->p())
	   && tracksCI->momentum().e() > 2.
	   && fabs(tracksCI->charge()) == 1 && partIndex >0) 
	   {
	   
	double distance=getDist(tracksCI->momentum().eta(),tracksCI->momentum().phi(),bll->eta(),bll->phi());
	double distanceCM=getDistInCM(tracksCI->momentum().eta(),tracksCI->momentum().phi(),bll->eta(),bll->phi());
	
	if (distanceMin > distance) {
	  distanceMin = distance;
	  SimPtMatched = tracksCI->momentum().pt();
	  SimPhiMatched = tracksCI->momentum().phi();
	  SimEtaMatched = tracksCI->momentum().eta();
	  SimDistMatched = distance;
	  SimPMatched = sqrt(tracksCI->momentum().pt()*tracksCI->momentum().pt() + tracksCI->momentum().pz()*tracksCI->momentum().pz());
	}

        std::cout<<"    Pt "<<tracksCI->momentum().pt() 
	<< " Energy " << tracksCI->momentum().e() 
	<< " Eta "<< tracksCI->momentum().eta() 
	<< " Phi "<< tracksCI->momentum().phi() 
	<< " Ind "  << partIndex
	<< " Cha "  << tracksCI->charge() 
	<< " Dis "  << distance
	<< " DCM "  << distanceCM
	<< std::endl;       
	
       }
       
       	
        if (
	      tracksCI->momentum().eta() > (bll->eta()-0.5) 
	   && tracksCI->momentum().eta() < (bll->eta()+0.5)
	   && tracksCI->momentum().phi() > (bll->phi()-0.5) 
	   && tracksCI->momentum().phi() < (bll->phi()+0.5) 
//	   && tracksCI->momentum().e() > 2.
	   && tracksCI->charge() == 0 && partIndex >0) 
	   {	

	double distance=getDist(tracksCI->momentum().eta(),tracksCI->momentum().phi(),bll->eta(),bll->phi());
	double distanceCM=getDistInCM(tracksCI->momentum().eta(),tracksCI->momentum().phi(),bll->eta(),bll->phi());

        std::cout<<"NEU Pt "<<tracksCI->momentum().pt() 
	<< " Energy " << tracksCI->momentum().e() 
	<< " Eta "<< tracksCI->momentum().eta() 
	<< " Phi "<< tracksCI->momentum().phi() 
	<< " Ind "  << partIndex
	<< " Cha "  << tracksCI->charge() 
	<< " Dis "  << distance
	<< " DCM "  << distanceCM
	<< std::endl;
	
	 if (distanceCM < 40.){
	
	  neuen = neuen + tracksCI->momentum().e();
	  neun = neun + 1; 
	  if (neuenm < tracksCI->momentum().e()) neuenm = tracksCI->momentum().e();
	
	 } 
	
	} 
       
       }

          hSimNN->Fill(neun,1);
          hSimNE->Fill(neuen,1);
          hSimNM->Fill(neuenm,1);

	if (distanceMin < 0.1) {

          hSimPt->Fill(SimPtMatched,1);
          hSimPhi->Fill(SimPhiMatched,1);
          hSimEta->Fill(SimEtaMatched,1);
          hSimAbsEta->Fill(fabs(SimEtaMatched),1);
	  hSimDist->Fill(SimDistMatched,1);
	  hSimPtRatOff->Fill(SimPtMatched/bll->pt(),1);
          hSimP->Fill(SimPMatched,1);
          hSimN->Fill(1,1);

          std::cout<<"S    Pt "<<  SimPtMatched
	  << std::endl;
	}
	
	if (distanceMin > 0.1) {
          hSimN->Fill(0,1);
	}
	
	
	
	
  }

// Sergey -      
    
    
    
}

void ValidationHcalIsoTrackAlCaReco::beginJob()
{
  dbe_ = edm::Service<DQMStore>().operator->();
  dbe_->setCurrentFolder(folderName_);

  hl3Pt=dbe_->book1D("hl3Pt","pT of hlt L3 objects",1000,0,1000);
  hl3Pt->setAxisTitle("pT(GeV)",1);
  hl3eta=dbe_->book1D("hl3eta","eta of hlt L3 objects",16,-2,2);
  hl3eta->setAxisTitle("eta",1);
  hl3AbsEta=dbe_->book1D("hl3AbsEta","|eta| of hlt L3 objects",8,0,2);
  hl3AbsEta->setAxisTitle("eta",1);
  hl3phi=dbe_->book1D("hl3phi","phi of hlt L3 objects",16,-3.2,3.2);
  hl3phi->setAxisTitle("phi",1);

  hOffEta=dbe_->book1D("hOffEta","eta of alcareco objects",100,-2,2);
  hOffEta->setAxisTitle("eta",1);
  hOffPhi=dbe_->book1D("hOffPhi","phi of alcareco objects",100,-3.2,3.2);
  hOffPhi->setAxisTitle("phi",1);
  hOffP=dbe_->book1D("hOffP","p of alcareco objects",1000,0,1000);
  hOffP->setAxisTitle("E(GeV)",1);
  hOffP_0005=dbe_->book1D("hOffP_0005","p of alcareco objects, |eta|<0.5",1000,0,1000);
  hOffP_0005->setAxisTitle("E(GeV)",1);
  hOffP_0510=dbe_->book1D("hOffP_0510","p of alcareco objects, 0.5<|eta|<1.0",1000,0,1000);
  hOffP_0510->setAxisTitle("E(GeV)",1);
  hOffP_1015=dbe_->book1D("hOffP_1015","p of alcareco objects, 1.0<|eta|<1.5",1000,0,1000);
  hOffP_1015->setAxisTitle("E(GeV)",1);
  hOffP_1520=dbe_->book1D("hOffP_1520","p of alcareco objects, 1.5<|eta|<2.0",1000,0,1000);
  hOffP_1520->setAxisTitle("E(GeV)",1);
  hOffEtaFP=dbe_->book1D("hOffEtaFP","eta of alcareco objects, FP",16,-2,2);
  hOffEtaFP->setAxisTitle("eta",1);
  hOffAbsEta=dbe_->book1D("hOffAbsEta","|eta| of alcareco objects",8,0,2);
  hOffAbsEta->setAxisTitle("|eta|",1);
  hOffPhiFP=dbe_->book1D("hOffPhiFP","phi of alcareco objects, FP",16,-3.2,3.2);
  hOffPhiFP->setAxisTitle("phi",1);
  hTracksSumP=dbe_->book1D("hTracksSumP","summary p of tracks in the isolation cone",100,0,20);
  hTracksSumP->setAxisTitle("E(GeV)");
  hTracksMaxP=dbe_->book1D("hTracksMaxP","maximum p among tracks in the isolation cone",100,0,20);
  hTracksMaxP->setAxisTitle("E(GeV)");

  hDeposEcalInner=dbe_->book1D("hDeposEcalInner","ecal energy deposition in inner cone around track",1000,0,1000);
  hDeposEcalInner->setAxisTitle("E(GeV)");
  hDeposEcalOuter=dbe_->book1D("hDeposEcalOuter","ecal energy deposition in outer cone around track",1000,0,1000);
  hDeposEcalOuter->setAxisTitle("E(GeV)");

  hOccupancyFull=dbe_->book2D("hOccupancyFull","number of tracks per tower, full energy range",48,-25,25,73,0,73);
  hOccupancyFull->setAxisTitle("ieta",1);
  hOccupancyFull->setAxisTitle("iphi",2);
  hOccupancyFull->getTH2F()->SetOption("colz");
  hOccupancyFull->getTH2F()->SetStats(kFALSE);
  hOccupancyHighEn=dbe_->book2D("hOccupancyHighEn","number of tracks per tower, high energy tracks",48,-25,25,73,0,73);
  hOccupancyHighEn->setAxisTitle("ieta",1);
  hOccupancyHighEn->setAxisTitle("iphi",2);
  hOccupancyHighEn->getTH2F()->SetOption("colz");
  hOccupancyHighEn->getTH2F()->SetStats(kFALSE);

  hOffL3TrackMatch=dbe_->book1D("hOffL3TrackMatch","Distance from L3 object to offline track",200,0,0.5);
  hOffL3TrackMatch->setAxisTitle("R(eta,phi)",1);
  hOffL3TrackPtRat=dbe_->book1D("hOffL3TrackPtRat","Ratio of pT: L3/offline",100,0,10);
  hOffL3TrackPtRat->setAxisTitle("ratio L3/offline",1);


// Sergey +

  hSimPt=dbe_->book1D("hSimPt","pT of matched SimTrack",1000,0,1000);
  hSimPt->setAxisTitle("pT(GeV)",1);

  hSimPhi=dbe_->book1D("hSimPhi","Phi of matched SimTrack",100,-3.2,3.2);
  hSimPhi->setAxisTitle("phi",1);
  
  hSimEta=dbe_->book1D("hSimEta","Eta of matched SimTrack",100,-2.,2.);
  hSimEta->setAxisTitle("eta",1);
  
  hSimAbsEta=dbe_->book1D("hSimAbsEta","|eta| of matched SimTrack",8,0.,2.);
  hSimAbsEta->setAxisTitle("|eta|",1);

  hSimDist=dbe_->book1D("hSimDist","Distance from matched SimTrack to Offline Track",200,0,0.1);
  hSimDist->setAxisTitle("R(eta,phi)",1);

  hSimPtRatOff=dbe_->book1D("hSimPtRatOff","pT Sim / pT Offline",100,0,10);
  hSimPtRatOff->setAxisTitle("pT Sim / pT Offline",1);

  hSimP=dbe_->book1D("hSimP","p of matched SimTrack",1000,0,1000);
  hSimP->setAxisTitle("p(GeV)",1);
  
  hSimN=dbe_->book1D("hSimN","Number matched",2,0,2);
  hSimN->setAxisTitle("Offline/SimTRack - Matched or Not",1);

  hSimNN=dbe_->book1D("hSimNN","Number of the neutral particles in cone on ECAL",100,0,100);
  hSimNN->setAxisTitle("Number",1);

  hSimNE=dbe_->book1D("hSimNE","Total energy of the neutral particles in cone on ECAL",100,0,100);
  hSimNE->setAxisTitle("Energy",1);

  hSimNM=dbe_->book1D("hSimNM","Maximum energy of the neutral particles in cone on ECAL",100,0,100);
  hSimNM->setAxisTitle("Energy",1);
  
// Sergey -


}

void ValidationHcalIsoTrackAlCaReco::endJob() {

if(dbe_) 
  {
    if (saveToFile_) dbe_->save(outRootFileName_);
  }
}

