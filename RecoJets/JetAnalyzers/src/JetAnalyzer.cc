//
//  Example analysis related to calorimeter and jets
//  Anal A Bhatti The Rockefeller University 
//  Based on example from Leonard Apanasevich  University of Illinois Chicago
//  August 30, 2006
//
//  
#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>

#include "RecoJets/JetAnalyzers/interface/JetAnalyzer.h"
#include "RecoJets/JetAnalyzers/interface/JetUtil.h"

#include "RecoJets/JetAnalyzers/interface/CaloTowerBoundries.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#include "RecoJets/Geom/interface/TTrajectoryPoint.hh"
//#include "RecoJets/Geom/interface/TSimpleExtrapolator.hh"

//#include "RecoJets/JetAlgorithms/interface/LorentzVector.h"
//#include "RecoJets/JetAlgorithms/interface/PhysicsTower.h"
//#include "RecoJets/JetAlgorithms/interface/Cluster.h"
//#include "RecoJets/JetAlgorithms/interface/MidPointAlgorithm.h"

#include <sstream>
#include <stdlib.h>
#include <string.h>

#include <CLHEP/Vector/TwoVector.h>

typedef CaloJetCollection::const_iterator CalJetIter;
typedef GenJetCollection::const_iterator GenJetIter;

using namespace reco;


double radius(const CalCell& i,const CalCell& j){
  return radius(i.Momentum.eta(),i.Momentum.phi(),j.Momentum.eta(),j.Momentum.phi());
}
double radius(const CalCluster& i,const CalCluster& j){
  return radius(i.Momentum.eta(),i.Momentum.phi(),j.Momentum.eta(),j.Momentum.phi());
}

//double CorrectedE(double E_Em,double E_Hd,int ifit=1);

double CorrectedE(double E_Em,double E_Hd,int ifit,double& f,double& EoHEm,double& EoHHd,double& EoPiEm,double& EoPiHd);

JetAnalyzer::JetAnalyzer(const edm::ParameterSet& pSet) {
  m_file=0; // set to null
  evtCounter=0;

  //set parameter defaults 

  _EtaMin=-5.2;
  _EtaMax=5.2;
  _HistName="test.root"; 

  _Monte=true;
  _PlotTrigger=false;
  _PlotRecHits=true;
  _PlotDigis=true;
  _PlotDijets=true;
  _PlotMCParticles=true;
  _PlotLocalClusters=true;

  ExcludeInteractions_=false;

  // If your module takes parameters, here is where you would define
  // their names and types, and access them to initialize internal
  // variables. Example as follows:
  //

  std::cout << " Beginning JetAnalyzer Analysis " << std::endl;

  calojets_   = pSet.getParameter< std::string > ("calojets");
  genJets_    = pSet.getParameter< std::string > ("genjets");
  recmet_     = pSet.getParameter< std::string > ("recmet");
  genmet_     = pSet.getParameter< std::string > ("genmet");
  calotowers_ = pSet.getParameter< std::string > ("calotowers");

  //  GenJetPtBins_ =cfg.getParameter< std::vector<double> >("GenJetPtBins");
  //  std::vector<double> EtaBins =cfg.getParameter< std::vector<double> >("RecJetEtaBins");

  errCnt=0;

  edm::ParameterSet myJetParams = pSet.getParameter<edm::ParameterSet>("RunParameters");
  vector<std::string> parameterNames = myJetParams.getParameterNames() ;
  
  for ( vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "EtaMin" ) _EtaMin =  myJetParams.getParameter<double>( *iParam );
    else if ( (*iParam) == "EtaMax" ) _EtaMax =  myJetParams.getParameter<double>( *iParam );
    else if ( (*iParam) == "HistogramFile" ) _HistName =  myJetParams.getParameter<string>( *iParam );
    else if ( (*iParam) == "PlotRecHits" ) _PlotRecHits =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "PlotDigis" ) _PlotDigis =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "PlotTrigger" ) _PlotTrigger =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "PlotDijets" ) _PlotDijets =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "PlotMCParticles" ) _PlotMCParticles =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "PlotLocalClusters" ) _PlotLocalClusters =  myJetParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "ExcludeInteractions" ) ExcludeInteractions_ =  myJetParams.getParameter<bool>( *iParam );
  }

  cout << "---------- Input Parameters ---------------------------" << endl;
  cout << "  Monte:  " << _Monte << endl;    
  cout << "  EtaMin: " << _EtaMin << endl;    
  cout << "  EtaMax: " << _EtaMax << endl;    
  cout << "  Output histograms written to: " << _HistName << std::endl;
  cout << "-------------------------------------------------------" << endl;  

  if (_PlotMCParticles){
    cout << "Sorry, PlotMCParticles option has been disabled for now" << endl;
    _PlotMCParticles = false;
  }
  if (_PlotLocalClusters){
    cout << "Sorry, PlotLocalClusters option has been disabled for now" << endl;
    _PlotLocalClusters = false;
  }

  // open the histogram file

  m_file=new TFile(_HistName.c_str(),"RECREATE");
  m_file->mkdir(EnergyDir());
  m_file->mkdir(PulseDir());
  m_file->cd();
  bookHistograms();

  for(int ieta=0;ieta<NETA+1;ieta++){
    cout << " ieta " << ieta << " eta min " << CaloTowerEtaBoundries[ieta] <<endl;
  }
}
void JetAnalyzer::analyze(edm::Event const& evt, edm::EventSetup const& iSetup) {

   edm::ESHandle<CaloGeometry> geometry;
   iSetup.get<CaloGeometryRecord>().get(geometry);

  // These declarations create handles to the types of records that you want
  // to retrieve from event "evt".
  //

  edm::Handle<edm::HepMCProduct> genEventHandle;
  edm::Handle<edm::SimVertexContainer> simVertexContainer;
  edm::Handle<edm::SimTrackContainer>  simTrackContainer;

  //  edm::Handle<CaloJetCollection> calojets;
  //  edm::Handle<GenJetCollection>  genjets;
  //  edm::Handle<CaloMETCollection> recmet;
  //  edm::Handle<GenMETCollection>  genmet;
  //  edm::Handle<CaloTowerCollection> caloTowers;
  //  edm::Handle<CaloTowerCollection> caloTowers;
  //   edm::Handle<EBRecHitCollection> EBRecHits;
  //   edm::Handle<EERecHitCollection> EERecHits;
  //   edm::Handle<HBHERecHitCollection> HBHERecHits;
  //   edm::Handle<HORecHitCollection> HORecHits;
  //   edm::Handle<HFRecHitCollection> HFRecHits;
  //  edm::Handle<HBHEDigiCollection> HBHEDigis;
  //   edm::Handle<HODigiCollection> HODigis;
  //   edm::Handle<HFDigiCollection> HFDigis;
  //   edm::Handle<HcalTBTriggerData> trigger;

  edm::Handle<std::vector<reco::PFCluster> > PFCluster;
  //  edm::Handle< vector<reco::PFCluster> > clustersECAL;

  // Data objects

  string errMsg("");

  evt.getByLabel (calojets_,calojets);
  evt.getByLabel (genmet_,genmet);
  evt.getByLabel (recmet_,recmet);

  std::string pfClusterModuleLabel="particleFlowCluster";
  std::string pfClusterECALInstanceName= "ECAL";


  //  try {
  //   evt.getByLabel(pfClusterModuleLabel,pfClusterECALInstanceName,PFCluster);
  // } catch( const cms::Exception& e) {
  //  errMsg = errMsg + "------>>>> No PF Clusters because\n"+e.what();
  // }


  //  evt.getByLabel("particleFlowCluster","ECAL",PFCluster);

  //  int nPFClusters=PFCluster->size();
  // cout << " PF Clusters " << PFCluster->size() << endl;

  //  std::vector<reco::PFCluster> PFC = *PFCluster;

  //for(int i=0;i<nPFClusters;i++){
  //  cout << i << "  " << PFC[i].energy() <<endl;
  // }

  evt.getByLabel (calotowers_,caloTowers);
  if (! caloTowers.isValid() ) { errMsg=errMsg + "  -- No CaloTowers" ; caloTowers=caloTowersDummy;}

  if(_PlotRecHits){
    evt.getByLabel( "ecalRecHit","EcalRecHitsEB", EBRecHits );
    evt.getByLabel( "ecalRecHit","EcalRecHitsEE", EERecHits );
    evt.getByLabel( "hbhereco", HBHERecHits );
    evt.getByLabel( "hfreco", HFRecHits );
    evt.getByLabel( "horeco", HORecHits );

    if (! EBRecHits.isValid() ) { errMsg=errMsg + "  -- No EB Rechits" ; EBRecHits=EBRecHitsDummy;}
    if (! EERecHits.isValid() ) { errMsg=errMsg + "  -- No EE Rechits" ; EERecHits=EERecHitsDummy;}
    if (! HBHERecHits.isValid() ) { errMsg=errMsg + "  -- No HBHE Rechits" ; HBHERecHits=HBHERecHitsDummy;}
    if (! HFRecHits.isValid() ) { errMsg=errMsg + "  -- No HF Rechits" ; HFRecHits=HFRecHitsDummy;}
    if (! HORecHits.isValid() ) { errMsg=errMsg + "  -- No HO Rechits" ; HORecHits=HORecHitsDummy;}
  }

  if(_PlotDigis) {
    evt.getByLabel( "hcalDigis", HBHEDigis );
    evt.getByLabel( "hcalDigis", HODigis );
    evt.getByLabel( "hcalDigis", HFDigis );

    if (! HBHEDigis.isValid() ) { errMsg=errMsg + "  -- No HBHE digis" ; HBHEDigis=HBHEDigisDummy;}
    if (! HODigis.isValid() ) { errMsg=errMsg + "  -- No HO digis" ; HODigis=HODigisDummy;}
    if (! HFDigis.isValid() ) { errMsg=errMsg + "  -- No HF digis" ; HFDigis=HFDigisDummy;}
  }

  // Trigger Information
  
  if(_PlotTrigger){
    evt.getByType(trigger);
    if (! trigger.isValid() ) { errMsg=errMsg + "  -- No TB Trigger info" ; trigger=triggerDummy;}
  }


  if(_PlotMCParticles) {
    try {
      evt.getByLabel("source","",genEventHandle);
      genEvent = genEventHandle->getHepMCData();
      evt.getByLabel("genParticleCandidates",genParticles);
  
    } catch (...) {
      errMsg=errMsg + "  -- No MC truth";
    }
  }

  if(_Monte) {

    evt.getByLabel ("g4SimHits",simVertexContainer);
    simVertex = *simVertexContainer;

    evt.getByLabel ("g4SimHits",simTrackContainer);
    simTrack = *simTrackContainer;


    std::vector<edm::Handle<std::vector<PCaloHit> > > caloSimHitHandle;

    evt.getManyByType(caloSimHitHandle);

    //    int sc=caloSimHitHandle.size();

//      for (int ii=0;ii<sc;ii++){
//        edm::BranchDescription desc = caloSimHitHandle[ii].provenance()->product;
//        const std::string subdet = desc.productInstanceName_;
//        cout << " subDet name " << subdet << endl;

//        const edm::PCaloHitContainer *calohits = caloSimHitHandle[ii].product();
//        caloSimHits_.insert(std::map <std::string,edm::PCaloHitContainer>::value_type(subdet,*calohits));
//      }




//     int simvert_size = SimVertexContainer->size();
//     for (int j=0;j<simvert_size;++j){
//       cout << "************************" << sim_vert[j].position().x()<<endl;
//     }

    try {
      evt.getByLabel (genJets_,genJets);
    } catch (...) {
      errMsg=errMsg + "  -- No GenJets";
    }

    try {
      evt.getByLabel (genmet_,genmet);
    } catch (...) {
      errMsg=errMsg + "  No -- GenMet by Label";
    }
  }

  if ((errMsg != "") && (errCnt < errMax())){
    errCnt=errCnt+1;
    errMsg=errMsg + ".";
    std::cout << "%JetAnalyzer-Warning" << errMsg << std::endl;
    if (errCnt == errMax()){
      errMsg="%JetAnalyzer-Warning -- Maximum error count reached -- No more messages will be printed.";
      std::cout << errMsg << std::endl;    
    }
  }

  //std::cout << " Beginning JetAnalyzer " << std::endl;
  //  cout <<  " Number of towers " <<  (*caloTowers).size() <<endl;

  // Now analyzed the event

  analyze(*calojets,*genJets,*recmet,*genmet,*caloTowers,genEvent,*EBRecHits,*EERecHits,
  	  *HBHERecHits,*HBHEDigis,*HORecHits,*HODigis,*HFRecHits,*HFDigis,*trigger,*geometry);
}
void JetAnalyzer::endJob() {
  done();
}
void JetAnalyzer::fillHist1D(const TString histName, const Double_t value, const Double_t wt) {
  std::map<TString, TH1*>::iterator hid;
  hid=m_HistNames.find(histName);
  if (hid==m_HistNames.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(value,wt); 
}

void JetAnalyzer::fillHist2D(const TString histName, const Double_t x,const Double_t y,const Double_t wt) {
  std::map<TString, TH2*>::iterator hid;
  hid=m_HistNames2D.find(histName);
  if (hid==m_HistNames2D.end())
    std::cout << "%fillHist -- Could not find histogram with name: " << histName << std::endl;
  else
    hid->second->Fill(x,y,wt); 
}

void JetAnalyzer::bookHistograms() {

  bookGeneralHistograms();

  bookFillEoPCorrectionPlots();

  bookTBTriggerHists();

  bookCaloTowerHists();

  if (_Monte) bookJetHistograms("Gen");
  bookJetHistograms("Calo");

  if (_Monte) bookMetHists("Gen");
  bookMetHists("Calo");

  if (_Monte) bookCalculateEfficiency();

  if (_Monte) bookDiJetBalance("Gen");
  bookDiJetBalance("Calo");

  if (_Monte) bookMCParticles();

  bookSubClusterHistograms();

  bookPtSpectrumInAJet();

  bookRecHitHists("EB");
  bookRecHitHists("EE");
  bookRecHitHists("HB");
  bookRecHitHists("HE");
  bookRecHitHists("HFS");
  bookRecHitHists("HFL");
  bookRecHitHists("HO");

}
/** Analyze the hits */
void JetAnalyzer::analyze( const CaloJetCollection& calojets,
			   const GenJetCollection& genJets,
			   const CaloMETCollection& recmets,
			   const GenMETCollection& genmets,
			   const CaloTowerCollection& caloTowers,
			   const HepMC::GenEvent genEvent,
			   const EBRecHitCollection& EBRecHits,
			   const EERecHitCollection& EERecHits,
			   const HBHERecHitCollection& HBHERecHits, 
			   const HBHEDigiCollection& HBHEDigis, 
			   const HORecHitCollection& HORecHits, 
			   const HODigiCollection& HODigis, 
			   const HFRecHitCollection& HFRecHits, 
			   const HFDigiCollection& HFDigis, 
			   const HcalTBTriggerData& trigger,
  			   const CaloGeometry& caloGeometry){


  // count number of events being analyzed 

  TString hname="Nevents";
  fillHist1D(hname,1.0);


  int nSimVertex = simVertex.size();
  int nSimTrack  = simTrack.size();

  cout << " Number of simVertices " << nSimVertex << "  " <<  nSimTrack << endl; 

  //  for (int j=0;j<nSimVertex;++j){
  //    cout << "************************" << simVertex[j].position().x()<<endl;
  // }


  if(ExcludeInteractions_) {
    if(nSimTrack>1 || nSimVertex > 1) return;
  }

  if(_PlotTrigger) {
    if(&trigger) fillTBTriggerHists(trigger);
  }


  // fill calojet and genjet hists 
  fillJetHists(calojets,"Calo");
  if (&genJets)fillJetHists(genJets,"Gen");

  // fill recmet and genjet hists
  fillMetHists(recmets,"Calo");

  if (&genmets) fillMetHists(genmets,"Gen");

  if(&caloTowers) fillCaloTowerHists(caloTowers);

  if(&genJets) CalculateEfficiency(genJets, calojets);

  if(_PlotDijets) {
    DiJetBalance(calojets,"Calo");
    if (&genJets) DiJetBalance(genJets,"Gen");
  }

  if(_PlotMCParticles) {
    if(&genEvent) fillMCParticles(genParticles);
  }

  // Plot RecHits
  if (_PlotRecHits){
    if (&HBHERecHits) fillRecHits(HBHERecHits);
    if (&HORecHits) fillRecHits(HORecHits);
    if (&HFRecHits) fillRecHits(HFRecHits);

    double sumEB(0.0);
    double sumEE(0.0);
    double sumHB(0.0);
    double sumHE(0.0);
    double sumHO(0.0);

    fillRecHitHists(caloGeometry,EBRecHits,"EB",sumEB);
    fillRecHitHists(caloGeometry,EERecHits,"EE",sumEE);
    fillRecHitHists(caloGeometry,HBHERecHits,"HB",sumHB);
    fillRecHitHists(caloGeometry,HBHERecHits,"HE",sumHE);
    fillRecHitHists(caloGeometry,HORecHits,"HO",sumHO);

    double sumEmRecHits=sumEB+sumEE;
    double sumHadRecHits=sumHB+sumHE+sumHO;
    double sumEnergyRecHits=sumEmRecHits+sumHadRecHits;
    
    hname="RecHitsSumEmEnergy";
    fillHist1D(hname,sumEmRecHits);
    
    hname="RecHitsSumHadEnergy";
    fillHist1D(hname,sumHadRecHits);
    
    hname="RecHitsSumEnergy";
    fillHist1D(hname,sumEnergyRecHits);
    
    if(sumEmRecHits<0.6) {
      hname="RecHitsSumEmEnergyMIP";
      fillHist1D(hname,sumEmRecHits);
      
      hname="RecHitsSumHadEnergyMIP";
      fillHist1D(hname,sumHadRecHits);
      
      hname="RecHitsSumEnergyMIP";
      fillHist1D(hname,sumEnergyRecHits);
    }
    
    IsItMIP_ = (sumEB+sumEE<0.6) ? true : false;
  
    MakeLocalClusters(caloGeometry,calojets,recmets,caloTowers,EBRecHits,EERecHits,HBHERecHits,HORecHits,HFRecHits);

  }

  // Plot Digis
  if (_PlotDigis){
    if (&HBHEDigis) fillDigis(HBHEDigis);
    if (&HODigis)   fillDigis(HODigis);
    if (&HFDigis)   fillDigis(HFDigis);
  }

  // local clustering using caloTowers

  if(_PlotLocalClusters) {    
    if(&genEvent && &genJets){
      std::vector<Candidate*> ParentParton;
      GetParentPartons(ParentParton);
      PtSpectrumInSideAJet(genJets,genEvent);
    }
  }

}

void JetAnalyzer::bookGeneralHistograms() {

  TString hname; TString htitle;
  hname="Nevents"; htitle="Number of events";
  m_HistNames[hname]= new TH1F(hname,htitle,5,0.0,5.0);

  hname="EoPCorrections";
  htitle="EoPCorrections";

  m_HistNames2D[hname] = new TH2F(hname,htitle,100,0.0,100.,100,0.0,100.);

}

void JetAnalyzer::bookTBTriggerHists() {


  TString hname="Trigger"; TString htitle="Trigger Histogram";

  TH1F* id =new TH1F(hname,htitle,5,0.0,5.0);
  id->GetXaxis()->SetBinLabel(1,trigBeam());
  id->GetXaxis()->SetBinLabel(2,trigIped());
  id->GetXaxis()->SetBinLabel(3,trigOped());
  id->GetXaxis()->SetBinLabel(4,trigLED());
  id->GetXaxis()->SetBinLabel(5,trigLaser());
  m_HistNames[hname]= id;
}

// void JetAnalyzer::fillTBTriggerHists(const HcalTBTriggerData& trigger){

//   TString hname="Trigger";

//   hid=m_HistNames.find(hname);
//   if (hid==m_HistNames.end())
//     std::cout << "%fillHist -- Could not find histogram with name: " << hname << std::endl;
//   else
//     {
//       if (trigger.wasBeamTrigger()) hid->second->Fill(trigBeam(),1.0);
//       if (trigger.wasInSpillPedestalTrigger()) hid->second->Fill(trigIped(),1.0);
//       if (trigger.wasOutSpillPedestalTrigger()) hid->second->Fill(trigOped(),1.0);
//       if (trigger.wasLEDTrigger()) hid->second->Fill(trigLED(),1.0);
//       if (trigger.wasLaserTrigger()) hid->second->Fill(trigLaser(),1.0);
//     }
// }

void JetAnalyzer::bookJetHistograms(const TString& prefix) {

  TString hname;
  TString htitle;

  std::ostringstream ch_eta; ch_eta << etaBarrel();
  std::ostringstream ch_etamin; ch_etamin << _EtaMin;
  std::ostringstream ch_etamax; ch_etamax << _EtaMax;

  hname=prefix + "NumberOfJets";
  htitle=prefix + "NumberOfJets";
  m_HistNames[hname]= new TH1F(hname,htitle,100,0.0,100.0);


  hname=prefix + "NumberOfConstituents";
  htitle=prefix + "NumberOfConstituents";
  m_HistNames[hname]= new TH1F(hname,htitle,100,0.0,100.0);

  hname =prefix + "JetEt";
  htitle=prefix + "JetEt";
  m_HistNames[hname]= new TH1F(hname,htitle,700,0.0,7000.);

  hname =prefix + "JetPt";
  htitle=prefix + "JetPt";
  m_HistNames[hname]= new TH1F(hname,htitle,700,0.0,7000.);


  hname =prefix + "JetEmf";
  htitle=prefix + "JetEmf";
  m_HistNames[hname]= new TH1F(hname,htitle,100,0.0,1.0);


  hname =prefix + "JetEnergy";
  htitle=prefix + "JetEnergy";
  m_HistNames[hname]= new TH1F(hname,htitle,700,0.0,7000.);


  hname=prefix + "JetPhi";
  htitle=prefix + "JetPhi";
  m_HistNames[hname] = new TH1F(hname,htitle,72,-M_PI,M_PI);

  hname=prefix + "JetEta";
  htitle=prefix + "JetEta";
  m_HistNames[hname] = new TH1F(hname,htitle,110,-5.5,5.5);


  hname=prefix + "EtJetBarrel";
  htitle=prefix + "EtJeBarrel";
  m_HistNames[hname] = new TH1F(hname,htitle,700,0.0,7000.0);

  hname=prefix + "PtJetBarrel";
  htitle=prefix + "PtJetBarrel";
  m_HistNames[hname] = new TH1F(hname,htitle,700,0.0,7000.0);

  hname=prefix + "JetPtMaxBarrel";
  htitle=prefix + "JetPtMaxBarrel";
  m_HistNames[hname] = new TH1F(hname,htitle,700,0.0,7000.0);

  hname=prefix + "JetEtMaxBarrel";
  htitle=prefix + "JetEtMaxBarrel";
  m_HistNames[hname] = new TH1F(hname,htitle,700,0.0,7000.0);

}

void JetAnalyzer::bookForId(const HcalDetId& id) {

  std::ostringstream ss,ss_t1,ss_t2,st;

  ss << "Energy " << id;
  ss_t1 << "LowEnergy" << id;  ss_t1 << "LowEnergy" << id;  ss_t1 << "LowEnergy" << id;


  ss_t2 << "Energy" << id;

  //std::cout << "Booking Histograms for HCAL Subdetector: " << id.subdet() 
  //	    << " Name: " << ss.str() << std::endl;

  Int_t nbins;
  Double_t xmin,xmax;

  switch (id.subdet()){
  case (HcalBarrel):{
    nbins=300; xmin=0.; xmax=600.;
  } break;
  case (HcalOuter):{
    nbins=100; xmin=0.; xmax=100.;
  } break;
  default:{
    nbins=300; xmin=0.; xmax=600.;
  }
  }
  channelmap1[id]=new TH1F(ss_t1.str().c_str(),ss.str().c_str(),520,-3.0,10.0);
  channelmap2[id]=new TH1F(ss_t2.str().c_str(),ss.str().c_str(),nbins,xmin,xmax);  
}

void JetAnalyzer::bookForId_TS(const HcalDetId& id) {

  std::ostringstream ss,st;
  ss << "ADC" << id;
  st << "fC" << id;

  digimap1[id]=new TH1I(ss.str().c_str(),ss.str().c_str(),10,-0.5,9.5);
  digimap2[id]=new TH1F(st.str().c_str(),st.str().c_str(),10,-0.5,9.5);

}

template <typename T> void JetAnalyzer::fillRecHits(const T& hits) {

  std::map<HcalDetId,TH1*>::iterator h1,h2;

  m_file->cd(EnergyDir());
  typedef typename T::const_iterator iter;
  for ( iter i=hits.begin(); i!=hits.end(); i++) {

    h1=channelmap1.find(i->id()); // look for a histogram with this hit's id
    h2=channelmap2.find(i->id()); // look for a histogram with this hit's id
    if (h1==channelmap1.end()) {
      bookForId(i->id());
      h1=channelmap1.find(i->id()); // look for a histogram with this hit's id
      h2=channelmap2.find(i->id()); // look for a histogram with this hit's id
    }
    //std::cout << "Energy: " << i->energy() << " hist1: " << h1->second->GetTitle() << endl;
    h1->second->Fill(i->energy()); // if it's there, fill it with energy
    h2->second->Fill(i->energy()); // if it's there, fill it with energy
  }
  m_file->cd();
}


template <typename T> void JetAnalyzer::fillDigis(const T& digis) {

  std::map<HcalDetId,TH1*>::iterator h1,h2;

  m_file->cd(PulseDir());
  typedef typename T::const_iterator iter;

  for ( iter ii=digis.begin(); ii!=digis.end(); ii++) {
    h1=digimap1.find(ii->id()); // look for a histogram with this hit's id
    h2=digimap2.find(ii->id()); // look for a histogram with this hit's id
    if (h1==digimap1.end()) {
      bookForId_TS(ii->id());
      h1=digimap1.find(ii->id()); // look for a histogram with this hit's id
      h2=digimap2.find(ii->id()); // look for a histogram with this hit's id
    }
    for (int indx=0; indx<ii->size(); indx++) {
      int i_adc=ii->sample(indx).adc();
      //std::cout << " LA: " << ii->sample(indx) << ii->sample(indx).capid() << std::endl;
      h1->second->Fill(indx,i_adc);
      Double_t wt=ii->sample(indx).nominal_fC();
      //std::cout << i_adc << " " << wt << std::endl;
      h2->second->Fill(indx,wt);
    }
  }
  m_file->cd();
}

template <typename T> void JetAnalyzer::fillJetHists(const T& jets, const TString& prefix) {

  typedef typename T::const_iterator iter;

  double maxPt(0.0);
  double maxEt(0.0);
  
  int njets=jets.size();

  fillHist1D(prefix + "NumberOfJets",njets);

  for ( iter ijet=jets.begin(); ijet!=jets.end(); ijet++) {

    Double_t jetEnergy = ijet->energy();
    Double_t jetEt = ijet->et();
    Double_t jetPt = ijet->pt();
    Double_t jetEta = ijet->eta();
    Double_t jetPhi = ijet->phi();

    //    const std::vector<CaloTowerDetId>&  barcodes=ijet->getTowerIndices();
    //   int nConstituents= barcodes.size();

    //  std::vector <CaloTowerRef> constituents = ijet->getConstituents ();
    // int nConstituents= constituents.size();

    //  if(prefix=="Calo") Double_t emEnergyFraction  = ijet->emEnergyFraction();
//     Double_t emEnergyInEB = ijet->emEnergyInEB();
//     Double_t emEnergyInEE = ijet->emEnergyInEE();

//     Double_t hdEnergyInHB = ijet->hadEnergyInHB();
//     Double_t emergyInHO   = ijet->hadEnergyInHO();
//     Double_t emEnergyInHF = ijet->jet->emEnergyInHF();
//     Double_t hdEnergyInHF = ijet->jet->hadEnergyInHF();



    if (jetEta > _EtaMin && jetEta < _EtaMax){
      fillHist1D(prefix + "JetEnergy",jetEnergy);
      fillHist1D(prefix + "JetEt",jetEt);
      fillHist1D(prefix + "JetPt",jetPt);

      //      fillHist1D(prefix + "JetEmf",emEnergyFraction);

      fillHist1D(prefix + "JetEta",jetEta);
      fillHist1D(prefix + "JetPhi",jetPhi);

      if (fabs(jetEta) < etaBarrel()){
	fillHist1D(prefix + "EtJetBarrel",jetEt);
	fillHist1D(prefix + "PtJetBarrel",jetPt);
	if (jetEt > maxEt) maxEt = jetEt;
	if (jetPt > maxPt) maxPt = jetPt;
      }
    }
  }
  fillHist1D(prefix + "JetEtMaxBarrel",maxEt);
  fillHist1D(prefix + "JetPtMaxBarrel",maxPt);
}

void JetAnalyzer::fillJetHists(const CaloJetCollection& jets,const TString& prefix) {

				  //  typedef typename T::const_iterator iter;

  typedef CaloJetCollection::const_iterator iter;

  TString hname;

  double maxPt(0.0);
  double maxEt(0.0);
  
  int njets=jets.size();

  fillHist1D(prefix + "NumberOfJets",njets);

  for ( iter ijet=jets.begin(); ijet!=jets.end(); ijet++) {

    Double_t jetEnergy = ijet->energy();
    Double_t jetEt = ijet->et();
    Double_t jetPt = ijet->pt();
    Double_t jetEta = ijet->eta();
    Double_t jetPhi = ijet->phi();

    //    const std::vector<CaloTowerDetId>&  barcodes=ijet->getTowerIndices();
    //  int nConstituents= barcodes.size();

    std::vector <CaloTowerRef> constituents = ijet->getConstituents ();
    int nConstituents= constituents.size();


    Double_t emEnergyFraction  = ijet->emEnergyFraction();

//     Double_t emEnergyInEB = ijet->emEnergyInEB();
//     Double_t emEnergyInEE = ijet->emEnergyInEE();

//     Double_t hdEnergyInHB = ijet->hadEnergyInHB();
//     Double_t emergyInHO   = ijet->hadEnergyInHO();
//     Double_t emEnergyInHF = ijet->jet->emEnergyInHF();
//     Double_t hdEnergyInHF = ijet->jet->hadEnergyInHF();



//    if (jetEta > _EtaMin && jetEta < _EtaMax){

      hname=prefix + "NumberOfConstituents";
      fillHist1D(hname,nConstituents);

      fillHist1D(prefix + "JetEnergy",jetEnergy);
      fillHist1D(prefix + "JetEt",jetEt);
      fillHist1D(prefix + "JetPt",jetPt);
      fillHist1D(prefix + "JetEmf",emEnergyFraction);

      fillHist1D(prefix + "JetEta",jetEta);
      fillHist1D(prefix + "JetPhi",jetPhi);

      if (fabs(jetEta) < etaBarrel()){
	fillHist1D(prefix + "EtJetBarrel",jetEt);
	fillHist1D(prefix + "PtJetBarrel",jetPt);
	if (jetEt > maxEt) maxEt = jetEt;
	if (jetPt > maxPt) maxPt = jetPt;
      }
  }
  //}
  fillHist1D(prefix + "JetEtMaxBarrel",maxEt);
  fillHist1D(prefix + "JetPtMaxBarrel",maxPt);
}


void JetAnalyzer::bookCaloTowerHists() {

  TString hname;
  TString htitle;

  hname="NumberOfCaloTowers";
  htitle="NumberOfCaloTowers";
  m_HistNames[hname] = new TH1F(hname,htitle,2000,0.0,2000.);

  hname="CaloTowerEt";
  m_HistNames[hname] = new TH1F(hname,"CaloTower E_{T}",1000,0.0,100.);

  hname="CaloTowerEta";
  m_HistNames[hname] = new TH1F(hname,"CaloTower #eta",NETA,CaloTowerEtaBoundries);

  hname="CaloTowerPhi";
  m_HistNames[hname] = new TH1F(hname,"CaloTower #eta",72,-M_PI,+M_PI);


  hname="CaloTowerEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower Energy",1000,0.0,100.);


  hname="CaloTowerEmEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower EmEnergy",1000,0.0,100.);

  hname="CaloTowerHadEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower HadEnergy",1000,0.0,100.);


  hname="CaloTowerEmHadEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower EmHadEnergy",1000,0.0,100.);


  hname="CaloTowerOuterEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower Outer Energy",1000,0.0,100.);

  hname="CaloTowerNRecHits";
  m_HistNames[hname] = new TH1F(hname,"CaloTower NumberOfRecHits",100,0.0,100.);

  hname="CaloTowerEnergyEtaPhi";
  m_HistNames2D[hname] = new TH2F(hname,"CaloTower Energy",110,-5.5,5.5,72,-M_PI,+M_PI);

  hname="HOvsHBEnergy";
  m_HistNames2D[hname] = new TH2F(hname,"HOvsHBEnergy",1000,0.0,1000,50,0.0,50);

  //


  hname="CaloTowerSumEt";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEt",1000,0.0,1000.);


  hname="CaloTowerSumEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEnergy",1000,0.0,1000.);


  hname="CaloTowerSumEmEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEmEnergy",1000,0.0,1000.);

  hname="CaloTowerSumHadEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumHadEnergy",1000,0.0,1000.);

  hname="CaloTowerSumEmHadEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEmHadEnergy",1000,0.0,1000.);


  hname="CaloTowerSumHOEnergy";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumHOEnergy",1000,0.0,1000.);

  //


  hname="CaloTowerSumEtMIP";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEt",1000,0.0,1000.);


  hname="CaloTowerSumEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEnergy",1000,0.0,1000.);


  hname="CaloTowerSumEmEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEmEnergy",1000,0.0,1000.);

  hname="CaloTowerSumHadEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumHadEnergy",1000,0.0,1000.);

  hname="CaloTowerSumEmHadEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumEmHadEnergy",1000,0.0,1000.);


  hname="CaloTowerSumHOEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"CaloTower SumHOEnergy",1000,0.0,1000.);


  //----------------------rechits


  hname="RecHitsSumEt";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEt",1000,0.0,1000.);


  hname="RecHitsSumEnergy";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEnergy",1000,0.0,1000.);


  hname="RecHitsSumEmEnergy";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEmEnergy",1000,0.0,1000.);

  hname="RecHitsSumHadEnergy";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumHadEnergy",1000,0.0,1000.);

  hname="RecHitsSumEmHadEnergy";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEmHadEnergy",1000,0.0,1000.);


  hname="RecHitsSumHOEnergy";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumHOEnergy",1000,0.0,1000.);

  //


  hname="RecHitsSumEtMIP";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEt",1000,0.0,1000.);


  hname="RecHitsSumEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEnergy",1000,0.0,1000.);


  hname="RecHitsSumEmEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEmEnergy",1000,0.0,1000.);

  hname="RecHitsSumHadEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumHadEnergy",1000,0.0,1000.);

  hname="RecHitsSumEmHadEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumEmHadEnergy",1000,0.0,1000.);


  hname="RecHitsSumHOEnergyMIP";
  m_HistNames[hname] = new TH1F(hname,"RecHits SumHOEnergy",1000,0.0,1000.);


}


void JetAnalyzer::fillCaloTowerHists(const CaloTowerCollection& caloTowers){


  TString hname="NumberOfCaloTowers";
  int ntowers=caloTowers.size();
  fillHist1D(hname,ntowers);

  double sumEt(0.0);
  double sumTotEnergy(0.0);
  double sumEmEnergy(0.0);
  double sumHadEnergy(0.0);
  double sumHOEnergy(0.0);

  double sumEmHadEnergy(0.0);

  for ( CaloTowerCollection::const_iterator tower=caloTowers.begin(); 
	tower!=caloTowers.end(); tower++) {

    double et=tower->et();
    double eta=tower->eta();
    double phi=tower->phi();

    double  totEnergy= tower->energy();

    double  emEnergy= tower->emEnergy();
    double  hadEnergy= tower->hadEnergy();
    double  outerEnergy= tower->outerEnergy();

    double  emhadEnergy=  emEnergy+hadEnergy;

    sumEt +=et;

    sumTotEnergy +=totEnergy;
    sumEmEnergy +=emEnergy;
    sumHadEnergy +=hadEnergy;
    sumEmHadEnergy +=emhadEnergy;
    sumHOEnergy +=outerEnergy;

    int  numRecHits= tower->constituentsSize();

    fillHist1D("CaloTowerEt",et);
    fillHist1D("CaloTowerEta",eta);
    fillHist1D("CaloTowerPhi",phi);
    fillHist1D("CaloTowerEnergy",totEnergy);
    fillHist1D("CaloTowerEmEnergy",emEnergy);
    fillHist1D("CaloTowerHadEnergy",hadEnergy);
    fillHist1D("CaloTowerEmHadEnergy",emhadEnergy);

    fillHist1D("CaloTowerOuterEnergy",outerEnergy);
    fillHist1D("CaloTowerNRecHits",numRecHits);

    fillHist2D("CaloTowerEnergyEtaPhi",eta,phi,totEnergy);

    if(fabs(eta)<1.4){
      fillHist2D("HOvsHBEnergy",hadEnergy,outerEnergy);
    }

  }

  fillHist1D("CaloTowerSumEt",sumEt);
  fillHist1D("CaloTowerSumEnergy",sumTotEnergy);
  fillHist1D("CaloTowerSumEmEnergy",sumEmEnergy);
  fillHist1D("CaloTowerSumHadEnergy",sumHadEnergy);
  fillHist1D("CaloTowerSumEmHadEnergy",sumEmHadEnergy);
  fillHist1D("CaloTowerSumHOEnergy",sumHOEnergy);


  if(sumEmEnergy<0.6) {
    fillHist1D("CaloTowerSumEtMIP",sumEt);
    fillHist1D("CaloTowerSumEnergyMIP",sumTotEnergy);
    fillHist1D("CaloTowerSumEmEnergyMIP",sumEmEnergy);
    fillHist1D("CaloTowerSumHadEnergyMIP",sumHadEnergy);
    fillHist1D("CaloTowerSumEmHadEnergyMIP",sumEmHadEnergy);
    fillHist1D("CaloTowerSumHOEnergyMIP",sumHOEnergy);
  }
}

void JetAnalyzer::bookRecHitHists(const TString subDetName) {

  TString hname;
  TString htitle;

  hname="NumberOfRecHits"+subDetName;
  htitle="NumberOfRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,10000,0.0,10000.);

  hname="NumberOfRecHitsNZ"+subDetName;
  htitle="NumberOfRecHitsNZ"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,10000,0.0,10000.);

  hname="SumEnergyRecHitsA"+subDetName;
  htitle="SumEnergyRecHitsA"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,100.);

  hname="SumPtRecHits"+subDetName;
  htitle="SumPtRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,1000.);

  hname="SumEnergyRecHits"+subDetName;
  htitle="SumEnergyRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,1000.);

  hname="SumPtRecHitsA"+subDetName;
  htitle="SumPtRecHitsA"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,1000.);


  hname="PtRecHits"+subDetName;
  htitle="PtRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,1000.);

  hname="PtRecHitsA"+subDetName;
  htitle="PtRecHitsA"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,100.);


  hname="EtaRecHits"+subDetName;
  htitle="EatRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,NETA,CaloTowerEtaBoundries);

  hname="EtaPtWeightedRecHits"+subDetName;
  htitle="EtaPtWeightedRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,NETA,CaloTowerEtaBoundries);

  hname="PhiRecHits"+subDetName;
  htitle="PhiRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,72,-M_PI,M_PI);

  hname="EnergyRecHits"+subDetName;
  htitle="EnergyRecHits"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,1000.);

  hname="EnergyRecHitsA"+subDetName;
  htitle="EnergyRecHitsA"+subDetName;
  m_HistNames[hname] = new TH1F(hname,htitle,1000,0.0,100.);

  hname="RecHitsEnergyEtaPhi"+subDetName;
  htitle="RecHitsEnergyEtaPhi"+subDetName;
  m_HistNames2D[hname] = new TH2F(hname,htitle,110,-5.5,5.5,72,-M_PI,+M_PI);

}

void JetAnalyzer::MakeCellListFromCaloTowers(const CaloGeometry& caloGeometry,
					     const CaloTowerCollection& caloTowers,
					     const EBRecHitCollection& EBRecHits,
					     const EERecHitCollection& EERecHits,
					     const HBHERecHitCollection& HBHERecHits,
					     const HORecHitCollection& HORecHits,
					     const HFRecHitCollection& HFRecHits,
					     std::vector<CalCell>& EmCellList,
					     std::vector<CalCell>& HdCellList){

  for ( CaloTowerCollection::const_iterator tower=caloTowers.begin();tower!=caloTowers.end(); tower++) {

//    Double_t et=tower->et();
//     Double_t eta=tower->eta();
//     Double_t phi=tower->phi();

//     Double_t  totEnergy= tower->energy();
//     Double_t  emEnergy= tower->emEnergy();
//     Double_t  hadEnergy= tower->hadEnergy();
//     Double_t  outerEnergy= tower->outerEnergy();

    size_t  numRecHits= tower->constituentsSize();

    for(size_t j = 0; j <numRecHits ; j++){
      Double_t RecHitEnergy(0);
      int  RecHitEta;
      int  RecHitPhi;
      int  RecHitDepth;

      int RecHitix;
      int RecHitiy;

      DetId RecHitDetID=tower->constituent(j);
      const CaloCellGeometry *this_cell = caloGeometry.getGeometry(RecHitDetID);
      GlobalPoint position = this_cell->getPosition();
      double theta=position.theta();
      double phi=position.phi();

      DetId::Detector DetNum=RecHitDetID.det();
      if(DetNum == DetId::Hcal ){
	HcalDetId HcalID = RecHitDetID;
	RecHitEta=  HcalID.ieta();
	RecHitPhi=  HcalID.iphi();      
	RecHitDepth =HcalID.depth();

	HcalSubdetector HcalNum = HcalID.subdet();
	if(HcalNum == HcalBarrel ){
	  HBHERecHitCollection::const_iterator theRecHit=HBHERecHits.find(HcalID);	    
	  RecHitEnergy = theRecHit->energy();
	}
	else if(HcalNum == HcalEndcap){
	  HBHERecHitCollection::const_iterator theRecHit=HBHERecHits.find(HcalID);
          RecHitEnergy= theRecHit->energy();
	}
	else if(  HcalNum == HcalOuter  ){
	  HORecHitCollection::const_iterator theRecHit=HORecHits.find(HcalID);	    
          RecHitEnergy = theRecHit->energy();
	}
	else if(  HcalNum == HcalForward ){
	  HFRecHitCollection::const_iterator theRecHit=HFRecHits.find(HcalID);	    
          RecHitEnergy = theRecHit->energy();
	}
	double px=RecHitEnergy*sin(theta)*cos(phi);
	double py=RecHitEnergy*sin(theta)*sin(phi);
	double pz=RecHitEnergy*cos(theta);
	double e=RecHitEnergy;

	CLHEP::HepLorentzVector P4(px,py,pz,e);

	CalCell cell;
	cell.pid=RecHitHd;
	cell.used=false;
	cell.Momentum = P4;
	HdCellList.push_back(cell);  
      }
      else if( DetNum == DetId::Ecal ){
	int EcalNum =  RecHitDetID.subdetId();
	if( EcalNum == EcalBarrel ){
	  EBDetId EcalID = RecHitDetID;
	  EBRecHitCollection::const_iterator theRecHit=EBRecHits.find(EcalID);	    
	  RecHitEta  = EcalID.ieta();
	  RecHitPhi  = EcalID.iphi();
	  //	  RecHitSM   = EcalID.ism();
	  RecHitEnergy = theRecHit->energy();
	}
	else if(  EcalNum == EcalEndcap ){
	  EEDetId EcalID = RecHitDetID;
	  EERecHitCollection::const_iterator theRecHit=EERecHits.find(EcalID);	    
	  RecHitix= EcalID.ix();
	  RecHitiy= EcalID.iy();
	  RecHitEnergy = theRecHit->energy();
	}
	double px=RecHitEnergy*sin(theta)*cos(phi);
	double py=RecHitEnergy*sin(theta)*sin(phi);
	double pz=RecHitEnergy*cos(theta);
	double e=RecHitEnergy;

	CLHEP::HepLorentzVector P4(px,py,pz,e);

	CalCell cell;
	cell.pid=RecHitEm;
	cell.used=false;
	cell.Momentum = P4;
	EmCellList.push_back(cell);  
      }
    }
  }
}

void JetAnalyzer::bookMetHists(const TString& prefix) {

  TString hname;
  TString htitle;

  hname=prefix + "num";
  htitle = prefix+" Number of MET objects";
  m_HistNames[hname] = new TH1I(hname,htitle,10,0.,10.);

  hname=prefix+"etMiss";
  htitle=prefix+" Missing Et";
  m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.);

  hname=prefix+"etMissX";
  htitle=prefix+" Missing Et-X";
  m_HistNames[hname] = new TH1F(hname,htitle,2000,-1000.0,1000.);

  hname=prefix+"etMissPhi";
  htitle=prefix+" Phi of Missing Et";
  m_HistNames[hname] = new TH1F(hname,htitle,100,-M_PI,M_PI);


  hname=prefix+"sumEt";
  htitle=prefix+" Sum Et";
  m_HistNames[hname] = new TH1F(hname,htitle,1400,0.0,14000.);

}
template <typename T> void JetAnalyzer::fillMetHists(const T& mets, const TString& prefix) {

  Int_t metnum=mets.size();
  fillHist1D(prefix + "num",metnum);

  typedef typename T::const_iterator iter;
  for ( iter met=mets.begin(); met!=mets.end(); met++) {

    Double_t mEt=met->et();
    
    //   Double_t mEt=met->momentum().Pt();
    Double_t sumEt=met->sumEt();
    Double_t mEtPhi=met->phi();
    Double_t mEtX=met->px();

    //    CLHEP::HepLorentzVector momentum = met->Momentum();
    //   Double_t mEtx= momentum.px();

    fillHist1D(prefix + "etMiss",mEt);
    fillHist1D(prefix + "sumEt",sumEt);
    fillHist1D(prefix + "etMissX",mEtX);
    fillHist1D(prefix + "etMissPhi",mEtPhi);
  }
}



void JetAnalyzer::dummyAnalyze(const CaloGeometry& geom) {

  std::cout << "Inside dummyAnalyse routine" << std::endl;

}
/** Finalization (close files, etc) */
void JetAnalyzer::done() {
  std::cout << "Closing up.\n";
  if (m_file!=0) { // if there was a histogram file...
    m_file->Write(); // write out the histrograms
    delete m_file; // close and delete the file
    m_file=0; // set to zero to clean up
  }
}
void JetAnalyzer::bookCalculateEfficiency(){

  TString hname;
  TString htitle;

  hname="rmin";
  htitle = " Minimum Distance between Calo and Gen Jet";
  m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,10.);

  hname="CalPtoGenPt";
  htitle = " CaloJet Pt/ GenJet Pt";
  m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,5.);

}

void JetAnalyzer::CalculateEfficiency(const GenJetCollection& genJets,const CaloJetCollection& calojets){

  const double GenJetPtCut=10;
  const double GenJetEtaCut=1.0;
  const double RCUT=0.25;
 
  for(GenJetIter i=genJets.begin();i!=genJets.end(); i++) {
    Double_t GenJetPt = i->pt();
    Double_t genJetEta = i->eta();
    if(GenJetPt>GenJetPtCut) {

      if(fabs(genJetEta)<GenJetEtaCut){

	if(calojets.size()>0){
	  double rmin(99);
	  CalJetIter caljet;
	  double  calpt(0);
	  for(CalJetIter j=calojets.begin();j!=calojets.end();j++){
	    double rr=radius(i,j);
	    if(rr<rmin){rmin=rr;caljet=j;calpt=j->pt();}
	  }
	  fillHist1D("rmin",rmin);
	  if(rmin<RCUT){
            double response=calpt/GenJetPt;
	    //	    cout << " radius " << rmin << " GenPt " << GenJetPt << " CalPt " << caljet->pt() <<" rr  " << calpt/GenJetPt <<endl;
	    fillHist1D("CalPtoGenPt",response);
	    //     PtSpectrumInAJet(i,genEvent,response);
	  }
	}
      }
    }
  }
}

void JetAnalyzer::bookDiJetBalance(const TString& prefix) {
  TString hname;
  TString htitle;
 
  hname=prefix + "DiJetBalance";
  htitle = prefix+" (PtTrigger-PtProbe)/PtAve ";
  m_HistNames[hname] = new TH1F(hname,htitle,100,-2.,2.);

}
template <typename T> void JetAnalyzer::DiJetBalance(const T& jets, const TString& prefix) {

  const double TrigEtaCut=1.0;
  const double AvePtCut=10;
  const double DPhiCut=2.7;
  const double PtJet3Cut=8;

  typedef typename T::const_iterator iter;

  if(jets.size()<2) return;

  double PtJet3= (jets.size()>2) ? jets[2].pt() : 0;

  double AvePt=(jets[0].pt()+jets[1].pt())/2.0;

  bool FirstJetIsCentral = (fabs(jets[0].eta())<TrigEtaCut); 
  bool SecondJetIsCentral = (fabs(jets[1].eta())<TrigEtaCut); 


  int trig(-1);
  int probe(-1);

  if( FirstJetIsCentral &&  SecondJetIsCentral) {
    if(jets[0].phi()<jets[1].phi()){ trig=0;probe=1; } // Both central pick trigger randomly, phi is random
    else{trig=1;probe=0;}
  }
  else if(FirstJetIsCentral) {trig=0;probe=1;}
  else if(SecondJetIsCentral){trig=1;probe=0;}
  else {return;}

  double B= (jets[trig].pt()-jets[probe].pt())/AvePt;

  if(AvePt>AvePtCut){
    if(fabs(dPhi(jets[0].phi(),jets[1].phi()))>DPhiCut){
      if(PtJet3<PtJet3Cut){
	fillHist1D(prefix+"DiJetBalance",B);       
      }
    }
  }
}
void JetAnalyzer::bookMCParticles(){

  TString hname;
  TString htitle;
 
  const int imax=1;
  const int jmax=1;


  for(int i=0;i<imax;i++){
    std::ostringstream oi; oi << i;
 
    for(int j=0;j<jmax;++j){
      std::ostringstream oj; oj << j;
 
      istringstream ints(oj.str());

      int k;
      ints>>k;
      cout << k << endl;

      hname="VertexZ"+oi.str()+oi.str();
      m_HistNames[hname] = new TH1F(hname,hname,100,-50.,50.);

      hname="VertexX"+oi.str()+oj.str();
      m_HistNames[hname] = new TH1F(hname,hname,200,-5.,5.);

      hname="VertexY"+oi.str()+oj.str();
      m_HistNames[hname] = new TH1F(hname,hname,200,-5.,5.);

      hname="Pt"+oi.str()+oj.str();
      m_HistNames[hname] = new TH1F(hname,hname,500,0.0,500.);

      hname="Pid"+oi.str()+oj.str();
      m_HistNames[hname] = new TH1F(hname,hname,10000,0.0,10000.);

    }
  }
}

void JetAnalyzer::fillMCParticles(edm::Handle<CandidateCollection> genParticles){


  for (size_t i =0;i< genParticles->size(); i++) {

    const Candidate &p = (*genParticles)[i];

    int Status =  p.status();

    //    int Status =  reco::status(&p);

    bool ParticleIsStable = Status==1;
      
    if(ParticleIsStable){
      math::XYZVector vertex(p.vx(),p.vy(),p.vz());
      math::XYZTLorentzVector momentum=p.p4();
      int id = p.pdgId();  
      //cout << "MC particle id " << id << ", creationVertex " << vertex << " cm, initialMomentum " << momentum << " GeV/c" << endl;   
       fillHist1D("Pid00",id); 
     
       fillHist1D("VertexY00",vertex.y());  
       fillHist1D("VertexZ00",vertex.z());  

       fillHist1D("Pt00",momentum.pt());
    }
  }
}
void JetAnalyzer::fillMCParticlesInsideJet(const HepMC::GenEvent genEvent,const GenJetCollection& genJets){
  

  const double GenJetEtaCut=1.0;

  const double  GenJetPtCut[6]={10.,30.,50.,100.,120.,170.};
  const int imax=5;

  int njet=0;
  for(GenJetIter ijet=genJets.begin();ijet!=genJets.end(); ijet++){

      njet++;
      if(njet>10) return;

      Double_t GenJetPt = ijet->pt();
      Double_t GenJetEta = ijet->eta();
      Double_t GenJetPhi = ijet->phi();
   
      for(int ipt=0;ipt<=imax;ipt++){
	if(GenJetPt>GenJetPtCut[ipt] && GenJetPt<GenJetPtCut[ipt+1]){
	  std::ostringstream pi; pi << ipt;
    
	  if(fabs(GenJetEta)<GenJetEtaCut){
	  
	    fillHist1D("GenJetPt"+pi.str(),GenJetPt);
	    fillHist1D("GenJetPhi"+pi.str(),GenJetPhi);
	    fillHist1D("GenJetEta"+pi.str(),GenJetEta);
         
	  
	    int NumParticle(0);

  	    HepLorentzVector P4Jet(0,0,0,0);


	    std::vector <const GenParticle*> jetconst = ijet->getConstituents() ;
	    int nConstituents= jetconst.size();

	    for (int i = 0; i <nConstituents ; i++){

	      NumParticle++;

	      HepLorentzVector p (jetconst[i]->px(),jetconst[i]->py(),jetconst[i]->pz(),jetconst[i]->energy());

	      P4Jet +=p;
                              
	      Double_t Eta = jetconst[i]->eta();
	      Double_t Phi = jetconst[i]->phi();
	      Double_t Pt  = jetconst[i]->pt();

	      int pdgCode = jetconst[i]->pdgId();

	      fillHist1D("PtOfParticleinJet"+pi.str(),Pt);          

	      double rr=radius(GenJetEta,GenJetPhi,Eta,Phi);

	      fillHist1D("Radius"+pi.str(),rr);
              fillHist1D("JetShape"+pi.str(),rr,Pt);
              fillHist1D("Ptdensity"+pi.str(),rr,Pt/(2*M_PI*rr));              
	    }

	    fillHist1D("P4Jet"+pi.str(),P4Jet.perp());
	    fillHist1D("P4JetoGenJetPt"+pi.str(),P4Jet.perp()/GenJetPt);
	    fillHist1D("NumParticle"+pi.str(),NumParticle);
	    
	  }	
	}      
     }
   }
}

void JetAnalyzer::bookMCParticles(const TString& prefix ){

  TString hname;
  TString htitle;
   
  const double  k[6]={10.,30.,50.,100.,120.,170.};

  const int imax=5;
  for(int i=0;i<=imax;i++){
    
    std::ostringstream pti;pti << i;
    std::ostringstream ki;ki << k[i];
    std::ostringstream si;si << k[i+1];
    
    
      hname="GenJetEta"+pti.str();
      htitle = "GenJetEta"+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,600,-6.0,6.);

      hname="GenJetPhi"+pti.str();
      htitle = "GenJetPhi"+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,72,-M_PI,M_PI);

      hname="GenJetPt"+pti.str();
      htitle = "GenJet Pt "+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.);
      
      hname="P4JetoGenJetPt"+pti.str();
      htitle = "P4Jet / GenJet Pt"+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,100,0.9,1.1);

      
      hname="P4Jet"+pti.str();
      htitle = "P4Jet"+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.);
      
      hname="NumParticle"+pti.str();
      htitle = "NumParticle "+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500);

      hname="PtOfParticleinJet"+pti.str();
      htitle = "PtOfParticleinJet"+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,100.);

      hname="Radius"+pti.str();
      htitle = "radius "+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,300,0.0,3.);
      
      hname="JetShape"+pti.str();
      htitle="radius"+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
      m_HistNames[hname] = new TH1F(hname,htitle,300,0.0,3.);


	hname="Ptdensity"+pti.str();
  	htitle="Ptdensity"+pti.str()+"      "+ki.str()+"GeV-"+si.str()+"GeV";
        m_HistNames[hname] = new TH1F(hname,htitle,300,0.0,3.);
   }
}

void Kperp(CLHEP::Hep2Vector jet1,CLHEP::Hep2Vector jet2,double& kperp,double& kparallel){


  double AvePt=(jet1.mag()+jet2.mag())/2.0;

   // unit vector  bisecting  the angle between two leading jets

  CLHEP::Hep2Vector parallel= (jet1.unit()+jet2.unit()).unit();
  CLHEP::Hep2Vector perp = parallel.orthogonal();

   double jparallel= jet1.dot(parallel)*jet2.dot(parallel);
   double jperp = jet1.dot(perp)*jet2.dot(perp);

   kparallel=jparallel/AvePt;
   kperp=jperp/AvePt;

}

template <typename T> void JetAnalyzer::DarkEnergyPlots(const T& jets, const TString& prefix,const CaloTowerCollection& caloTowers ) {


  std::vector<CaloTowerDetId> UsedTowerList;
  std::vector<CaloTower> TowerUsedInJets;
  std::vector<CaloTower> TowerNotUsedInJets;

  typedef typename T::const_iterator iter;

  double SumPtJet(0);
  double SumPtTowers(0);
  
  double sumJetPx(0);
  double sumJetPy(0);

  for ( iter ijet=jets.begin(); ijet!=jets.end(); ijet++) {

    Double_t jetPt = ijet->pt();
    Double_t jetPhi = ijet->phi();

    if(jetPt>5.0) {
      Double_t jetPx = jetPt*cos(jetPhi);
      Double_t jetPy = jetPt*sin(jetPhi);

      sumJetPx +=jetPx;
      sumJetPy +=jetPy;   

      //      const std::vector<CaloTowerDetId>&  barcodes=ijet->getTowerIndices();
      //   int nConstituents= barcodes.size();
      //  for (int i = 0; i <nConstituents ; i++){
      //	UsedTowerList.push_back(barcodes[i]);
      //  }
      SumPtJet +=jetPt;
    }
  }
  //  cout << " UseTowerListSize  " <<  UsedTowerList.size() << endl;

  int NTowersUsed=UsedTowerList.size();

  int    DarkTowers(0);
  double DarkEt(0);

  double sumTowerAllPx(0);
  double sumTowerAllPy(0);

  double sumTowerDarkPx(0);
  double sumTowerDarkPy(0);


  for (CaloTowerCollection::const_iterator tower=caloTowers.begin(); 
	tower!=caloTowers.end(); tower++) {

    CaloTower  t =*tower;
    
    Double_t  et=tower->et();

    if(et>0) {

      Double_t phi=tower->phi();

      SumPtTowers +=tower->et();

      sumTowerAllPx+=et*cos(phi);
      sumTowerAllPy+=et*sin(phi);


      bool used=false;
      for(int i=0;i<NTowersUsed;i++){
        if(tower->id()==UsedTowerList[i]){
          used=true;
          break;
	}
      }

      if(used){
        TowerUsedInJets.push_back(t);
      }
      else{
        TowerNotUsedInJets.push_back(t);
      }
    }
  }


  int ntowersUsed=    TowerUsedInJets.size();
  int ntowersNotUsed= TowerNotUsedInJets.size();

  for(int i=0;i<ntowersNotUsed;i++){
    Double_t et=TowerNotUsedInJets[i].et();

      //Double_t eta=tower->eta();
    Double_t phi=TowerNotUsedInJets[i].phi();
      //Double_t  totEnergy= tower->energy();
      //Double_t  emEnergy= tower->emEnergy();
      //Double_t  hadEnergy= tower->hadEnergy();
      //Double_t  outerEnergy= tower->outerEnergy();
    DarkEt +=et;
    DarkTowers++;
    sumTowerDarkPx+=et*cos(phi);
    sumTowerDarkPy+=et*sin(phi);
  }



  int nUsed= TowerUsedInJets.size();
  int nNotUsed=  TowerNotUsedInJets.size();

  double SumPtJets(0);
  double SumPtNotJets(0);

  for(int i=0;i<nUsed;i++){
    SumPtJets += TowerUsedInJets[i].et();
  }

  for(int i=0;i<nNotUsed;i++){
    SumPtNotJets += TowerNotUsedInJets[i].et();
  }

  // cout << " SumPtJets " <<  SumPtJets  << " Not Used " <<  SumPtNotJets <<  endl;


  int ntowers =caloTowers.size();

  double MEtAllTowers =sqrt(sumTowerAllPx*sumTowerAllPx+sumTowerAllPy*sumTowerAllPy);
  double MEtJets  =sqrt(sumJetPx*sumJetPx+sumJetPy*sumJetPy);

  double MEtDarkTowers =sqrt(sumTowerDarkPx*sumTowerDarkPx+sumTowerDarkPy*sumTowerDarkPy);

  fillHist1D("METTowersAll0",MEtAllTowers);
  fillHist1D("METTowersDark0",MEtDarkTowers);
  fillHist1D("METJets0",MEtJets);

  //  std::cout << " Total Towers " << ntowers <<" JetTowers " << UsedTowerList.size() << " Dark Towers " << DarkTowers <<
  //    " SumPtTowers " << SumPtTowers <<" SumPtJets "<< SumPtJet << " DarkEt " <<  DarkEt << endl;

//   std::cout << " Tower Px     " << sumTowerAllPx << " Tower Py     " << sumTowerAllPy <<  " MEt Tower     " << MEtAllTowers <<endl;
//   std::cout << " Jet Px       " << sumJetPx <<      " Jet Py       " << sumJetPy <<       " MEt Jet       " << MEtJets <<endl;
//   std::cout << " DarkTower Px " << sumTowerDarkPx <<" DarkTower Py " << sumTowerDarkPy << " MEt DarkTower " << MEtDarkTowers <<endl;

}


void JetAnalyzer::bookDarkMetPlots(const TString& prefix ){

  TString hname;
  TString htitle;
   
  //  const double  k[6]={10.,30.,50.,100.,120.,170.};

  const int imax=5;
  for(int i=0;i<=imax;i++){
    
    std::ostringstream pti;pti << i;
        
    hname="METTowersAll"+pti.str();
    htitle = "METTowersAll"+pti.str();
    m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.);

    hname="METTowersDark"+pti.str();
    htitle = "METTowersDark"+pti.str();
    m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.);

    hname="METJets"+pti.str();
    htitle = "METJets"+pti.str();
    m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.);


  }
}

void JetAnalyzer::MakeHadCellList(const CaloTowerCollection& caloTowers,std::vector<CalCell>& CellList){

  for(CaloTowerCollection::const_iterator tower=caloTowers.begin(); 
      tower!=caloTowers.end(); tower++) {

    //    Double_t em=tower->emEnergy();
    Double_t had=tower->hadEnergy();

    Double_t  px=tower->momentum().x();
    Double_t  py=tower->momentum().y();
    Double_t  pz=tower->momentum().z();
    Double_t  E=tower->energy();

    px *=had/E;
    py *=had/E;
    pz *=had/E;
    E  *=had/E;
    
    CLHEP::HepLorentzVector P4(px,py,pz,E);

    CalCell cell;
    cell.pid=CaloTowerHd;
    cell.used=false;
    cell.Momentum = P4;
    CellList.push_back(cell);  
  }
}
void JetAnalyzer::MakeEmCellList(const CaloTowerCollection& caloTowers,std::vector<CalCell>& CellList){

  for(CaloTowerCollection::const_iterator tower=caloTowers.begin(); 
      tower!=caloTowers.end(); tower++) {

    //    Double_t em=tower->emEnergy();
    Double_t had=tower->hadEnergy();

    Double_t  px=tower->momentum().x();
    Double_t  py=tower->momentum().y();
    Double_t  pz=tower->momentum().z();
    Double_t  E=tower->energy();

    px *=had/E;
    py *=had/E;
    pz *=had/E;
    E  *=had/E;
    
    CLHEP::HepLorentzVector P4(px,py,pz,E);

    CalCell cell;
    cell.pid=0;
    cell.used=false;
    cell.Momentum = P4;
    CellList.push_back(cell);  
  }
}

void JetAnalyzer::MakeEmCellList(const CaloGeometry& caloGeometry,
				 const EBRecHitCollection& EBRecHits,
				 const EERecHitCollection& EERecHits,
				 std::vector<CalCell>& CellList){

  CellList.clear();

  for(EBRecHitCollection::const_iterator theRecHit=EBRecHits.begin();theRecHit!=EBRecHits.end();theRecHit++){



    double RecHitEnergy = theRecHit->energy();
    if(RecHitEnergy>0.0){
       EBDetId RecHitDetID = theRecHit->detid();

    //    int RecHitEta  = RecHitDetID.ieta();
    //    int RecHitPhi  = RecHitDetID.iphi();
      const CaloCellGeometry *this_cell = caloGeometry.getGeometry(RecHitDetID);
      GlobalPoint position = this_cell->getPosition();

      double theta=position.theta();
      double phi=position.phi();
      double px=RecHitEnergy*sin(theta)*cos(phi);
      double py=RecHitEnergy*sin(theta)*sin(phi);
      double pz=RecHitEnergy*cos(theta);
      double e=RecHitEnergy;

      CLHEP::HepLorentzVector P4(px,py,pz,e);

      CalCell cell;
      cell.pid=1;
      cell.used=false;
      cell.Momentum = P4;
      CellList.push_back(cell);

      //      cout << " eta " << position.eta() << " phi " << position.phi() << " theta "<< position.theta() <<endl;
      // cout << " RecHit energy " << RecHitEnergy << " px " << px << " py " << py << " pz " << pz << " m " <<P4.m() <<endl;

    }  
  }


  for(EERecHitCollection::const_iterator theRecHit=EERecHits.begin();theRecHit!=EERecHits.end();theRecHit++){

    double RecHitEnergy = theRecHit->energy();
    if(RecHitEnergy>0.0){
      EEDetId RecHitDetID = theRecHit->detid();

    //    int RecHitEta  = RecHitDetID.ieta();
    //    int RecHitPhi  = RecHitDetID.iphi();
      const CaloCellGeometry *this_cell = caloGeometry.getGeometry(RecHitDetID);
      GlobalPoint position = this_cell->getPosition();

      double theta=position.theta();
      double phi=position.phi();
      double px=RecHitEnergy*sin(theta)*cos(phi);
      double py=RecHitEnergy*sin(theta)*sin(phi);
      double pz=RecHitEnergy*cos(theta);
      double e=RecHitEnergy;

      CLHEP::HepLorentzVector P4(px,py,pz,e);

      CalCell cell;
      cell.pid=0;
      cell.used=false;
      cell.Momentum = P4;
      CellList.push_back(cell);

      //      cout << " eta " << position.eta() << " phi " << position.phi() << " theta "<< position.theta() <<endl;
      //  cout << " RecHit energy " << RecHitEnergy << " px " << px << " py " << py << " pz " << pz << " m " <<P4.m() <<endl;

    }  
  }
}

void JetAnalyzer::MakeHadCellList(const CaloGeometry& caloGeometry,
				 const HBHERecHitCollection& HBHERecHits,
				 const HORecHitCollection& HORecHits,
				 const HFRecHitCollection& HFRecHits,
				  std::vector<CalCell>& CellList){

  CellList.clear();

  for(HBHERecHitCollection::const_iterator theRecHit=HBHERecHits.begin();theRecHit!=HBHERecHits.end();theRecHit++){

    double RecHitEnergy = theRecHit->energy();
    if(RecHitEnergy>0.0){
      HcalDetId RecHitDetID = theRecHit->detid();
      const CaloCellGeometry *this_cell = caloGeometry.getGeometry(RecHitDetID);
      GlobalPoint position = this_cell->getPosition();
      CLHEP::HepLorentzVector P4;
      Convert2HepLorentzVector(position,RecHitEnergy,P4);

      CalCell cell; cell.pid=2; cell.used=false; cell.Momentum = P4;
      CellList.push_back(cell);

      //      cout << " eta " << position.eta() << " phi " << position.phi() << " theta "<< position.theta() <<endl;
      //  cout << " RecHit energy " << RecHitEnergy;
      //  cout  << " px() " << P4.px() << " py " << P4.py() << " pz " << P4.pz()  << " m " << P4.m() <<endl;
    }
  }


  for(HORecHitCollection::const_iterator theRecHit=HORecHits.begin();theRecHit!=HORecHits.end();theRecHit++){

    double RecHitEnergy = theRecHit->energy();
    if(RecHitEnergy>0.0){
      HcalDetId RecHitDetID = theRecHit->detid();
      const CaloCellGeometry *this_cell = caloGeometry.getGeometry(RecHitDetID);
      GlobalPoint position = this_cell->getPosition();
      CLHEP::HepLorentzVector P4;
      Convert2HepLorentzVector(position,RecHitEnergy,P4);

      CalCell cell; cell.pid=2; cell.used=false; cell.Momentum = P4;
      CellList.push_back(cell);

      //      cout << " eta " << position.eta() << " phi " << position.phi() << " theta "<< position.theta() <<endl;
      //  cout << " RecHit energy " << RecHitEnergy;
      // cout  << " px() " << P4.px() << " py " << P4.py() << " pz " << P4.pz()  << " m " << P4.m() <<endl;
    }
  }

  for(HFRecHitCollection::const_iterator theRecHit=HFRecHits.begin();theRecHit!=HFRecHits.end();theRecHit++){

    double RecHitEnergy = theRecHit->energy();
    if(RecHitEnergy>0.0){
      HcalDetId RecHitDetID = theRecHit->detid();
      const CaloCellGeometry *this_cell = caloGeometry.getGeometry(RecHitDetID);
      GlobalPoint position = this_cell->getPosition();
      CLHEP::HepLorentzVector P4;
      Convert2HepLorentzVector(position,RecHitEnergy,P4);

      CalCell cell; cell.pid=2; cell.used=false; cell.Momentum = P4;
      CellList.push_back(cell);

      //      cout << " eta " << position.eta() << " phi " << position.phi() << " theta "<< position.theta() <<endl;
      //  cout << " RecHit energy " << RecHitEnergy;
      //cout  << " px() " << P4.px() << " py " << P4.py() << " pz " << P4.pz()  << " m " << P4.m() <<endl;
    }
  }
}

void JetAnalyzer::GetGenPhoton(math::XYZTLorentzVector& momentum){

  for (size_t i =0;i< genParticles->size(); i++) {
    const Candidate &p = (*genParticles)[i];
    int Status =  p.status();
    //   int Status =  reco::status(&p);

    int id = p.pdgId();
    if(Status ==3 && id==22){
      momentum = p.p4();
      break;
    }
  }
}


void JetAnalyzer::GetIntegratedEnergy(GenJetCollection::const_iterator ijet,int nbin,const HepMC::GenEvent genEvent,std::vector<double>& Bins,std::vector<double>& e,std::vector<double>& pt){

  e.clear();
  pt.clear();

  double sumPt(0);
  double sumE(0);

//   double etaJet =ijet->eta();
//   double phiJet =ijet->phi();

    std::vector <const GenParticle*> jetconst =  ijet->getConstituents() ;
    int nConstituents= jetconst.size();
    for (int i = 0; i <nConstituents ; i++){

    Double_t Pt  = jetconst[i]->pt();
    double E= jetconst[i]->energy();


    if(E<Bins[nbin-1]){
      sumE+=E;
      for(int ib=0;ib<nbin-1;ib++){
	if(E>Bins[ib] && E<=Bins[ib+1]){for(int j=ib;j<nbin;j++){ e[j]+=E;} continue;}
      }
    }

    if(Pt<Bins[nbin-1]){
      sumPt+=Pt;
      for(int ib=0;ib<nbin-1;ib++){
	if(Pt>Bins[ib] && Pt<=Bins[ib+1]){for(int j=ib;j<nbin;j++){pt[j]+=Pt;}continue;}
      }
    }
  }
  for(int ib=0;ib<nbin-1;ib++){
    e[ib]/=sumE;
    pt[ib]/=sumPt;
    //  cout << " ib " <<ib << " Pt fraction " << pt[ib] << endl;
  }
}
void JetAnalyzer::Convert2HepLorentzVector(GlobalPoint position,double energy,CLHEP::HepLorentzVector& P4){

  double theta=position.theta();
  double phi=position.phi();;
  P4.setPx(energy*sin(theta)*cos(phi));
  P4.setPy(energy*sin(theta)*sin(phi));
  P4.setPz(energy*cos(theta));
  P4.setE(energy);
}
void JetAnalyzer::SimpleConeCluster(const int type,const double SEEDCUT, const double TOWERCUT,const double RADIUS,std::vector<CalCell> CellList,std::vector<CalCluster>& ClusterList){


  std::sort(CellList.begin(),CellList.end(),CellEtGreater());

  int ncells= CellList.size();
 
  ClusterList.clear();

  if(ncells<=0) return;

  for(int i=0;i<ncells;i++){
    if(!CellList[i].used) {
      if(CellList[i].Momentum.perp()>SEEDCUT){
	CalCluster cluster;

	cluster.Momentum =CellList[i].Momentum;
	cluster.ncells=1;
        cluster.type=type;
        if(CellList[i].pid==1) cluster.em=CellList[i].Momentum.e();
        if(CellList[i].pid==2) cluster.hd=CellList[i].Momentum.e();
	cluster.clusterCellList.push_back(CellList[i]);
	CellList[i].used=true;
	for(int j=1;j<ncells;j++){
	  if(!CellList[j].used) {
	    if(CellList[j].Momentum.perp()>TOWERCUT){
	      double RR=radius(CellList[i],CellList[j]);
	      if(RR<RADIUS){
	        cluster.Momentum+=CellList[j].Momentum;
		if(CellList[j].pid==1) cluster.em=CellList[j].Momentum.e();
		if(CellList[j].pid==2) cluster.hd=CellList[j].Momentum.e();
	        CellList[j].used=true;
	        cluster.ncells++;
	        cluster.clusterCellList.push_back(CellList[j]);
	      }
	    }
	  }
	}
	if(cluster.Momentum.e()>0.0){
	  ClusterList.push_back(cluster);
	}
      }
    }
  }
  std::sort(ClusterList.begin(),ClusterList.end(),ClusterPtGreater());
  //  int nClusters=ClusterList.size();
  //  cout << " Number of Clusters " << nClusters << endl;
  // for(int i=0;i<nClusters;i++){
  //  cout <<i <<" " << ClusterList[i].ncells << " eta " << ClusterList[i].Momentum.eta() << " phi " << ClusterList[i].Momentum.phi() << " Pt " << ClusterList[i].Momentum.perp() << endl;
  // }
  // cout << endl;
}
void JetAnalyzer::MakeCaloTowerList(const CaloGeometry& caloGeometry,const CaloTowerCollection& caloTowers,std::vector<CalCell>& CellList){


  for ( CaloTowerCollection::const_iterator tower=caloTowers.begin(); 
	tower!=caloTowers.end(); tower++) {

//     Double_t et=tower->et();
//     Double_t eta=tower->eta();
//     Double_t phi=tower->phi();

//     Double_t  totEnergy= tower->energy();
//     //    double theta=position.theta();

//     double px=totEnergy*sin(theta)*cos(phi);
//     double py=totEnergy*sin(theta)*sin(phi);
//     double pz=totEnergy*cos(theta);
//     double e=totEnergy;

    CLHEP::HepLorentzVector P4(tower->momentum().x(),tower->momentum().y(),tower->momentum().z(),tower->energy());


    CalCell cell;
    cell.pid=0;
    cell.used=false;
    cell.Momentum = P4;
    CellList.push_back(cell);  

      //      cout << " eta " << position.eta() << " phi " << position.phi() << " theta "<< position.theta() <<endl;
      //      cout << " RecHit energy " << RecHitEnergy << " px " << px << " py " << py << " pz " << pz << " m " <<P4.m() <<endl;

  }
}
void JetAnalyzer::MakeIRConeJets(const double SEEDCUT, const double TOWERCUT,const double RADIUS,std::vector<CalCell> CellList,std::vector<CalCluster>& ClusterList){

  std::sort(CellList.begin(),CellList.end(),CellEtGreater());

  // cout << " Cells " << CellList.size() << endl;
  // cout << " Highest Tower " << CellList[0].energy << " RR " << radius(CellList[0],CellList[1]) << endl;

  int ncells= CellList.size();
 
  if(ncells<=0) return;

  const int ItrMax(3);

  for(int i=0;i<ncells;i++){
    if(!CellList[i].used){
      if(CellList[i].Momentum.perp()>SEEDCUT){

	CalCluster cluster;

	int iteration(0);
        double etaCluster=CellList[i].Momentum.eta();
        double phiCluster=CellList[i].Momentum.phi();

      //      double phithr=phiCluster;

        std::vector<int> TowerUsedList[10];
        bool Changed=true;

	std::vector<double> ECluster;
	for(int  i=0;i<20;i++){
	  ECluster.push_back(0);
	}
      
	double ChangeInE(9999);          

      //      cout << endl<< " New Cluster     " <<endl;

	do{
	//        cout << " iteration " << iteration << " eta " << etaCluster << " phi "<< phiCluster << endl;
          cluster.Momentum.set(0.,0.,0.,0.);

          for(int j=0;j<ncells;j++){
	    if(!CellList[j].used) {
              if(CellList[j].Momentum.perp()>TOWERCUT){
	        double RR=radius(etaCluster,phiCluster,CellList[j].Momentum.eta(),CellList[j].Momentum.phi());
		if(RR<RADIUS){
	          TowerUsedList[iteration].push_back(j);
	          ECluster[iteration]+=CellList[j].Momentum.e();
	          cluster.Momentum   +=CellList[j].Momentum;
	          cluster.ncells++;
		}
	      }
	    }
	  }

	  if(iteration>0){
            ChangeInE=ECluster[iteration]-ECluster[iteration-1];
	    Changed = !VectorsAreEqual(TowerUsedList[iteration],TowerUsedList[iteration-1]);
	  }
	  iteration++;
	} while(iteration<ItrMax &&  fabs(ChangeInE)>0.1 && Changed);

        if(cluster.Momentum.e()>0.0){
          int size= TowerUsedList[iteration-1].size();
          for(int i=0;i<size;i++){
	    int j=TowerUsedList[iteration-1][i];
	    CellList[j].used=true;
	  }
	  ClusterList.push_back(cluster);
	}
      }
    }
  }
  std::sort(ClusterList.begin(),ClusterList.end(),ClusterPtGreater());
}
bool JetAnalyzer::VectorsAreEqual(std::vector<int> VecA,std::vector<int> VecB){

  int sa=VecA.size();
  int sb=VecB.size();

  if(sa!=sb) return false;

  for(int i=0;i<sa;i++){
    if(VecA[i]!=VecB[i]) return false;
  }
  return true;
}
// void  MakePhysicsTowers(const CaloTowerCollection& caloTowers,std::vector<PhysicsTower>& PhysicsTowerList){
//   for(CaloTowerCollection::const_iterator tower=caloTowers.begin(); tower!=caloTowers.end(); tower++) {

//     double px=tower->momentum().x();
//     double py=tower->momentum().y();
//     double pz=tower->momentum().x();
//     double E=tower->energy();

//     if(tower->et()>0.5){
//       PhysicsTowerList.push_back(LorentzVector(px,py,pz,E));
//     }
//   }
// }
double CorrectedE(double E_Em,double E_Hd,int ifit,double& f00,double& EoHEm,double& EoHHd,double& EoPiEm,double& EoPiHd){

  //  const int IFIT=1;

  const double a[2]={0.11,0.11};  //a3
  const double b[2]={1.00,0.087}; //a4

  const double c[2]={1.39,1.390}; //a0
  const double d[2]={0.00,1.385}; //a1

  const double e[2]={2.00,1.26};  //a5
  const double f[2]={0.00,5.64};  //a2 


  double Etot=E_Em+E_Hd;

  //  double f0=a[ifit]*pow(log(Etot),b[ifit]);
  double f0=a[ifit]*log(pow(Etot,b[ifit]));

  f0 = f0>0? f0:0;

  double EoH_Hd= c[ifit]*(1+d[ifit]/Etot);

  double EoH_Em= e[ifit]*(1+f[ifit]/Etot);

  double EoPi_Em = EoH_Em/(1+(EoH_Em-1)*f0);
  double EoPi_Hd = EoH_Hd/(1+(EoH_Hd-1)*f0);
 
  double E= EoPi_Em*E_Em+EoPi_Hd*E_Hd;

  f00=f0;
  EoHHd=EoH_Hd;
  EoHEm=EoH_Em;

  EoPiEm = EoPi_Em;
  EoPiHd = EoPi_Hd;


  return E;

}
double CorrectedE(double E_Em,double E_Hd,int ifit){

  //  const int IFIT=1;

  const double a[2]={0.11,0.11}; //a3
  const double b[2]={1.00,0.87}; //a4

  const double c[2]={1.39,1.390}; //a0
  const double d[2]={0.00,1.385}; //a1

  const double e[2]={2.00,1.26}; //a5
  const double f[2]={0.00,5.64}; //a2 


  double Etot=E_Em+E_Hd;

  //  double f0=a[ifit]*pow(log(Etot),b[ifit]);
  double f0=a[ifit]*log(pow(Etot,b[ifit]));

  f0 = f0>0? f0:0;

  double EoH_Hd= c[ifit]*(1+d[ifit]/Etot);

  double EoH_Em= e[ifit]*(1+f[ifit]/Etot);

  double EoPi_Em = EoH_Em/(1+(EoH_Em-1)*f0);
  double EoPi_Hd = EoH_Hd/(1+(EoH_Hd-1)*f0);
 
  double E= EoPi_Em*E_Em+EoPi_Hd*E_Hd;

  return E;

}


void JetAnalyzer::MatchEmHadClusters(std::vector<CalCluster> EmClusterList,std::vector<CalCluster> HdClusterList,std::vector<CalCluster>& EmHdClusterList){

  const double RCUT=0.12;

  int nEmClusters=EmClusterList.size();
  int nHdClusters=HdClusterList.size();

  std::vector<bool> EmClusterUsed(nEmClusters);

  // Copy Had clusters

  for(int i=0;i<nHdClusters;i++){
    HdClusterList[i].hd=1.0;
    HdClusterList[i].em=0.0;
    EmHdClusterList.push_back(HdClusterList[i]);
  }

  int nEmHdClusters=EmHdClusterList.size();

  // For each Had cluster find Em Cluster and add info

  for(int i=0;i<nEmHdClusters;i++){
    double HdEnergy= EmHdClusterList[i].Momentum.e();
    double EmEnergy(0.0);
    for(int iem=0;iem<nEmClusters;iem++){
      if(!EmClusterUsed[iem]){
        double rr= radius(EmHdClusterList[i],EmClusterList[iem]);
        if(rr<RCUT){
          EmClusterUsed[iem]=true;
 	  EmHdClusterList[i].SubClusterList.push_back(EmClusterList[iem]);
          MatchParam match;
          match.distance=rr;
          match.index=iem;
	  EmHdClusterList[i].MatchedClusters.push_back(match);     
	  EmHdClusterList[i].type=ClusterEmHd;
          EmEnergy= EmClusterList[i].Momentum.e();          
	}
      }
    }
    if(EmEnergy>0.0){
      EmHdClusterList[i].hd=HdEnergy/(EmEnergy+HdEnergy);
      EmHdClusterList[i].em=EmEnergy/(EmEnergy+HdEnergy);
    }
  }

  for(int i=0;i<nEmClusters;i++){
    if(!EmClusterUsed[i]){
      EmClusterList[i].hd=0.0;
      EmClusterList[i].em=1.0;
      EmHdClusterList.push_back(EmClusterList[i]);
    }
  }
  //  cout << " Number of combined clusters  " << EmHdClusterList.size() << endl;
}
void JetAnalyzer::bookSubClusterHistograms(){

  bookClusterPlot("CaloTowerClusterR05");
  bookClusterPlot("CaloTowerClusterR03");
  bookClusterPlot("CaloTowerClusterR015");

  bookClusterPlot("EmRecHitClusterR0030");
  bookClusterPlot("EmRecHitClusterR0045");
  bookClusterPlot("EmRecHitClusterR0050");
  bookClusterPlot("EmRecHitClusterR0060");
  bookClusterPlot("EmRecHitClusterR0070");

  bookClusterPlot("HadRecHitClusterR015");
  bookClusterPlot("HadRecHitClusterR025");


  TString hname; TString htitle;
   
  hname="NumberOfEmHdClusters"; htitle="NumberOfEmHdClusters";
  m_HistNames[hname]= new TH1F(hname,htitle,1000,0.0,1000.0);

  hname="NumberOfMatchesPerCluster";
  htitle="NumberOfMatchedPerCluster";
  m_HistNames[hname]= new TH1F(hname,htitle,50,0.0,50.0);

  //================================================================
  hname="dSumEt";
  htitle="SumEtRecMet-SumEtCluster";
  m_HistNames[hname]= new TH1F(hname,htitle,1000,-500.0,500.0);

  hname="dMet";
  htitle="MissEtRecMet-MissEtCluster";
  m_HistNames[hname]= new TH1F(hname,htitle,400,-100.0,100.0);

  hname="Met2D";
  htitle="MissEtRecMetvMissEtCluster";
  m_HistNames2D[hname]= new TH2F(hname,htitle,100,0.0,200.0,100,0.0,200.);

  hname="SumEt2D";
  htitle="SumEtRecMetvSumEtCluster";
  m_HistNames2D[hname]= new TH2F(hname,htitle,200,0.0,2000.0,200,0.0,2000.);

  hname="SumEt2D";
  htitle="SumEtRecMetvSumEtCluster";
  m_HistNames2D[hname]= new TH2F(hname,htitle,200,0.0,2000.0,200,0.0,2000.);


  hname="dSumEtR";
  htitle="SumEtRecHit-SumEtCluster";
  m_HistNames[hname]= new TH1F(hname,htitle,1000,-500.0,500.0);

  hname="dMetR";
  htitle="MissEtRecHit-MissEtCluster";
  m_HistNames[hname]= new TH1F(hname,htitle,400,-200.0,200.0);

  hname="Met2DR";
  htitle="MissEtRecHitvMissEtCluster";
  m_HistNames2D[hname]= new TH2F(hname,htitle,100,0.0,200.0,100,0.0,200.);

  hname="SumEt2DR";
  htitle="SumEtRecHitvSumEtCluster";
  m_HistNames2D[hname]= new TH2F(hname,htitle,100,0.0,2000.0,100,0.0,2000.);

  hname="SumEt2DR2";
  htitle="SumEtRecHitvSumEtCluster2";
  m_HistNames2D[hname]= new TH2F(hname,htitle,100,0.0,500.0,100,0.0,500.);

}

void JetAnalyzer::bookClusterPlot(TString name){

  TString hname; TString htitle;
  hname="NumberOf"+name;
  htitle="NumberOf"+name;
  m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,100.0);

  hname="SumEnergy"+name+"Clusters";
  htitle=hname;
  m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.0);

  hname="NCells"+name+"Clusters";
  htitle=hname;
  m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,100.0);
  
  hname="Energy"+name+"Clusters";
  htitle=hname;
  m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.0);

  for(int i=0;i<5;i++){
    std::ostringstream oi; oi << i;

    hname="NCells"+name+"Cluster"+oi.str();
    htitle=hname;
    m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,100.0);
  
    hname="Energy"+name+"Cluster"+oi.str();
    htitle=hname;
    m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.0);

    if(i==0) {
      hname="NCells"+name+"ClusterMIP"+oi.str();
      htitle=hname;
      m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,100.0);
  
      hname="Energy"+name+"ClusterMIP"+oi.str();
      htitle=hname;
      m_HistNames[hname] = new TH1F(hname,htitle,500,0.0,500.0);
    }
  }
}

void JetAnalyzer::fillClusterPlot(TString name, std::vector<CalCluster> ClusterList){

  int nClusters=ClusterList.size();

  TString hname;

  hname="NumberOf"+name;
  fillHist1D(hname,nClusters);

  double sumEnergy(0);

  for(int i=0;i<nClusters;i++){
    sumEnergy +=ClusterList[i].Momentum.e();
    hname="NCells"+name+"Clusters";
    fillHist1D(hname,ClusterList[i].ncells);
    hname="Energy"+name+"Clusters";
    fillHist1D(hname,ClusterList[i].Momentum.e());
  }

  for(int i=0;i<TMath::Min(5,nClusters);i++){
    std::ostringstream oi; oi << i;
    hname="NCells"+name+"Cluster"+oi.str();
    fillHist1D(hname,ClusterList[i].ncells);
    hname="Energy"+name+"Cluster"+oi.str();
    fillHist1D(hname,ClusterList[i].Momentum.e());
   
    if(i==0 && IsItMIP_) {
      hname="NCells"+name+"ClusterMIP"+oi.str();
      fillHist1D(hname,ClusterList[i].ncells);
      hname="Energy"+name+"ClusterMIP"+oi.str();
      fillHist1D(hname,ClusterList[i].Momentum.e());
    }

  }

  hname="SumEnergy"+name+"Clusters";
  fillHist1D(hname,sumEnergy);

}

void JetAnalyzer::fillSubClusterPlot(std::vector<CalCluster> CaloClusterR05List,
				     std::vector<CalCluster> CaloClusterR03List,
				     std::vector<CalCluster> CaloClusterR015List,
				     std::vector<CalCluster> HdRHClusterR015List,
				     std::vector<CalCluster> HdRHClusterR025List,
				     std::vector<CalCluster> EmRHClusterR003List,
				     std::vector<CalCluster> EmRHClusterR006List,
				     std::vector<CalCluster> EmHdClusterList){

  TString hname="NumberOfCaloTowerR05Clusters";
  fillHist1D(hname,CaloClusterR05List.size());

  if(CaloClusterR05List.size()>0.0){
    hname="NCellCaloTowerR05Cluster0";
    fillHist1D(hname,CaloClusterR05List[0].ncells);
    hname="EnergyCaloTowerR05Cluster0";
    fillHist1D(hname,CaloClusterR05List[0].Momentum.e());
  }


  hname="NumberOfCaloTowerR03Clusters";
  fillHist1D(hname,CaloClusterR03List.size());

  hname="NumberOfCaloTowerR15Clusters";
  fillHist1D(hname,CaloClusterR015List.size());

  hname="NumberOfEmRecHitR003Clusters";
  fillHist1D(hname,EmRHClusterR003List.size());

  hname="NumberOfEmRecHitR006Clusters";
  fillHist1D(hname,EmRHClusterR006List.size());

  hname="NumberOfHdRecHitR15Clusters";
  fillHist1D(hname,HdRHClusterR015List.size());


  hname="NumberOfHdRecHitR25Clusters";
  fillHist1D(hname,HdRHClusterR025List.size());

  hname="NumberOfEmHdClusters";
  fillHist1D(hname,EmHdClusterList.size());





  int EmHdClusters(0);
  int EmOnlyClusters(0);
  int HdOnlyClusters(0);
  
  double SumPtEvent(0.0);
  CLHEP::HepLorentzVector P4Event(0.,0.,0.,0.);

  int nEmHd=EmHdClusterList.size();
  for(int i=0;i<nEmHd;i++){
     int type=EmHdClusterList[i].type;
    //  int ncells=EmHdClusterList[i].ncells;
    int nmatch=EmHdClusterList[i].SubClusterList.size();
    //   double eta=EmHdClusterList[i].Momentum.eta();
    //   double phi=EmHdClusterList[i].Momentum.phi();
    double pt=EmHdClusterList[i].Momentum.perp();

    double SumPtCluster=pt;
    CLHEP::HepLorentzVector P4Cluster =EmHdClusterList[i].Momentum;

    if(type== ClusterEm) EmOnlyClusters++;
    else if(type==ClusterHd) HdOnlyClusters++;
    else if(type==ClusterEmHd) EmHdClusters++;
    else {cout << " Cluster type  not defined " <<endl;}

    //    cout << " Cluster type "<<type <<" ncells " << ncells << " eta " << eta << " phi " << phi << " pt " << pt << " Matches " << nmatch<< endl; 
    CLHEP::HepLorentzVector SumP4Match(0.,0.,0.,0.);
  
    for(int j=0;j<nmatch;j++){
      SumP4Match+=EmHdClusterList[i].SubClusterList[j].Momentum;
      //   int    ncells=EmHdClusterList[i].SubClusterList[j].ncells;
      //   double eta=EmHdClusterList[i].SubClusterList[j].Momentum.eta();
      //   double phi=EmHdClusterList[i].SubClusterList[j].Momentum.phi();
      //  double pt=EmHdClusterList[i].SubClusterList[j].Momentum.perp();
      //      cout << " SubCluster j=" << j <<" ncells " << ncells << " eta " << eta << " phi " << phi << " pt " << pt <<endl;
    }
 
    P4Cluster +=SumP4Match;
    SumPtCluster+=SumP4Match.perp();

    P4Event +=P4Cluster;
    SumPtEvent+=SumPtCluster;
    
    hname="NumberOfMatchesPerCluster";
    fillHist1D(hname,nmatch);
  
    //   if (nmatch>0) cout << " Number of matched clusters " << nmatch << " Pt " << SumP4Match.perp() << " eta " <<   SumP4Match.perp() << " phi " << SumP4Match.phi()<<endl;    
  }
  //  cout << " SumPt Evenet " << SumPtEvent << " Missing Et " << P4Event.perp() <<endl;
}

void JetAnalyzer::CalculateSumEtMET(std::vector<CalCluster> EmHdClusterList,double& SumEt,double& MET){
  
  double SumPtEvent(0.0);
  CLHEP::HepLorentzVector P4Event(0.,0.,0.,0.);

  int nEmHd=EmHdClusterList.size();
  for(int i=0;i<nEmHd;i++){
    //    int type=EmHdClusterList[i].type;
    int nmatch=EmHdClusterList[i].SubClusterList.size();
    double SumPtCluster =EmHdClusterList[i].Momentum.perp();
    CLHEP::HepLorentzVector P4Cluster =EmHdClusterList[i].Momentum;

    CLHEP::HepLorentzVector SumP4Match(0.,0.,0.,0.);
 
    for(int j=0;j<nmatch;j++){
      SumP4Match+=EmHdClusterList[i].SubClusterList[j].Momentum;
    }
 
    P4Cluster +=SumP4Match;
    SumPtCluster+=SumP4Match.perp();

    P4Event +=P4Cluster;
    SumPtEvent+=SumPtCluster;
  }
  //  cout << " SumPt " << SumPtEvent << " Missing Et " << P4Event.perp() <<endl;
  SumEt= SumPtEvent;
  MET= P4Event.perp();
}
void JetAnalyzer::bookPtSpectrumInAJet(){

  TString hname;
  TString htitle;


  for(int i=0;i<6;i++){
    std::ostringstream oi; oi << i;

    hname="SumEtAllParticles"+oi.str();
    htitle="SumEtAllParticles"+oi.str();
    m_HistNames[hname]= new TH1F(hname,htitle,1000,0.0,10000.);

    hname="MissEtAllParticles"+oi.str();
    htitle="MissEtAllParticles"+oi.str();
    m_HistNames[hname]= new TH1F(hname,htitle,100,0.0,100.);


    hname="SumEtJetParticles"+oi.str();
    htitle="SumEtJetParticles"+oi.str();
    m_HistNames[hname]= new TH1F(hname,htitle,1000,0.0,10000.);

    hname="MissEtJetParticles"+oi.str();
    htitle="MissEtJetParticles"+oi.str();
    m_HistNames[hname]= new TH1F(hname,htitle,100,0.0,100.);


    hname="SumEtNonJetParticles"+oi.str();
    htitle="SumEtNonJetParticles"+oi.str();
    m_HistNames[hname]= new TH1F(hname,htitle,1000,0.0,10000.);

    hname="MissEtNonJetParticles"+oi.str();
    htitle="MissEtNonJetParticles"+oi.str();
    m_HistNames[hname]= new TH1F(hname,htitle,100,0.0,100.);

  }

  for(int icut=0;icut<6;icut++){
    std::ostringstream oicut; oicut << icut;
    for(int ipt=0;ipt<10;ipt++){
      std::ostringstream oipt; oipt << ipt;
      hname="PtParticlesJetLostPt"+oipt.str()+"T"+oicut.str();
      htitle="PtParticlesJetLotPt"+oipt.str()+"T"+oicut.str();
      m_HistNames[hname]= new TH1F(hname,htitle,500,0.0,250.);
    }
  }
}
void JetAnalyzer::PtSpectrumInAJet(GenJetCollection::const_iterator ijet,const HepMC::GenEvent genEvent,const double response){

  std::vector <const GenParticle*> jetconst =  ijet->getConstituents() ;
  int nConstituents= jetconst.size();
  for (int i = 0; i <nConstituents ; i++){

    Double_t  energy = jetconst[i]->energy();

    TString hname="AllJetsPtSpectrum";
    fillHist1D(hname,energy);

    if(response <0.7) {
      hname="LowResponseJetsPtSpectrum";
      fillHist1D(hname,energy);
    }
    else {
      hname="HighResponseJetsPtSpectrum";
      fillHist1D(hname,energy);
    }
  }
}

void JetAnalyzer::PtSpectrumInSideAJet(const GenJetCollection& genJets,const HepMC::GenEvent genEvent){

  const int nbins=6;

  const double PtCut[nbins]={0.0,0.5,1.0,1.5,2.0,2.5};
  HepMC::GenEvent::particle_const_iterator it;

  std::vector<const GenParticle*> genPartUsedInJets;
  std::vector<const Candidate*> genPartNotUsedInJets;

  double  SumPtJet[10][6];

  int njet(0);
  for (GenJetIter ijet=genJets.begin(); ijet!=genJets.end(); ijet++) {
    Double_t jetPt = ijet->pt();
    if(jetPt>10.){

      std::vector <const GenParticle*> jetconstituents =  ijet->getConstituents() ;
      int nConstituents= jetconstituents.size();
      genPartUsedInJets=jetconstituents;
      
      for (int i = 0; i <nConstituents ; i++){
	
        if(njet<10){
          double pt= jetconstituents[i]->pt();
          for(int icut=0;icut<nbins;icut++){
            if(pt>PtCut[icut]){
	      SumPtJet[njet][icut] +=pt;
	    }
	  }
	}
      }
      njet++;
    }
  }
  

  for(int ijet=0;ijet<10;ijet++){
    for(int icut=0;icut<6;icut++){
      std::ostringstream oicut; oicut << icut;
      int ipt=GetPtBin(SumPtJet[ijet][0]);
      std::ostringstream oipt; oipt << ipt;
      TString hname="PtParticlesJetLostPt"+oipt.str()+"T"+oicut.str();
      fillHist1D(hname,SumPtJet[ijet][icut]);
    }
  }

  int NumPartUsedInJets= genPartUsedInJets.size();


  int NumStableparticles(0);

  for (size_t i =0;i< genParticles->size(); i++) {

    const Candidate &p = (*genParticles)[i];
    int Status =  p.status();
    //  int Status =  reco::status(&p);

    bool ParticleIsStable = Status==1;
    if(ParticleIsStable){

      NumStableparticles++;


      bool ParticleUsed(false);
      for(int i=0;i<NumPartUsedInJets;i++){

      const GenParticle* genpartused =   genPartUsedInJets[i];
      //std::cout <<  "genpartused " << genpartused->px() << " " <<  genpartused->py() << " " << genpartused->energy() << std::endl;

      if( genpartused->px() == p.px()
	  && genpartused->py() == p.py() 
	  && genpartused->pz() == p.pz() 
	  && genpartused->energy() == p.energy()
	  && genpartused->charge() == p.charge()
	  && genpartused->pdgId() ==  p.pdgId() 
	  //	  && reco::status(genpartused) ==   reco::status(&p)

	  ){
	//std::cout << "MATCHED " <<  genpart->px() << " " << genpartused->px() << std::endl;
          ParticleUsed=true; continue;
       }
      }
      if(!ParticleUsed) genPartNotUsedInJets.push_back(&p);

    }
  }



//   cout << " Particles total " <<   NumStableparticles;
//   cout << " in jets " <<  genPartUsedInJets.size();
//   cout << " outside jets  " <<   genPartNotUsedInJets.size() <<endl;



  std::vector<double> SumPtFromAllParticles(nbins);
  std::vector<double> SumPxFromAllParticles(nbins);
  std::vector<double> SumPyFromAllParticles(nbins);
  std::vector<double> GenMetFromAllParticles(nbins);

  for (size_t i =0;i< genParticles->size(); i++) {

    const Candidate &p = (*genParticles)[i];

    int Status =  p.status();
    //int Status =  reco::status(p);

    bool ParticleIsStable = Status==1;
      
    if(ParticleIsStable){
      double  pt =p.pt();
      double  px =p.px();
      double  py =p.py();
      for(int i=0;i<nbins;i++){
	if(pt>PtCut[i]){
	  SumPtFromAllParticles[i] +=pt;
	  SumPxFromAllParticles[i] +=px;
	  SumPyFromAllParticles[i] +=py;
	}
      }
    }
  }

  std::vector<double> SumPtFromJetParticles(nbins);
  std::vector<double> SumPxFromJetParticles(nbins);
  std::vector<double> SumPyFromJetParticles(nbins);
  std::vector<double> GenMetFromJetParticles(nbins);

  for(int  i=0;i<NumPartUsedInJets;i++){
    double pt =genPartUsedInJets[i]->pt();
    double px =genPartUsedInJets[i]->px();
    double py =genPartUsedInJets[i]->py();

    for(int i=0;i<nbins;i++){
      if(pt>PtCut[i]){
        SumPtFromJetParticles[i] +=pt;
        SumPxFromJetParticles[i] +=px;
        SumPyFromJetParticles[i] +=py;
      }
    }
  }

  std::vector<double> SumPtFromNonJetParticles(nbins);
  std::vector<double> SumPxFromNonJetParticles(nbins);
  std::vector<double> SumPyFromNonJetParticles(nbins);
  std::vector<double> GenMetFromNonJetParticles(nbins);

  int NumPartNotUsedInJets= genPartNotUsedInJets.size();
  //  cout << " Num of particles not in jet " << NumPartNotUsedInJets << endl;
  for(int  i=0;i<NumPartNotUsedInJets;i++){
    double pt=genPartNotUsedInJets[i]->pt();
    double px=genPartNotUsedInJets[i]->px();
    double py=genPartNotUsedInJets[i]->py();

    for(int i=0;i<nbins;i++){
      if(pt>PtCut[i]){
        SumPtFromNonJetParticles[i] +=pt;
        SumPxFromNonJetParticles[i] +=px;
        SumPyFromNonJetParticles[i] +=py;
      }
    }
  }


  std::vector<double> GenMetFromJetNonJetParticles(nbins);

  for(int i=0;i<nbins;i++){
    GenMetFromAllParticles[i] =sqrt(pow(SumPxFromAllParticles[i],2)+pow(SumPyFromAllParticles[i],2));
    GenMetFromJetParticles[i] =sqrt(pow(SumPxFromJetParticles[i],2)+pow(SumPyFromJetParticles[i],2));
    GenMetFromNonJetParticles[i] =sqrt(pow(SumPxFromNonJetParticles[i],2)+pow(SumPyFromNonJetParticles[i],2));
    GenMetFromJetNonJetParticles[i] =sqrt(pow((SumPxFromNonJetParticles[i]+SumPxFromJetParticles[i]),2)+
                                          pow((SumPyFromNonJetParticles[i]+SumPyFromJetParticles[i]),2));
  }
  
  for(int i=0;i<nbins;i++){
    std::ostringstream oi; oi << i;
    TString hname="SumEtAllParticles"+oi.str();
    fillHist1D(hname,SumPtFromAllParticles[i]);

    hname="MissEtAllParticles"+oi.str();
    fillHist1D(hname,GenMetFromAllParticles[i]);


    hname="SumEtJetParticles"+oi.str();
    fillHist1D(hname,SumPtFromJetParticles[i]);

    hname="MissEtJetParticles"+oi.str();
    fillHist1D(hname,GenMetFromJetParticles[i]);

    hname="SumEtNonJetParticles"+oi.str();
    fillHist1D(hname,SumPtFromNonJetParticles[i]);

    hname="MissEtNonJetParticles"+oi.str();
    fillHist1D(hname,GenMetFromNonJetParticles[i]);


  }
}

void JetAnalyzer::GetParentPartons(std::vector<Candidate*>& ParentParton){

  cout << " %GetParentPartons -- This function needs to be fixed " << endl;
  ParentParton.clear();
  int np(0);

  for (size_t i =0;i< genParticles->size(); i++) {
    const Candidate &p = (*genParticles)[i];
    int Status =  p.status();
    //   int Status =  reco::status(*p);

    np++;
    if(Status==3) {
      //      cout << " np " << np << "Status " << (*p)->status() <<" Mother 1 " << (*p)->Mother() << "  Mother 2 " << (*p)->SecondMother() << "  Type " << (*p)->pdg_id() << " Pt " << (*p)->Momentum().perp() <<endl; 

      if(p.numberOfMothers()==2){
	const Candidate &m0 = *(p.mother(0));
	const Candidate &m1 = *(p.mother(1));

			      
			      //==5 && *(p.mother(1))==6){

	//        ParentParton.push_back((*genParticles)[i]);
	//if(ParentParton.size()==2) return;
      }
    }
    if(np>10) return;
  }
}
int  JetAnalyzer::GetPtBin(double GenJetPt){
  int NPtBins=11;
  double GenJetPtBins[12]={0.0,10.,15.,20.,25.,30.,40.,50.,75.,100,1000.,10000.};
  for(int ip=0;ip<NPtBins;ip++){
    if(GenJetPtBins[ip] <GenJetPt && GenJetPt < GenJetPtBins[ip+1]){
      return ip;
    }
  }
  return 0;
}
void JetAnalyzer::fillTBTriggerHists(const HcalTBTriggerData& trigger){

  cout << " This  method is not implemented yet. " << endl;

}

void JetAnalyzer::MakeLocalClusters(const CaloGeometry& caloGeometry,
				    const CaloJetCollection& calojets,
				    const CaloMETCollection& recmets,
				    const CaloTowerCollection& caloTowers,
				    const EBRecHitCollection& EBRecHits,
				    const EERecHitCollection& EERecHits,
				    const HBHERecHitCollection& HBHERecHits,
				    const HORecHitCollection& HORecHits,
				    const HFRecHitCollection& HFRecHits){
  TString hname; TString htitle;

    std::vector<CalCluster> CaloClusterR05List;
    std::vector<CalCluster> CaloClusterR03List;
    std::vector<CalCluster> CaloClusterR15List;

    if(&caloTowers){
      std::vector<CalCell> CaloTowerList;
      MakeCaloTowerList(caloGeometry,caloTowers,CaloTowerList);
      SimpleConeCluster(ClusterTower,0.5,0.5,0.5,CaloTowerList,CaloClusterR05List);
      SimpleConeCluster(ClusterTower,0.5,0.5,0.3,CaloTowerList,CaloClusterR03List);
      SimpleConeCluster(ClusterTower,0.5,0.5,0.15,CaloTowerList,CaloClusterR15List);

      fillClusterPlot("CaloTowerClusterR05",CaloClusterR05List);
      fillClusterPlot("CaloTowerClusterR03",CaloClusterR03List);
      fillClusterPlot("CaloTowerClusterR015",CaloClusterR15List);

    }

  //   do local clustering using rechits

    if ((&caloTowers) && (&EBRecHits) && (&EERecHits) && (&HBHERecHits && (&HORecHits) && (&HFRecHits)) ){

      std::vector<CalCell> EmRecHitsList;
      MakeEmCellList(caloGeometry,EBRecHits,EERecHits,EmRecHitsList);
      std::vector<CalCell> HadRecHitsList;
      MakeHadCellList(caloGeometry,HBHERecHits,HORecHits,HFRecHits,HadRecHitsList);

      std::vector<CalCluster> EmRHClusterR0030List;
      SimpleConeCluster(ClusterEm,0.0,0.0,0.03,EmRecHitsList,EmRHClusterR0030List);

      std::vector<CalCluster> EmRHClusterR0045List;
      SimpleConeCluster(ClusterEm,0.0,0.0,0.045,EmRecHitsList,EmRHClusterR0045List);

      std::vector<CalCluster> EmRHClusterR0050List;
      SimpleConeCluster(ClusterEm,0.0,0.0,0.050,EmRecHitsList,EmRHClusterR0050List);

      std::vector<CalCluster> EmRHClusterR0060List;
      SimpleConeCluster(ClusterEm,0.0,0.0,0.060,EmRecHitsList,EmRHClusterR0060List);

      std::vector<CalCluster> EmRHClusterR0070List;
      SimpleConeCluster(ClusterEm,0.0,0.0,0.070,EmRecHitsList,EmRHClusterR0070List);

      std::vector<CalCluster> HdRHClusterR15List;
      SimpleConeCluster(ClusterHd,0.1,0.1,0.15,HadRecHitsList,HdRHClusterR15List);

      std::vector<CalCluster> HdRHClusterR25List;
      SimpleConeCluster(ClusterHd,0.1,0.1,0.25,HadRecHitsList,HdRHClusterR25List);

      fillClusterPlot("EmRecHitClusterR0030",EmRHClusterR0030List);
      fillClusterPlot("EmRecHitClusterR0045",EmRHClusterR0045List);
      fillClusterPlot("EmRecHitClusterR0050",EmRHClusterR0050List);
      fillClusterPlot("EmRecHitClusterR0060",EmRHClusterR0060List);
      fillClusterPlot("EmRecHitClusterR0070",EmRHClusterR0070List);

      fillClusterPlot("HadRecHitClusterR015",HdRHClusterR15List);
      fillClusterPlot("HadRecHitClusterR025",HdRHClusterR25List);
 

      //  Now use towers 

      std::vector<CalCell> EmCalCellList;
      std::vector<CalCell> HdCalCellList;




      MakeCellListFromCaloTowers(caloGeometry,caloTowers,EBRecHits,EERecHits,HBHERecHits,HORecHits,HFRecHits,EmCalCellList,HdCalCellList);

      std::vector<CalCluster> ClusterFromTowerEmCellList;
      SimpleConeCluster(CaloTowerEm,0.0,0.0,0.5,EmCalCellList,ClusterFromTowerEmCellList);

      std::vector<CalCell> HadTowerList;
      MakeHadCellList(caloTowers,HadTowerList);

      std::vector<CalCluster> ClusterFromTowerHdCellList;
      SimpleConeCluster(CaloTowerHd,0.0,0.0,0.5,HdCalCellList,ClusterFromTowerHdCellList);

      std::vector<CalCluster> EmHdClusterTowerList;
      MatchEmHadClusters(ClusterFromTowerEmCellList,ClusterFromTowerHdCellList,EmHdClusterTowerList);
    //  cout <<" Number of clusters " << EmHdClusterList.size() << endl;
    // cout << " Number of Towers "<< CaloTowerList.size() << " CaloClusters " <<CaloClusterR05List.size() << " Njets " << calojets.size() << endl;

      std::vector<CalCluster> EmHdRecHitClusterList;
      MatchEmHadClusters(EmRHClusterR0030List,HdRHClusterR15List,EmHdRecHitClusterList);
 
   //  cout <<" Number of clusters " << EmHdClusterList.size() << endl;

      //      fillSubClusterPlot(CaloClusterR05List,CaloClusterR05List,CaloClusterR15List,HdClusterR15List,
      //	       EmRHClusterR003List,EmHdClusterList);




      double sumEtRecMet(0);
      double missEtRecMet(0);
      for (CaloMETCollection::const_iterator met=recmets.begin(); met!=recmets.end(); met++){
	missEtRecMet=met->et();
	sumEtRecMet=met->sumEt();
      }

      double sumEtCluster(0.);
      double missEtCluster(0.);

      double sumEtRecHit(0.);
      double missEtRecHit(0.);

      CalculateSumEtMET(EmHdClusterTowerList,sumEtCluster,missEtCluster);
      CalculateSumEtMET(EmHdRecHitClusterList,sumEtRecHit,missEtRecHit);

      hname ="dSumEt";
      fillHist1D(hname,sumEtRecMet-sumEtCluster);

      hname ="dMet";  
      fillHist1D(hname,missEtRecMet-missEtCluster);

      hname ="Met2D";  
      fillHist2D(hname,missEtRecMet,missEtCluster);

      hname ="SumEt2D";    
      fillHist2D(hname,sumEtRecMet,sumEtCluster);

      hname ="dSumEtR";  
      fillHist1D(hname,sumEtRecHit-sumEtCluster);

      hname ="dMetR";  
      fillHist1D(hname,missEtRecHit-missEtCluster);  

      hname ="Met2DR";  
      fillHist2D(hname,missEtRecHit,missEtCluster);

      hname ="SumEt2DR";  
      fillHist2D(hname,sumEtRecHit,sumEtCluster);


      hname ="SumEt2DR2";  
      fillHist2D(hname,sumEtRecHit,sumEtCluster);
    }
}

////////////////////////////////////////////////////
template <typename T> void JetAnalyzer::fillRecHitHists(const CaloGeometry& caloGeometry,const T& RecHits,const TString subDetName,double& sumEnergy){


  //  TString subDetName;
  //  subDetName="EB";

  int nRecHits=RecHits.size();

  TString hname="NumberOfRecHits"+subDetName;
  fillHist1D(hname,nRecHits);

  typedef typename T::const_iterator iter;

  int nRecHitsNZ(0);

  double sumPx(0);
  double sumPy(0);
  double sumPz(0);
  sumEnergy=0.0;

  HepLorentzVector sumP4(0.,0.,0.,0.);

  for(iter theRecHit=RecHits.begin();theRecHit!=RecHits.end();theRecHit++){
    double energy = theRecHit->energy();
    if(energy>0.0){
      nRecHitsNZ++;

      DetId RecHitDetID = theRecHit->detid();
      DetId::Detector DetNum=RecHitDetID.det();

      bool UseIt=true;

      if(DetNum == DetId::Hcal ){
	HcalDetId HcalID = RecHitDetID;
	HcalSubdetector HcalNum = HcalID.subdet();

	if(HcalNum == HcalBarrel && subDetName != "HB") UseIt=false; 
	if(HcalNum == HcalEndcap && subDetName != "HE") UseIt=false;
      }

      if(!UseIt) continue;

    //  int RecHitIeta  = RecHitDetID.ieta();
    //  int RecHitPhi  = RecHitDetID.iphi();
      const CaloCellGeometry *this_cell = caloGeometry.getGeometry(RecHitDetID);
      GlobalPoint position = this_cell->getPosition();

      double eta=position.eta();

      //      if(subDetName=="HB" && fabs(eta)>1.5) continue;
      //   if(subDetName=="HE" && fabs(eta)<1.5) continue;

      double phi=position.phi();
      double theta=position.theta();

      double pt=energy*sin(theta);

      double px=energy*sin(theta)*cos(phi);
      double py=energy*sin(theta)*sin(phi);
      double pz=energy*cos(theta);
      double e=energy;

      sumPx +=px;
      sumPy +=py;
      sumPz +=pz;
      sumEnergy  +=e;

      hname="PtRecHits"+subDetName;
      fillHist1D(hname,pt);

      hname="EnergyRecHits"+subDetName;
      fillHist1D(hname,energy);

      hname="EtaRecHits"+subDetName;
      fillHist1D(hname,eta);

      hname="EtaPtWeightedRecHits"+subDetName;
      fillHist1D(hname,eta,pt);

      hname="PhiRecHits"+subDetName;
      fillHist1D(hname,phi);
    }  
  }

  hname="NumberOfRecHitsNZ"+subDetName;
  fillHist1D(hname,nRecHitsNZ);

  hname="SumEnergyRecHits"+subDetName;
  fillHist1D(hname,sumEnergy);

  hname="SumEnergyRecHitsA"+subDetName;
  fillHist1D(hname,sumEnergy);

  double pt=sqrt(sumPx+sumPx+sumPy*sumPy);

  hname="SumPtRecHits"+subDetName;
  fillHist1D(hname,pt);


  hname="SumPtRecHitsA"+subDetName;
  fillHist1D(hname,pt);

}
void JetAnalyzer::bookFillEoPCorrectionPlots(){

  TString hname;
  TString htitle;

  for(int i=0;i<50;i++){
    std::ostringstream oi; oi << i;
    hname="EoPCorrection"+oi.str();
    htitle="EoPCorrection"+oi.str();
    m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,1.0);

    hname="F0"+oi.str();
    htitle="F0"+oi.str();
    m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,1.0);

    hname="EoHEm"+oi.str();
    htitle="EoHEm"+oi.str();
    m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,1.0);

    hname="EoHHd"+oi.str();
    htitle="EoHHd"+oi.str();
    m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,1.0);

    hname= "EoPiEm"+oi.str();
    htitle="EoPiEm"+oi.str();
    m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,1.0);

    hname= "EoPiHd"+oi.str();
    htitle="EoPiHd"+oi.str();
    m_HistNames[hname] = new TH1F(hname,htitle,100,0.0,1.0);
  }

  for(int i=0;i<10;i++){
    std::ostringstream oi; oi << i;
    TString hname="EoPCorrection"+oi.str();
    TString hnamef0="F0"+oi.str();
    TString hnameEoHEm="EoHEm"+oi.str();
    TString hnameEoHHd="EoHHd"+oi.str();
    TString hnameEoPiEm="EoPiEm"+oi.str();
    TString hnameEoPiHd="EoPiHd"+oi.str();

    double Etot=(i+1)*1.0;
    //  cout << " Etot " << Etot << endl;
    for(int j=0;j<21;j++){
      double Em= double(j)*0.05*Etot;
      double Had=Etot-Em;
      double f0(0);
      double EoHEm(0);
      double EoHHd(0);

      double EoPiEm(0);
      double EoPiHd(0);
  

      double CorrectedEnergy=CorrectedE(Em,Had,1,f0,EoHEm,EoHHd,EoPiEm,EoPiHd);
      double CF=CorrectedEnergy/(Etot);
      //cout << " Etot " << Etot << " Em " << Em << " Had " << Had << " Corrected E " << CorrectedEnergy << " R " << CorrectedEnergy/Etot <<endl;

      fillHist1D(hname,Em/Etot,CF);
      fillHist1D(hnamef0,Em/Etot,f0);
      fillHist1D(hnameEoHEm,Em/Etot,EoHEm);
      fillHist1D(hnameEoHHd,Em/Etot,EoHHd);
      fillHist1D(hnameEoPiEm,Em/Etot,EoPiEm);
      fillHist1D(hnameEoPiHd,Em/Etot,EoPiHd);
    }
  }

  for(int i=10;i<50;i++){
    std::ostringstream oi; oi << i;
    TString hname="EoPCorrection"+oi.str();
    TString hnamef0="F0"+oi.str();
    TString hnameEoHEm="EoHEm"+oi.str();
    TString hnameEoHHd="EoHHd"+oi.str();
    TString hnameEoPiEm="EoPiEm"+oi.str();
    TString hnameEoPiHd="EoPiHd"+oi.str();

    double Etot=(i-8)*5.0;
    // cout << " Etot " << Etot << endl;
    for(int j=0;j<21;j++){
      double Em= double(j)*0.05*Etot;
      double Had=Etot-Em;
      double f0(0);
      double EoHEm(0);
      double EoHHd(0);

      double EoPiEm(0);
      double EoPiHd(0);
  

      double CorrectedEnergy=CorrectedE(Em,Had,1,f0,EoHEm,EoHHd,EoPiEm,EoPiHd);
      double CF=CorrectedEnergy/(Etot);
      //  cout << " Etot " << Etot << " Em " << Em << " Had " << Had << " Corrected E " << CorrectedEnergy << " R " << CorrectedEnergy/Etot <<endl;

      fillHist1D(hname,Em/Etot,CF);
      fillHist1D(hnamef0,Em/Etot,f0);
      fillHist1D(hnameEoHEm,Em/Etot,EoHEm);
      fillHist1D(hnameEoHHd,Em/Etot,EoHHd);
      fillHist1D(hnameEoPiEm,Em/Etot,EoPiEm);
      fillHist1D(hnameEoPiHd,Em/Etot,EoPiHd);
    }
  }
}
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE(); 
DEFINE_FWK_MODULE( JetAnalyzer );

