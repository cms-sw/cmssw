/*
 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
// Original Author: Anna Cimmino

#include "../interface/HLTOniaSource.h"
//FWCore
#include "FWCore/ServiceRegistry/interface/Service.h"
//DataFormats
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/TrackReco/interface/Track.h"

//HLTrigger
#include "HLTrigger/HLTanalyzers/interface/JetUtil.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

using namespace edm;
using namespace std;
using namespace reco;
using namespace trigger;

HLTOniaSource::HLTOniaSource(const edm::ParameterSet& pset){

  LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Constructor";
 
  //HLTrigger Path Names
  vector<string> myTriggerPaths;
  myTriggerPaths.push_back("HLT_Mu0_Track0_Jpsi");
  myTriggerPaths.push_back("HLT_Mu3_Track0_Jpsi");
  myTriggerPaths.push_back("HLT_Mu5_Track0_Jpsi");
  triggerPath_ = pset.getUntrackedParameter<vector<string> >("TriggerPathNames",myTriggerPaths);

  //Tag for Onia Muons
  vector<InputTag> myOniaMuonTags;
                                         
  myOniaMuonTags.push_back(InputTag("hltMu0TrackJpsiL3Filtered0", "", "HLT"));
  myOniaMuonTags.push_back(InputTag("hltMu3TrackJpsiL3Filtered3", "", "HLT"));
  myOniaMuonTags.push_back(InputTag("hltMu5TrackJpsiL3Filtered5", "", "HLT"));
  oniaMuonTag_ = pset.getUntrackedParameter<std::vector<edm::InputTag> >("OniaMuonTag",myOniaMuonTags);

  //Tag for Pixel tracks before Onia filter
  pixelTag_ = pset.getUntrackedParameter<InputTag>("PixelTag",edm::InputTag("hltPixelTracks", "", "HLT"));
 
  //Tag for Tracker tracks before Onia filter
  trackTag_ = pset.getUntrackedParameter<InputTag>("TrackTag",edm::InputTag("hltMuTrackJpsiCtfTrackCands","", "HLT"));

  beamSpotTag_ = pset.getUntrackedParameter<InputTag>("BeamSpotTag",edm::InputTag("hltOfflineBeamSpot", "", "HLT"));

  //Tag Trigger Summary
  triggerSummaryRAWTag_ = pset.getUntrackedParameter<InputTag>("TriggerSummaryTag",edm::InputTag("hltTriggerSummaryRAW", "", "HLT"));
  hltProcessName_  = pset.getUntrackedParameter<string>("TriggerProcessName","HLT");
  //Tag for Pixel tracks after Onia filter
  vector<InputTag> pxlTagsAfterFilter;
  pxlTagsAfterFilter.push_back(InputTag("hltMu0TrackJpsiPixelMassFiltered", "", "HLT"));
  pxlTagsAfterFilter.push_back(InputTag("hltMu3TrackJpsiPixelMassFiltered", "", "HLT"));
  pxlTagsAfterFilter.push_back(InputTag("hltMu5TrackJpsiPixelMassFiltered", "", "HLT"));
  pixelTagsAfterFilter_=  pset.getUntrackedParameter< vector<edm::InputTag> >("PixelTagAfterFilter",pxlTagsAfterFilter);

  //Tag for Tracker tracks after Onia filter
  vector<InputTag> trxTagsAfterFilter;
  trxTagsAfterFilter.push_back(InputTag("hltMu0TrackJpsiTrackMassFiltered", "", "HLT"));
  trxTagsAfterFilter.push_back(InputTag("hltMu3TrackJpsiTrackMassFiltered", "", "HLT"));
  trxTagsAfterFilter.push_back(InputTag("hltMu5TrackJpsiTrackMassFiltered", "", "HLT"));
  trackTagsAfterFilter_ = pset.getUntrackedParameter< vector<edm::InputTag> >("TrackTagAfterFilter",trxTagsAfterFilter);

  //Foldering output
  subsystemFolder_ = pset.getUntrackedParameter<string>("SubSystemFolder","HLT/HLTMonMuon/Onia");
}


HLTOniaSource::~HLTOniaSource(){dbe_ = 0;}

void  HLTOniaSource::beginJob(){

  dbe_ = Service<DQMStore>().operator->();
  if( !dbe_ ) {
    LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Could not access DQM Store.";
    return;
  }

  // Book Pixel Histos
  if (pixelTag_.label()!= ""){
    dbe_->setCurrentFolder(subsystemFolder_+"/Pixel");
    this->bookOniaTriggerMEs(pixelME_,  pixelTag_.label());
  }

  // Book Track Histos
  if (trackTag_.label()!= ""){ 
    dbe_->setCurrentFolder(subsystemFolder_+"/Track");
    this->bookOniaTriggerMEs(trackME_, trackTag_.label());
  }
  
 
  //Book Onia Histos
  for (size_t i = 0 ; i<oniaMuonTag_.size() && i<pixelTagsAfterFilter_.size() && i<trackTagsAfterFilter_.size(); i++){
  
    if (oniaMuonTag_[i].label()!= "") {
      dbe_->setCurrentFolder(subsystemFolder_+"/MuonFilters");
      this->bookOniaTriggerMEs(muonME_, oniaMuonTag_[i].label()); 
    }
    if (pixelTagsAfterFilter_[i].label() != ""){ 
      dbe_->setCurrentFolder(subsystemFolder_+"/PixelFilters");
      this->bookOniaTriggerMEs(pixelAfterFilterME_, pixelTagsAfterFilter_[i].label() );
    }   
    if (trackTagsAfterFilter_[i].label() != ""){ 
      dbe_->setCurrentFolder(subsystemFolder_+"/TrackFilters");
      this->bookOniaTriggerMEs(trackAfterFilterME_, trackTagsAfterFilter_[i].label() );
    } 
    if (oniaMuonTag_[i].label()!= "" && pixelTag_.label() != ""){
      dbe_->setCurrentFolder(subsystemFolder_+"/Pixel");
      this->bookOniaTriggerInvariantMassMEs( massME_, oniaMuonTag_[i].label(),pixelTag_.label() );
    }
    if (oniaMuonTag_[i].label() != "" && trackTag_.label()!= "" ){ 
      dbe_->setCurrentFolder(subsystemFolder_+"/Track");
      this->bookOniaTriggerInvariantMassMEs( massME_, oniaMuonTag_[i].label(), trackTag_.label() );
    }
    if (oniaMuonTag_[i].label()!= "" && pixelTagsAfterFilter_[i].label()!= ""){
      dbe_->setCurrentFolder(subsystemFolder_+"/PixelFilters");
      this->bookOniaTriggerInvariantMassMEs( massME_,oniaMuonTag_[i].label(),pixelTagsAfterFilter_[i].label() );
    }  
    if (oniaMuonTag_[i].label()!= "" &&  trackTagsAfterFilter_[i].label()!= ""){ 
      dbe_->setCurrentFolder(subsystemFolder_+"/TrackFilters");
      this->bookOniaTriggerInvariantMassMEs( massME_, oniaMuonTag_[i].label(), trackTagsAfterFilter_[i].label() );
    }
  }
  //Check HLT Trigger Configuration
  if(hltProcessName_ != "" ) this->checkHLTConfiguration(hltProcessName_);
  
}


void HLTOniaSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
   
  //Get Pixel Tracks
  Handle<TrackCollection> pixelCands;
  iEvent.getByLabel(pixelTag_, pixelCands);

  TrackCollection mypixelCands; //This is needed for the sort!!!!
  if (pixelCands.isValid()) {
    mypixelCands =  *  pixelCands;   
    sort(mypixelCands.begin(), mypixelCands.end(),PtGreater());  
    this->fillOniaTriggerMEs(pixelCands , pixelTag_.label(), pixelME_ );
  }else {
    LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Could not access pixel collection with tag "<<pixelTag_;
  }
   
  //Get Tracker Tracks
  Handle<RecoChargedCandidateCollection>  trackCands;
  iEvent.getByLabel(trackTag_, trackCands);
  RecoChargedCandidateCollection mytrackCands; //This is needed for the sort!!!!
  if(trackCands.isValid()) {
    mytrackCands =  * trackCands;   
    sort(mytrackCands.begin(),mytrackCands.end(),PtGreater());  
    this->fillOniaTriggerMEs(trackCands ,  trackTag_.label(), trackME_ );   
   }else {
    LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Could not access track collection with tag "<<trackTag_;
   }
    
  //Get Beamspot 
  Handle<BeamSpot> recoBeamSpotHandle;
  iEvent.getByLabel(beamSpotTag_, recoBeamSpotHandle);
  if (recoBeamSpotHandle.isValid()) {
    BSPosition_ = recoBeamSpotHandle->position();
  }else {
    LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Could not access beam spot info with tag "<<beamSpotTag_;
  }  


  //Get Trigger Summary RA
  Handle<TriggerEventWithRefs> rawTriggerEvent;
  iEvent.getByLabel(triggerSummaryRAWTag_, rawTriggerEvent );
  
  if( rawTriggerEvent.isValid() ){

    for(size_t i=0; i<oniaMuonTag_.size(); i++){

      vector<RecoChargedCandidateRef> myMuonFilterCands;        
      vector<RecoChargedCandidateRef> myPixelFilterCands;   
      vector<RecoChargedCandidateRef> myTrackFilterCands;   
      
      //Get Onia Muons
      size_t indexM = rawTriggerEvent->filterIndex(oniaMuonTag_[i]);

      if ( indexM < rawTriggerEvent->size() ){
	rawTriggerEvent->getObjects( indexM, TriggerMuon, myMuonFilterCands );
	this->fillOniaTriggerMEs( myMuonFilterCands,  oniaMuonTag_[i].label(), muonME_ );
      }else{
	LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Did not find muons with tag "<<oniaMuonTag_[i];
      }
      //Get Onia Pixel
      size_t indexP = rawTriggerEvent->filterIndex(pixelTagsAfterFilter_[i]);
      if ( indexP < rawTriggerEvent->size() ){
	rawTriggerEvent->getObjects( indexP, TriggerTrack , myPixelFilterCands );
	this->fillOniaTriggerMEs( myPixelFilterCands,pixelTagsAfterFilter_[i].label(), pixelAfterFilterME_);   
	sort(myPixelFilterCands.begin(), myPixelFilterCands.end(),PtGreaterRef());    
      }else{
	LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Did not find pixel with tag "<<pixelTagsAfterFilter_[i];
      }

      //Get Onia Tracker Tracks
      size_t indexT = rawTriggerEvent->filterIndex(trackTagsAfterFilter_[i]);
      if ( indexT < rawTriggerEvent->size() ){
	rawTriggerEvent->getObjects( indexT, TriggerTrack , myTrackFilterCands );
     	this->fillOniaTriggerMEs( myTrackFilterCands,trackTagsAfterFilter_[i].label(), trackAfterFilterME_ );   
	sort(myTrackFilterCands.begin(), myTrackFilterCands.end(),PtGreaterRef());    
      }else{
	LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Did not find tracks with tag "<<trackTagsAfterFilter_[i];
      }

      if( myMuonFilterCands.size() > 0){
	if ( myPixelFilterCands.size() > 0 )this->fillInvariantMass( myMuonFilterCands, myPixelFilterCands ,oniaMuonTag_[i].label(),pixelTagsAfterFilter_[i].label());
	if ( myTrackFilterCands.size() > 0 )this->fillInvariantMass( myMuonFilterCands, myTrackFilterCands ,oniaMuonTag_[i].label(),trackTagsAfterFilter_[i].label());
	if (pixelCands.isValid())   this->fillInvariantMass( myMuonFilterCands, mypixelCands , oniaMuonTag_[i].label(), pixelTag_.label());
      	if (trackCands.isValid())   this->fillInvariantMass( myMuonFilterCands, mytrackCands , oniaMuonTag_[i].label(), trackTag_.label());
      }
   

    }//ADD INVARIANT MASSES
  }else{
    LogVerbatim ("oniatriggermonitor") << "[HLTOniaSource]: Could not access trigger collection with tag "<<triggerSummaryRAWTag_;
  }
  
}


void  HLTOniaSource::endJob() {}

void  HLTOniaSource::bookOniaTriggerMEs( map<string, MonitorElement *>  & myMap, string meName){

    stringstream myMeName;

    //PT    
    myMeName.str("");
    myMeName<<meName<<"_Pt";
    myMap[meName+"pt"]=dbe_->book1D(myMeName.str(), myMeName.str(),200, 0.0, 20.0);
    myMap[meName+"pt"]->setAxisTitle("Pt (GeV)", 1);

    //P    
    myMeName.str("");
    myMeName<<meName<<"_P";
    myMap[meName+"p"]=dbe_->book1D(myMeName.str(), myMeName.str(),250, 0.0, 50.0);
    myMap[meName+"p"]->setAxisTitle("P (GeV)", 1);
    
    //Eta
    myMeName.str("");
    myMeName<<meName<<"_Eta";
    myMap[meName+"eta"]=dbe_->book1D(myMeName.str(), myMeName.str(), 50, -2.5, 2.5 );
    myMap[meName+"eta"]->setAxisTitle("Eta", 1);
    
    //Phi
    myMeName.str("");
    myMeName<<meName<<"_Phi";
    myMap[meName+"phi"]=dbe_->book1D(myMeName.str(), myMeName.str(), 144, -3.1416, 3.1416 );
    myMap[meName+"phi"]->setAxisTitle("Phi", 1);

    //Phi
    myMeName.str("");
    myMeName<<meName<<"_Phi";
    myMap[meName+"phi"]=dbe_->book1D(myMeName.str(), myMeName.str(), 144, -3.1416, 3.1416 );
    myMap[meName+"phi"]->setAxisTitle("Phi", 1);

    //Charge
    myMeName.str("");
    myMeName<<meName<<"_Charge";
    myMap[meName+"charge"]=dbe_->book1D(myMeName.str(), myMeName.str(), 3, -1.5, 1.5 );
    myMap[meName+"charge"]->setAxisTitle("charge", 1);

    //Dz
    myMeName.str("");
    myMeName<<meName<<"_Dz";
    myMap[meName+"dz"]=dbe_->book1D(myMeName.str(), myMeName.str(), 400, -20.0, 20.0 );
    myMap[meName+"dz"]->setAxisTitle("dz", 1);

    //Dxy
    myMeName.str("");
    myMeName<<meName<<"_Dxy";
    myMap[meName+"dxy"]=dbe_->book1D(myMeName.str(), myMeName.str(), 100, -0.5, 0.5 );
    myMap[meName+"dxy"]->setAxisTitle("dxy", 1);

    //EtaVsPhi
    myMeName.str("");
    myMeName<<meName<<"_EtaPhi";
    myMap[meName+"etaphi"]=dbe_->book2D(myMeName.str(), myMeName.str(), 144, -3.1416, 3.1416 ,50, -2.5, 2.5 );
    myMap[meName+"etaphi"]->setAxisTitle("Phi", 1);
    myMap[meName+"etaphi"]->setAxisTitle("Eta", 2);

    //EtaVsPt
    myMeName.str("");
    myMeName<<meName<<"_EtaPt";
    myMap[meName+"etapt"]=dbe_->book2D(myMeName.str(), myMeName.str(), 100, 0.0, 100.0, 50, -2.5, 2.5 );
    myMap[meName+"etapt"]->setAxisTitle("Pt (GeV)", 1);
    myMap[meName+"etapt"]->setAxisTitle("Eta", 2);

    //ValidHits
    myMeName.str("");
    myMeName<<meName<<"_ValidHits";
    myMap[meName+"validhits"]=dbe_->book1D(myMeName.str(), myMeName.str(), 50, 0.0, 50.0 );
    myMap[meName+"validhits"]->setAxisTitle("ValidHits", 1);

    //Norm Chi2
    myMeName.str("");
    myMeName<<meName<<"_NormChi2";
    myMap[meName+"normchi"]=dbe_->book1D(myMeName.str(), myMeName.str(), 300, 0.0, 30.0 );
    myMap[meName+"normchi"]->setAxisTitle("Normalized Chi2", 1);

    //Number Of Candi
    myMeName.str("");
    myMeName<<meName<<"_NrCandidates";
    myMap[meName+"nrcand"]=dbe_->book1D(myMeName.str(), myMeName.str(), 50, 0.0, 50.0 );
    myMap[meName+"nrcand"]->setAxisTitle("Nr per Event", 1);
    //return true;
}


void  HLTOniaSource::bookOniaTriggerInvariantMassMEs( map<string, MonitorElement *>  & myMap, string label1, string label2 ){

  stringstream meName;
  //Same charge 
  meName.str("");
  meName<<label1<<"_"<<label2<<"_SameCharge_InvariantMass";
  massME_[label1+label2+"same"]=dbe_->book1D(meName.str(), meName.str(),120, 0.0, 6.0);
  massME_[label1+label2+"same"]->setAxisTitle("mass (GeV)", 1);

  //Opposite charge 
  meName.str("");
  meName<<label1<<"_"<<label2<<"_OppositeCharge_InvariantMass";
  massME_[label1+label2+"opposite"]=dbe_->book1D(meName.str(), meName.str(),120, 0.0, 6.0);
  massME_[label1+label2+"opposite"]->setAxisTitle("mass (GeV)", 1);
      
  //Same charge Highest PT
  meName.str("");
  meName<<label1<<"_"<<label2<<"_SameCharge_HighestPT_InvariantMass";
  massME_[label1+label2+"same"+"highestpt"]=dbe_->book1D(meName.str(), meName.str(),120, 0.0, 6.0);
  massME_[label1+label2+"same"+"highestpt"]->setAxisTitle("mass (GeV)", 1);
          
  //Opposite charge Highest PT
  meName.str("");
  meName<<label1<<"_"<<label2<<"_OppositeCharge_HighestPT_InvariantMass";
  massME_[label1+label2+"opposite"+"highestpt"]=dbe_->book1D(meName.str(), meName.str(),120, 0.0, 6.0);
  massME_[label1+label2+"opposite"+"highestpt"]->setAxisTitle("mass (GeV)", 1);

 
 
  // Same Charge Deltaz Muon - Track  Highest PT
  meName.str("");
  meName<<label1<<"_"<<label2<<"_SameCharge_HighestPT_MaxDzMuonTrack";
  massME_[label1+label2+"same"+"highestpt"+"maxdzmuontrack"]=dbe_->book1D(meName.str(), meName.str(),50, 0.0, 5.0);
  massME_[label1+label2+"same"+"highestpt"+"maxdzmuontrack"]->setAxisTitle("MaxDz Muon-Track", 1);
  // Same Charge Deltaz Muon - Track
  meName.str("");
  meName<<label1<<"_"<<label2<<"_SameCharge_MaxDzMuonTrack";
  massME_[label1+label2+"same"+"maxdzmuontrack"]=dbe_->book1D(meName.str(), meName.str(),50, 0.0, 5.0);
  massME_[label1+label2+"same"+"maxdzmuontrack"]->setAxisTitle("MaxDz Muon-Track", 1);
  // Opposite charge  Deltaz Muon - Track Highest PT
  meName.str("");
  meName<<label1<<"_"<<label2<<"_OppositeCharge_HighestPT_MaxDzMuonTrack";
  massME_[label1+label2+"opposite"+"highestpt"+"maxdzmuontrack"]=dbe_->book1D(meName.str(), meName.str(),50, 0.0, 5.0);
  massME_[label1+label2+"opposite"+"highestpt"+"maxdzmuontrack"]->setAxisTitle("MaxDz Muon-Track", 1);   
  // Opposite charge  Deltaz Muon - Track
  meName.str("");
  meName<<label1<<"_"<<label2<<"_OppositeCharge_MaxDzMuonTrack";
  massME_[label1+label2+"opposite"+"maxdzmuontrack"]=dbe_->book1D(meName.str(), meName.str(),50, 0.0, 5.0);
  massME_[label1+label2+"opposite"+"maxdzmuontrack"]->setAxisTitle("MaxDz Muon-Track", 1);
}


void  HLTOniaSource::fillOniaTriggerMEs( Handle<TrackCollection> &  collection, string collectionLabel,  map<string, MonitorElement *>  & mapME ){
   // cout << "[HLTOniaSource]: fillOniaTriggerMEs " << collectionLabel << endl;

  TrackCollection myCollection;
  if (collection.isValid()) {
    myCollection = * collection;
 
    // int nCollection= myCollection.size();

    typedef TrackCollection::const_iterator cand;
    int num = 0;
    for (cand tk=myCollection.begin(); tk!=myCollection.end(); tk++) {
      num++;
      //Fill MEs  
      if(mapME[collectionLabel+"pt"]){  mapME[collectionLabel+"pt"]->Fill(tk->pt()); }
      if(mapME[collectionLabel+"p"])  { mapME[collectionLabel+"p"]->Fill(tk->p()); }
      if(mapME[collectionLabel+"eta"]) { mapME[collectionLabel+"eta"]->Fill(tk->eta()); }
      if(mapME[collectionLabel+"phi"]) { mapME[collectionLabel+"phi"]->Fill(tk->phi()); }
      if(mapME[collectionLabel+"etaphi"]){ mapME[collectionLabel+"etaphi"]->Fill(tk->phi(),tk->eta()); }
      if(mapME[collectionLabel+"etapt"]){ mapME[collectionLabel+"etapt"]->Fill(tk->pt(),tk->eta()); }
      if(mapME[collectionLabel+"charge"]){ mapME[collectionLabel+"charge"]->Fill(tk->charge()); }

      if(mapME[collectionLabel+"dxy"]){ mapME[collectionLabel+"dxy"]->Fill(tk->dxy(BSPosition_)); }
      if(mapME[collectionLabel+"dz"]){ mapME[collectionLabel+"dz"]->Fill(tk->dz(BSPosition_)); }

      if(mapME[collectionLabel+"validhits"]) { mapME[collectionLabel+"validhits"]->Fill(tk->numberOfValidHits()); }
      if(mapME[collectionLabel+"normchi"]){  mapME[collectionLabel+"normchi"]->Fill(tk->normalizedChi2()); }
    }
    if(mapME[collectionLabel+"nrcand"]){  mapME[collectionLabel+"nrcand"]->Fill(num);}

  }
}


void  HLTOniaSource::fillOniaTriggerMEs(std::vector<reco::RecoChargedCandidateRef>  &  candidateVector, string collectionLabel,  map<string, MonitorElement *>  & mapME ){
    
  for (unsigned int  i=0; i!=candidateVector.size(); i++) {
   TrackRef tk = candidateVector[i]->get<TrackRef>();

   //Fill MEs  
   if(mapME[collectionLabel+"pt"]){  mapME[collectionLabel+"pt"]->Fill(tk->pt()); }
   if(mapME[collectionLabel+"p"])  { mapME[collectionLabel+"p"]->Fill(tk->p()); }
   if(mapME[collectionLabel+"eta"]) { mapME[collectionLabel+"eta"]->Fill(tk->eta()); }
   if(mapME[collectionLabel+"phi"]) { mapME[collectionLabel+"phi"]->Fill(tk->phi()); }
   if(mapME[collectionLabel+"etaphi"]){ mapME[collectionLabel+"etaphi"]->Fill(tk->phi(),tk->eta()); }
   if(mapME[collectionLabel+"etapt"]){ mapME[collectionLabel+"etapt"]->Fill(tk->pt(),tk->eta()); }
   if(mapME[collectionLabel+"charge"]){ mapME[collectionLabel+"charge"]->Fill(tk->charge()); }
   
   if(mapME[collectionLabel+"dxy"]){ mapME[collectionLabel+"dxy"]->Fill(tk->dxy(BSPosition_)); }
   if(mapME[collectionLabel+"dz"]){ mapME[collectionLabel+"dz"]->Fill(tk->dz(BSPosition_)); }
   
   if(mapME[collectionLabel+"validhits"]) { mapME[collectionLabel+"validhits"]->Fill(tk->numberOfValidHits()); }
   if(mapME[collectionLabel+"normchi"]){  mapME[collectionLabel+"normchi"]->Fill(tk->normalizedChi2()); }
  }
 
  if(mapME[collectionLabel+"nrcand"]){  mapME[collectionLabel+"nrcand"]->Fill( candidateVector.size());}
}


void  HLTOniaSource::fillOniaTriggerMEs( Handle<RecoChargedCandidateCollection> &  collection, string collectionLabel,  map<string, MonitorElement *>  & mapME ){

  RecoChargedCandidateCollection myCollection;
  if (collection.isValid()) {
    myCollection = * collection;
 
    // int nCollection= myCollection.size();
    int num = 0;
    typedef RecoChargedCandidateCollection::const_iterator cand;
    for (cand i=myCollection.begin(); i!=myCollection.end(); i++) {
      TrackRef tk = i->get<TrackRef>();
      num++; 
     //Fill MEs  
      if(mapME[collectionLabel+"pt"]){  mapME[collectionLabel+"pt"]->Fill(tk->pt()); }
      if(mapME[collectionLabel+"p"])  { mapME[collectionLabel+"p"]->Fill(tk->p()); }
      if(mapME[collectionLabel+"eta"]) { mapME[collectionLabel+"eta"]->Fill(tk->eta()); }
      if(mapME[collectionLabel+"phi"]) { mapME[collectionLabel+"phi"]->Fill(tk->phi()); }
      if(mapME[collectionLabel+"etaphi"]){ mapME[collectionLabel+"etaphi"]->Fill(tk->phi(),tk->eta()); }
      if(mapME[collectionLabel+"etapt"]){ mapME[collectionLabel+"etapt"]->Fill(tk->pt(),tk->eta()); }
      if(mapME[collectionLabel+"charge"]){ mapME[collectionLabel+"charge"]->Fill(tk->charge()); }

      if(mapME[collectionLabel+"dxy"]){ mapME[collectionLabel+"dxy"]->Fill(tk->dxy(BSPosition_)); }
      if(mapME[collectionLabel+"dz"]){ mapME[collectionLabel+"dz"]->Fill(tk->dz(BSPosition_)); }

      if(mapME[collectionLabel+"validhits"]) { mapME[collectionLabel+"validhits"]->Fill(tk->numberOfValidHits()); }
      if(mapME[collectionLabel+"normchi"]){  mapME[collectionLabel+"normchi"]->Fill(tk->normalizedChi2()); }
    }
    if(mapME[collectionLabel+"nrcand"]){  mapME[collectionLabel+"nrcand"]->Fill(num);}
  }
}

 
void HLTOniaSource::fillInvariantMass(vector<RecoChargedCandidateRef> & cand1,  vector<RecoChargedCandidateRef> & cand2, string cand1Label, string  cand2Label){

    //Loop on collection to calculate invariate mass
    for(size_t i = 0 ; i< cand1.size(); i++) {
      if( cand2.size() > 0 )  {
	//Highest PT
	string chargeLabel = "same";
	
	//Check relative charge
	if(cand1[i]->charge() * cand2[0]->charge() < 0) chargeLabel = "opposite";
	if(massME_[cand1Label+cand2Label+chargeLabel+"highestpt"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"highestpt"]->Fill((cand1[i]->p4()+cand2[0]->p4()).mass());
	}
	if(massME_[cand1Label+cand2Label+chargeLabel+"highestpt"+"maxdzmuontrack"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"highestpt"+"maxdzmuontrack"]->Fill(fabs(cand1[i]->get<TrackRef>()->dz(BSPosition_) - cand2[0]->get<TrackRef>()->dz(BSPosition_)));
	}

      }

      for (size_t j= 0; j< cand2.size(); j++) {

	if(cand2[j]->p() < 3) continue; //Check if momentum is greater than 3GeV.
	string chargeLabel = "same";
	//Check relative charge
	if(cand1[i]->charge() * cand2[j]->charge() < 0) chargeLabel = "opposite";
	if(massME_[cand1Label+cand2Label+chargeLabel]){
	   massME_[cand1Label+cand2Label+chargeLabel]->Fill((cand1[i]->p4()+cand2[j]->p4()).mass());
	}
	if(massME_[cand1Label+cand2Label+chargeLabel+"maxdzmuontrack"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"maxdzmuontrack"]->Fill(fabs(cand1[i]->get<TrackRef>()->dz(BSPosition_) - cand2[j]->get<TrackRef>()->dz(BSPosition_)));
	}
      }
    }

    


}

void HLTOniaSource::fillInvariantMass(std::vector<reco::RecoChargedCandidateRef> & cand1,  RecoChargedCandidateCollection &  cand2, string cand1Label, string  cand2Label){
  
  typedef RecoChargedCandidateCollection::const_iterator cand;
    //Loop on collection to calculate invariate mass
    for(size_t i = 0 ; i< cand1.size(); i++) {
      //Highest PT
      if(cand2.begin() != cand2.end()  &&  cand2.begin()->p()>3) {
	cand candItr = cand2.begin();
	string chargeLabel = "same";	
	//Check relative charge
	if(cand1[i]->charge() * candItr->charge() < 0) chargeLabel = "opposite";
	if(massME_[cand1Label+cand2Label+chargeLabel+"highestpt"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"highestpt"]->Fill((cand1[i]->p4()+candItr->p4()).mass());
	}
	
	if(massME_[cand1Label+cand2Label+chargeLabel+"highestpt"+"maxdzmuontrack"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"highestpt"+"maxdzmuontrack"]->Fill(fabs(cand1[i]->get<TrackRef>()->dz(BSPosition_) - candItr->get<TrackRef>()->dz(BSPosition_)));
	}

      }

      for (cand candIter= cand2.begin(); candIter!=cand2.end(); candIter++) {
	if(candIter->p() < 3) continue; //Check if momentum is greater than 3GeV.
	string  chargeLabel = "same";
	//Check relative charge
	if(cand1[i]->charge() * candIter->charge() < 0) chargeLabel = "opposite";
	if(massME_[cand1Label+cand2Label+chargeLabel]){
	   massME_[cand1Label+cand2Label+chargeLabel]->Fill((cand1[i]->p4()+candIter->p4()).mass());
	}
	if(massME_[cand1Label+cand2Label+chargeLabel+"maxdzmuontrack"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"maxdzmuontrack"]->Fill(fabs(cand1[i]->get<TrackRef>()->dz(BSPosition_) - candIter->get<TrackRef>()->dz(BSPosition_)));
	}
      }
    }
}



void HLTOniaSource::fillInvariantMass(std::vector<reco::RecoChargedCandidateRef> & cand1,  TrackCollection &  cand2, string cand1Label, string  cand2Label){
  
  typedef TrackCollection::const_iterator cand;

    //Loop on collection to calculate invariate mass
    for(size_t i = 0 ; i< cand1.size(); i++) {
      //Highest PT
      if(cand2.begin() != cand2.end()  &&  cand2.begin()->p()>3) {
	cand candItr = cand2.begin();
	math::PtEtaPhiMLorentzVector bestPtCandLVector(candItr->pt(), candItr->eta(), candItr->phi(), 1.056);
 
	string chargeLabel = "same";
	
	//Check relative charge
	if(cand1[i]->charge() * candItr->charge() < 0) chargeLabel = "opposite";
	if(massME_[cand1Label+cand2Label+chargeLabel+"highestpt"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"highestpt"]->Fill((cand1[i]->p4()+bestPtCandLVector).mass());
	}
	
	if(massME_[cand1Label+cand2Label+chargeLabel+"highestpt"+"maxdzmuontrack"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"highestpt"+"maxdzmuontrack"]->Fill(fabs(cand1[i]->get<TrackRef>()->dz(BSPosition_) - candItr->dz(BSPosition_)));
	}
      }

      for (cand candIter= cand2.begin(); candIter!=cand2.end(); candIter++) {

	if(candIter->p() < 3) continue; //Check if momentum is greater than 3GeV.

	math::PtEtaPhiMLorentzVector candLVector(candIter->pt(), candIter->eta(), candIter->phi(), 1.056);
 
	string chargeLabel = "same";
	//Check relative charge
	if(cand1[i]->charge() * candIter->charge() < 0) chargeLabel = "opposite";
	if(massME_[cand1Label+cand2Label+chargeLabel]){
	   massME_[cand1Label+cand2Label+chargeLabel]->Fill((cand1[i]->p4()+candLVector).mass());
	}
	if(massME_[cand1Label+cand2Label+chargeLabel+"maxdzmuontrack"]){
	  massME_[cand1Label+cand2Label+chargeLabel+"maxdzmuontrack"]->Fill(fabs(cand1[i]->get<TrackRef>()->dz(BSPosition_) - candIter->dz(BSPosition_)));
	}

      }
    }
}

void HLTOniaSource::checkHLTConfiguration(string triggerProcessName){

  HLTConfigProvider hltConfig;
  if(hltConfig.init(triggerProcessName)){
    LogVerbatim("hltoniasource") << "Successfully initialized HLTConfigProvider with process name: "<<triggerProcessName<<endl;
  }else{
    LogVerbatim("hltoniasource") << "Could not initialize HLTConfigProvider with process name: "<<triggerProcessName<<endl;
  }

  stringstream os;
  vector<string> triggerNames = hltConfig.triggerNames();

  for( size_t i = 0; i < triggerNames.size(); i++) {
    if (find(triggerPath_.begin(), triggerPath_.end(), triggerNames[i]) == triggerPath_.end()) continue; 
    LogVerbatim("hltoniasource") << "[HLTOniaSource]: Trigger Path: "<<triggerNames[i]<<endl;
    vector<string> moduleNames = hltConfig.moduleLabels( triggerNames[i] );
    for( size_t j = 0; j < moduleNames.size(); j++) {
      TString name = moduleNames[j];
      LogVerbatim("hltoniasource") << "\t  Fliter Name: "<<moduleNames[j]<<endl;
    }
  }

}
