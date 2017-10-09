#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_Muon_Hadronic.h"

bool comparePt(Lepton a, Lepton b) { return (a.pt > b.pt); }


SUSY_HLT_Muon_Hadronic::SUSY_HLT_Muon_Hadronic(const edm::ParameterSet& ps)
{

  edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "Constructor SUSY_HLT_Muon_Hadronic::SUSY_HLT_Muon_Hadronic " << std::endl;
  // Get parameters from configuration file
  theTrigSummary_ = consumes<trigger::TriggerEvent>(ps.getParameter<edm::InputTag>("trigSummary"));
  theMuonCollection_ = consumes<reco::MuonCollection>(ps.getParameter<edm::InputTag>("MuonCollection")); 
  thePfMETCollection_ = consumes<reco::PFMETCollection>(ps.getParameter<edm::InputTag>("pfMETCollection"));
  thePfJetCollection_ = consumes<reco::PFJetCollection>(ps.getParameter<edm::InputTag>("pfJetCollection"));
  theCaloJetCollection_ = consumes<reco::CaloJetCollection>(ps.getParameter<edm::InputTag>("caloJetCollection"));
  triggerResults_ = consumes<edm::TriggerResults>(ps.getParameter<edm::InputTag>("TriggerResults"));
  HLTProcess_ = ps.getParameter<std::string>("HLTProcess");
  triggerPath_ = ps.getParameter<std::string>("TriggerPath");
  triggerPathAuxiliaryForMuon_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForMuon");
  triggerPathAuxiliaryForHadronic_ = ps.getParameter<std::string>("TriggerPathAuxiliaryForHadronic");
  triggerFilter_ = ps.getParameter<edm::InputTag>("TriggerFilter");
  ptMuonOffline_ = ps.getUntrackedParameter<double>("ptMuonOffline");
  etaMuonOffline_ = ps.getUntrackedParameter<double>("etaMuonOffline");
  HTOffline_ = ps.getUntrackedParameter<double>("HTOffline");
  METOffline_ = ps.getUntrackedParameter<double>("METOffline");
  ptThrJet_ = ps.getUntrackedParameter<double>("PtThrJet"); 
  etaThrJet_ = ps.getUntrackedParameter<double>("EtaThrJet");
}

SUSY_HLT_Muon_Hadronic::~SUSY_HLT_Muon_Hadronic()
{
   edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "Destructor SUSY_HLT_Muon_Hadronic::~SUSY_HLT_Muon_Hadronic " << std::endl;
}

void SUSY_HLT_Muon_Hadronic::dqmBeginRun(edm::Run const &run, edm::EventSetup const &e)
{
 
  bool changed;
  
  if (!fHltConfig.init(run, e, HLTProcess_, changed)) {
    edm::LogError("SUSY_HLT_Muon_Hadronic") << "Initialization of HLTConfigProvider failed!!";
    return;
  }

  bool pathFound = false;
  const std::vector<std::string> allTrigNames = fHltConfig.triggerNames();
  for(size_t j = 0; j <allTrigNames.size(); ++j) {
    if(allTrigNames[j].find(triggerPath_) != std::string::npos) {
      pathFound = true;
    }
  }

  if(!pathFound) {
    LogDebug ("SUSY_HLT_Muon_Hadronic") << "Path not found" << "\n";
    return;
  }
  //std::vector<std::string> filtertags = fHltConfig.moduleLabels( triggerPath_ );
  //triggerFilter_ = edm::InputTag(filtertags[filtertags.size()-1],"",fHltConfig.processName());  
  //triggerFilter_ = edm::InputTag("hltPFMET120Mu5L3PreFiltered", "", fHltConfig.processName());

  edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "SUSY_HLT_Muon_Hadronic::beginRun" << std::endl;
}

 void SUSY_HLT_Muon_Hadronic::bookHistograms(DQMStore::IBooker & ibooker_, edm::Run const &, edm::EventSetup const &)
{
  edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "SUSY_HLT_Muon_Hadronic::bookHistograms" << std::endl;
  //book at beginRun
  bookHistos(ibooker_);
}

void SUSY_HLT_Muon_Hadronic::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
  edm::EventSetup const& context)
{
   edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "SUSY_HLT_Muon_Hadronic::beginLuminosityBlock" << std::endl;
}



void SUSY_HLT_Muon_Hadronic::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "SUSY_HLT_Muon_Hadronic::analyze" << std::endl;


  //-------------------------------
  //--- MET
  //-------------------------------
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  e.getByToken(thePfMETCollection_, pfMETCollection);
  if ( !pfMETCollection.isValid() ){
    edm::LogError ("SUSY_HLT_Muon_Hadronic") << "invalid collection: PFMET" << "\n";
   return;
  }
  //-------------------------------
  //--- Jets
  //-------------------------------
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  e.getByToken (thePfJetCollection_,pfJetCollection);
  if ( !pfJetCollection.isValid() ){
    edm::LogError ("SUSY_HLT_Muon_Hadronic") << "invalid collection: PFJets" << "\n";
    return;
  }
  edm::Handle<reco::CaloJetCollection> caloJetCollection;
  e.getByToken (theCaloJetCollection_,caloJetCollection);
  if ( !caloJetCollection.isValid() ){
  edm::LogError ("SUSY_HLT_Muon_Hadronic") << "invalid collection: CaloJets" << "\n";
  return;
  }
  
  //-------------------------------
  //--- Muon
  //-------------------------------
  edm::Handle<reco::MuonCollection> MuonCollection;
  e.getByToken (theMuonCollection_, MuonCollection);
  if ( !MuonCollection.isValid() ){
    edm::LogError ("SUSY_HLT_Mu_Hadronic") << "invalid collection: Muons " << "\n";
    return;
  }
  

  //-------------------------------
  //--- Trigger
  //-------------------------------
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(triggerResults_,hltresults);
  if(!hltresults.isValid()){
    edm::LogError ("SUSY_HLT_Muon_Hadronic") << "invalid collection: TriggerResults" << "\n";
    return;
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()) {
    edm::LogError ("SUSY_HLT_Muon_Hadronic") << "invalid collection: TriggerSummary" << "\n";
    return;
  }


  //get online objects
  std::vector<Lepton> onlineMuons;
  size_t filterIndex = triggerSummary->filterIndex( triggerFilter_ );
  trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
  if( !(filterIndex >= triggerSummary->sizeFilters()) ){
    const trigger::Keys& keys = triggerSummary->filterKeys( filterIndex );
    for( size_t j = 0; j < keys.size(); ++j ){
      trigger::TriggerObject foundObject = triggerObjects[keys[j]];
      if(fabs(foundObject.id()) == 13){ //It's a muon
        h_triggerMuPt->Fill(foundObject.pt());
        h_triggerMuEta->Fill(foundObject.eta());
        h_triggerMuPhi->Fill(foundObject.phi());
        Lepton theMuon; theMuon.pt = foundObject.pt(); theMuon.phi = foundObject.phi(); theMuon.eta = foundObject.eta();
        onlineMuons.push_back(theMuon);
      }
    }
  }


  bool hasFired = false;
  bool hasFiredAuxiliaryForMuonLeg = false;
  bool hasFiredAuxiliaryForHadronicLeg = false;
  const edm::TriggerNames& trigNames = e.triggerNames(*hltresults);
  unsigned int numTriggers = trigNames.size();
  for( unsigned int hltIndex=0; hltIndex<numTriggers; ++hltIndex ){
    if (trigNames.triggerName(hltIndex).find(triggerPath_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFired = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForMuon_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFiredAuxiliaryForMuonLeg = true;
    if (trigNames.triggerName(hltIndex).find(triggerPathAuxiliaryForHadronic_) != std::string::npos && hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)) hasFiredAuxiliaryForHadronicLeg = true;
  }

  

  if(hasFiredAuxiliaryForMuonLeg || hasFiredAuxiliaryForHadronicLeg || !e.isRealData()) {
  
    std::vector<Lepton> offlineMuons;
    for(reco::MuonCollection::const_iterator muon = MuonCollection->begin(); muon != MuonCollection->end() ; ++muon) {
      if(fabs(muon->eta())>etaMuonOffline_) continue;
      Lepton theMuon; theMuon.pt = muon->pt(); theMuon.phi = muon->phi(); theMuon.eta = muon->eta();
      offlineMuons.push_back(theMuon);
    }

    std::sort(onlineMuons.begin(), onlineMuons.end(), comparePt);
    std::sort(offlineMuons.begin(), offlineMuons.end(), comparePt);
    /*
    std::cout << "OFFLINE MUON" << std::endl;
    for(std::vector<Lepton>::iterator offline_muon = offlineMuons.begin(); offline_muon != offlineMuons.end(); offline_muon++) {
      std::cout << offline_muon->pt << std::endl;
    }
    std::cout << "ONLINE MUON" << std::endl;
    for(std::vector<Lepton>::iterator online_muon = onlineMuons.begin();  online_muon != onlineMuons.end(); online_muon++) {
      std::cout << online_muon->pt << std::endl;
    }
    */ 
    //Matching the muon
    int indexOfMatchedMuon = -1;
    int offlineCounter = 0;
    for(std::vector<Lepton>::iterator offline_muon = offlineMuons.begin(); (offline_muon != offlineMuons.end() && indexOfMatchedMuon == -1); offline_muon++) {
      for(std::vector<Lepton>::iterator online_muon = onlineMuons.begin();  online_muon != onlineMuons.end(); online_muon++) {
        if(sqrt( (offline_muon->phi-online_muon->phi)*(offline_muon->phi-online_muon->phi) +
                 (offline_muon->eta-online_muon->eta)*(offline_muon->eta-online_muon->eta)) < 0.5) {
          indexOfMatchedMuon = offlineCounter;
          //std::cout << "Offline " << offline_muon->pt << " " << "Online " << online_muon->pt << std::endl;
          break;
        }
      }
      offlineCounter++;
    }
    float caloHT = 0.0;
    float pfHT = 0.0;
    for (reco::PFJetCollection::const_iterator i_pfjet = pfJetCollection->begin(); i_pfjet != pfJetCollection->end(); ++i_pfjet){
      if (i_pfjet->pt() < ptThrJet_) continue;
      if (fabs(i_pfjet->eta()) > etaThrJet_) continue;
      pfHT += i_pfjet->pt();
    }
    for (reco::CaloJetCollection::const_iterator i_calojet = caloJetCollection->begin(); i_calojet != caloJetCollection->end(); ++i_calojet){
      if (i_calojet->pt() < ptThrJet_) continue;
      if (fabs(i_calojet->eta()) > etaThrJet_) continue;
      caloHT += i_calojet->pt();
    }
    //Need to apply the MET offline cut to be in the MET plateau
    if((hasFiredAuxiliaryForMuonLeg || !e.isRealData()) && offlineMuons.size()>0 && pfMETCollection->begin()->et() > METOffline_ && pfHT > HTOffline_) {
      if(hasFired && indexOfMatchedMuon >= 0) {
        h_MuTurnOn_num-> Fill(offlineMuons.at(indexOfMatchedMuon).pt);
        h_MuTurnOn_den-> Fill(offlineMuons.at(indexOfMatchedMuon).pt);
      } else {
        h_MuTurnOn_den-> Fill(offlineMuons.at(0).pt);
      } 
    }
    //Need to apply the pt offline cut to be in the muon pt plateau
    if((hasFiredAuxiliaryForHadronicLeg || !e.isRealData()) && indexOfMatchedMuon >= 0 && offlineMuons.at(indexOfMatchedMuon).pt > ptMuonOffline_ && pfHT > HTOffline_) {
      if(hasFired) {
        h_pfMetTurnOn_num-> Fill(pfMETCollection->begin()->et());
      } 
      h_pfMetTurnOn_den-> Fill(pfMETCollection->begin()->et());
    }
    if((hasFiredAuxiliaryForHadronicLeg || !e.isRealData()) && indexOfMatchedMuon >= 0 && offlineMuons.at(indexOfMatchedMuon).pt > ptMuonOffline_ && pfMETCollection->begin()->et() > METOffline_) {
      if(hasFired) {
        h_pfHTTurnOn_num-> Fill(pfHT);
      } 
      h_pfHTTurnOn_den-> Fill(pfHT);
    }
  }
}


void SUSY_HLT_Muon_Hadronic::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "SUSY_HLT_Muon_Hadronic::endLuminosityBlock" << std::endl;
}


void SUSY_HLT_Muon_Hadronic::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  edm::LogInfo("SUSY_HLT_Muon_Hadronic") << "SUSY_HLT_Muon_Hadronic::endRun" << std::endl;
}

void SUSY_HLT_Muon_Hadronic::bookHistos(DQMStore::IBooker & ibooker_)
{
  ibooker_.cd();
  ibooker_.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  //offline quantities

  //online quantities 
  h_triggerMuPt = ibooker_.book1D("triggerMuPt", "Trigger Muon Pt; GeV", 20, 0.0, 500.0);
  h_triggerMuEta = ibooker_.book1D("triggerMuEta", "Trigger Muon Eta", 20, -3.0, 3.0);
  h_triggerMuPhi = ibooker_.book1D("triggerMuPhi", "Trigger Muon Phi", 20, -3.5, 3.5);

  //num and den hists to be divided in harvesting step to make turn on curves
  h_pfMetTurnOn_num = ibooker_.book1D("pfMetTurnOn_num", "PF MET Turn On Numerator", 20, 0.0, 500.0 );
  h_pfMetTurnOn_den = ibooker_.book1D("pfMetTurnOn_den", "PF MET Turn OnDenominator", 20, 0.0, 500.0 );
  h_pfHTTurnOn_num = ibooker_.book1D("pfHTTurnOn_num", "PF HT Turn On Numerator", 30, 0.0, 1500.0 );
  h_pfHTTurnOn_den = ibooker_.book1D("pfHTTurnOn_den", "PF HT Turn On Denominator", 30, 0.0, 1500.0 );
  h_MuTurnOn_num = ibooker_.book1D("MuTurnOn_num", "Muon Turn On Numerator", 30, 0.0, 150 );
  h_MuTurnOn_den = ibooker_.book1D("MuTurnOn_den", "Muon Turn On Denominator", 30, 0.0, 150.0 );

  ibooker_.cd();
}

 //define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_Muon_Hadronic);
