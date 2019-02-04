#include "HLTriggerOffline/SUSYBSM/interface/SUSY_HLT_SingleLepton.h"

#include <limits>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

namespace{
  bool Contains(const std::string &text,
		const std::string &pattern){
    return text.find(pattern)!=std::string::npos;
  }

  void SetBinLabels(MonitorElement * const me){
    if(!me) return;
    me->setBinLabel(1, "No CSV Reqs.");
    me->setBinLabel(2, "N_{CSVL} >= 1");
    me->setBinLabel(3, "N_{CSVL} >= 2");
    me->setBinLabel(4, "N_{CSVL} >= 3");
    me->setBinLabel(5, "N_{CSVL} >= 4");
    me->setBinLabel(6, "N_{CSVM} >= 1");
    me->setBinLabel(7, "N_{CSVM} >= 2");
    me->setBinLabel(8, "N_{CSVM} >= 3");
    me->setBinLabel(9, "N_{CSVM} >= 4");
    me->setBinLabel(10, "N_{CSVT} >= 1");
    me->setBinLabel(11, "N_{CSVT} >= 2");
    me->setBinLabel(12, "N_{CSVT} >= 3");
    me->setBinLabel(13, "N_{CSVT} >= 4");
  }

  bool IsGood(const reco::GsfElectron &el, const reco::Vertex::Point &pv_position,
              const reco::BeamSpot::Point &bs_position,
              const edm::Handle<reco::ConversionCollection> &convs){
    const float dEtaIn = el.deltaEtaSuperClusterTrackAtVtx();
    const float dPhiIn = el.deltaPhiSuperClusterTrackAtVtx();
    const float sigmaietaieta = el.full5x5_sigmaIetaIeta();
    const float hOverE = el.hcalOverEcal();
    float d0 = 0.0, dz=0.0;
    try{
      d0=-(el.gsfTrack()->dxy(pv_position));
      dz = el.gsfTrack()->dz(pv_position);
    }catch(...){
      edm::LogError("SUSY_HLT_SingleLepton") << "Could not read electron.gsfTrack().\n";
      return false;
    }
    float ooemoop = 1e30;
    if(el.ecalEnergy()>0.0 && std::isfinite(el.ecalEnergy())){
      ooemoop = fabs(1.0/el.ecalEnergy() - el.eSuperClusterOverP()/el.ecalEnergy());
    }
    const auto &iso = el.pfIsolationVariables();
    const float absiso = iso.sumChargedHadronPt
      + std::max(0.0, iso.sumNeutralHadronEt + iso.sumPhotonEt -0.5 * iso.sumPUPt);
    const float relisowithdb = absiso/el.pt();

    bool pass_conversion = false;
    if(convs.isValid()){
      try{
        pass_conversion = !ConversionTools::hasMatchedConversion(el, *convs, bs_position);
      }catch(...){
        edm::LogError("SUSY_HLT_SingleLepton") << "Electron conversion matching failed.\n";
        return false;
      }
    }

    float etasc = 0.0;
    try{
      etasc = el.superCluster()->eta();
    }catch(...){
      edm::LogError("SUSY_HLT_SingleLepton") << "Could not read electron.superCluster().\n";
      return false;
    }
    if(fabs(etasc)>2.5){
      return false;
    }else if(fabs(etasc)>1.479){
      if(fabs(dEtaIn)>0.0106) return false;
      if(fabs(dPhiIn)>0.0359) return false;
      if(sigmaietaieta>0.0305) return false;
      if(hOverE>0.0835) return false;
      if(fabs(d0)>0.0163) return false;
      if(fabs(dz)>0.5999) return false;
      if(fabs(ooemoop)>0.1126) return false;
      if(relisowithdb>0.2075) return false;
      if(!pass_conversion) return false;
    }else{
      if(fabs(dEtaIn)>0.0091) return false;
      if(fabs(dPhiIn)>0.031) return false;
      if(sigmaietaieta>0.0106) return false;
      if(hOverE>0.0532) return false;
      if(fabs(d0)>0.0126) return false;
      if(fabs(dz)>0.0116) return false;
      if(fabs(ooemoop)>0.0609) return false;
      if(relisowithdb>0.1649) return false;
      if(!pass_conversion) return false;
    }
    return true;
  }

  bool IsGood(const reco::Muon &mu, const reco::Vertex::Point &pv_position){
    if(!mu.isGlobalMuon()) return false;
    if(!mu.isPFMuon()) return false;
    try{
      if(mu.globalTrack()->normalizedChi2()>=10.) return false;
      if(mu.globalTrack()->hitPattern().numberOfValidMuonHits()<=0) return false;
    }catch(...){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Could not read muon.globalTrack().\n";
      return false;
    }
    if(mu.numberOfMatchedStations()<=1) return false;
    try{
      if(fabs(mu.muonBestTrack()->dxy(pv_position))>=0.2) return false;
      if(fabs(mu.muonBestTrack()->dz(pv_position))>=0.5) return false;
    }catch(...){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Could not read muon.muonBestTrack().\n";
      return false;
    }
    try{
      if(mu.innerTrack()->hitPattern().numberOfValidPixelHits()<=0) return false;
      if(mu.innerTrack()->hitPattern().trackerLayersWithMeasurement()<=5) return false;
    }catch(...){
      edm::LogWarning("SUSY_HLT_SingleLepton") << "Could not read muon.innerTrack().\n";
      return false;
    }
    return true;
  }
}

SUSY_HLT_SingleLepton::SUSY_HLT_SingleLepton(const edm::ParameterSet &ps):
  theElectronTag_(ps.getParameter<edm::InputTag>("electronCollection")),
  theElectronCollection_(consumes<reco::GsfElectronCollection>(theElectronTag_)),
  theMuonTag_(ps.getParameter<edm::InputTag>("muonCollection")),
  theMuonCollection_(consumes<reco::MuonCollection>(theMuonTag_)),
  thePfMETTag_(ps.getParameter<edm::InputTag>("pfMetCollection")),
  thePfMETCollection_(consumes<reco::PFMETCollection>(thePfMETTag_)),
  thePfJetTag_(ps.getParameter<edm::InputTag>("pfJetCollection")),
  thePfJetCollection_(consumes<reco::PFJetCollection>(thePfJetTag_)),
  theJetTagTag_(ps.getParameter<edm::InputTag>("jetTagCollection")),
  theJetTagCollection_(consumes<reco::JetTagCollection>(theJetTagTag_)),

  theVertexCollectionTag_(ps.getParameter<edm::InputTag>("vertexCollection")),
  theVertexCollection_(consumes<reco::VertexCollection>(theVertexCollectionTag_)),
  theConversionCollectionTag_(ps.getParameter<edm::InputTag>("conversionCollection")),
  theConversionCollection_(consumes<reco::ConversionCollection>(theConversionCollectionTag_)),
  theBeamSpotTag_(ps.getParameter<edm::InputTag>("beamSpot")),
  theBeamSpot_(consumes<reco::BeamSpot>(theBeamSpotTag_)),

  theLeptonFilterTag_(ps.getParameter<edm::InputTag>("leptonFilter")),
  theHLTHTTag_(ps.getParameter<edm::InputTag>("hltHt")),
  theHLTHT_(consumes<reco::METCollection>(theHLTHTTag_)),
  theHLTMETTag_(ps.getParameter<edm::InputTag>("hltMet")),
  theHLTMET_(consumes<reco::METCollection>(theHLTMETTag_)),
  theHLTJetCollectionTag_(ps.getParameter<edm::InputTag>("hltJets")),
  theHLTJetCollection_(consumes<reco::CaloJetCollection>(theHLTJetCollectionTag_)),
  theHLTJetTagCollectionTag_(ps.getParameter<edm::InputTag>("hltJetTags")),
  theHLTJetTagCollection_(consumes<reco::JetTagCollection>(theHLTJetTagCollectionTag_)),

  theTriggerResultsTag_(ps.getParameter<edm::InputTag>("triggerResults")),
  theTriggerResults_(consumes<edm::TriggerResults>(theTriggerResultsTag_)),
  theTrigSummaryTag_(ps.getParameter<edm::InputTag>("trigSummary")),
  theTrigSummary_(consumes<trigger::TriggerEvent>(theTrigSummaryTag_)),

  fHltConfig_(),

  HLTProcess_(ps.getParameter<std::string>("hltProcess")),

  triggerPath_(ps.getParameter<std::string>("triggerPath")),
  triggerPathAuxiliary_(ps.getParameter<std::string>("triggerPathAuxiliary")),
  triggerPathLeptonAuxiliary_(ps.getParameter<std::string>("triggerPathLeptonAuxiliary")),

  csvlCut_(ps.getUntrackedParameter<double>("csvlCut")),
  csvmCut_(ps.getUntrackedParameter<double>("csvmCut")),
  csvtCut_(ps.getUntrackedParameter<double>("csvtCut")),

  jetPtCut_(ps.getUntrackedParameter<double>("jetPtCut")),
  jetEtaCut_(ps.getUntrackedParameter<double>("jetEtaCut")),
  metCut_(ps.getUntrackedParameter<double>("metCut")),
  htCut_(ps.getUntrackedParameter<double>("htCut")),

  lep_pt_threshold_(ps.getUntrackedParameter<double>("leptonPtThreshold")),
  ht_threshold_(ps.getUntrackedParameter<double>("htThreshold")),
  met_threshold_(ps.getUntrackedParameter<double>("metThreshold")),
  csv_threshold_(ps.getUntrackedParameter<double>("csvThreshold")),

  h_triggerLepPt_(nullptr),
  h_triggerLepEta_(nullptr),
  h_triggerLepPhi_(nullptr),
  h_HT_(nullptr),
  h_MET_(nullptr),
  h_maxCSV_(nullptr),
  h_leptonTurnOn_num_(nullptr),
  h_leptonTurnOn_den_(nullptr),
  h_pfHTTurnOn_num_(nullptr),
  h_pfHTTurnOn_den_(nullptr),
  h_pfMetTurnOn_num_(nullptr),
  h_pfMetTurnOn_den_(nullptr),
  h_CSVTurnOn_num_(nullptr),
  h_CSVTurnOn_den_(nullptr),
  h_btagTurnOn_num_(nullptr),
  h_btagTurnOn_den_(nullptr){
  edm::LogInfo("SUSY_HLT_SingleLepton")
    << "Constructor SUSY_HLT_SingleLepton::SUSY_HLT_SingleLepton\n";
  }

SUSY_HLT_SingleLepton::~SUSY_HLT_SingleLepton(){
  edm::LogInfo("SUSY_HLT_SingleLepton")
    << "Destructor SUSY_HLT_SingleLepton::~SUSY_HLT_SingleLepton\n";
}

void SUSY_HLT_SingleLepton::dqmBeginRun(const edm::Run &run, const edm::EventSetup &e){
  bool changed;

  if(!fHltConfig_.init(run, e, HLTProcess_, changed)){
    edm::LogError("SUSY_HLT_SingleLepton")
      << "Initialization of HLTConfigProvider failed!!\n";
    return;
  }

  bool pathFound = false;
  for(const auto &trig_name: fHltConfig_.triggerNames()){
    if(Contains(trig_name, triggerPath_)) pathFound = true;
  }

  if(!pathFound){
    LogDebug("SUSY_HLT_SingleLepton") << "Path not found: " << triggerPath_ << '\n';
    return;
  }

  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::beginRun\n";
}

void SUSY_HLT_SingleLepton::bookHistograms(DQMStore::IBooker &ibooker,
                                           const edm::Run &, const edm::EventSetup &){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::bookHistograms\n";
  //book at beginRun
  ibooker.cd();
  ibooker.setCurrentFolder("HLT/SUSYBSM/" + triggerPath_);

  bool is_mu = false;
  bool is_ele = false;
  if(theElectronTag_.label().empty() && !theMuonTag_.label().empty()){
    is_mu=true;
  }else if(!theElectronTag_.label().empty() && theMuonTag_.label().empty()){
    is_ele=true;
  }
  std::string lepton="lepton", Lepton="Lepton";
  if(is_mu && !is_ele){
    lepton="muon";
    Lepton="Muon";
  }else if(is_ele && !is_mu){
    lepton="electron";
    Lepton="Electron";
  }

  //online quantities
  h_triggerLepPt_ = ibooker.book1D("triggerLepPt",
                                   (";"+Lepton+" p_{T} [GeV];").c_str(),
                                   20, 0.0, 500.0);
  h_triggerLepEta_ = ibooker.book1D("triggerLepEta",
                                    (";"+Lepton+" #eta;").c_str(),
                                    20, -3.0, 3.0);
  h_triggerLepPhi_ = ibooker.book1D("triggerLepPhi",
                                    (";"+Lepton+" #phi;").c_str(),
                                    20, -3.5, 3.5);

  if(!theHLTHTTag_.label().empty()){
    h_HT_ = ibooker.book1D("HT",
                           ";HLT HT [GeV];",
                           40, 0.0, 1000.0);
  }

  if(!theHLTMETTag_.label().empty()){
    h_MET_ = ibooker.book1D("MET",
                            ";HLT MET [GeV];",
                            40, 0.0, 1000.0);
  }

  if(!theHLTJetCollectionTag_.label().empty() && !theHLTJetTagCollectionTag_.label().empty()){
    h_maxCSV_ = ibooker.book1D("maxCSV",
                               ";Max HLT CSV;",
                               20, 0.0, 1.0);
  }

  //num and den hists to be divided in harvesting step to make turn on curves
  h_leptonTurnOn_num_ = ibooker.book1D("leptonTurnOn_num",
                                       ("Numerator;Offline "+lepton+" p_{T} [GeV];").c_str(),
                                       30, 0.0, 150);
  h_leptonTurnOn_den_ = ibooker.book1D("leptonTurnOn_den",
                                       ("Denominator;Offline "+lepton+" p_{T} [GeV];").c_str(),
                                       30, 0.0, 150.0);
  h_pfHTTurnOn_num_ = ibooker.book1D("pfHTTurnOn_num",
                                     "Numerator;Offline H_{T} [GeV];",
                                     30, 0.0, 1500.0 );
  h_pfHTTurnOn_den_ = ibooker.book1D("pfHTTurnOn_den",
                                     "Denominator;Offline H_{T} [GeV];",
                                     30, 0.0, 1500.0 );

  if(!theHLTMETTag_.label().empty()){
    h_pfMetTurnOn_num_ = ibooker.book1D("pfMetTurnOn_num",
                                        "Numerator;Offline MET [GeV];",
                                        20, 0.0, 500.0 );
    h_pfMetTurnOn_den_ = ibooker.book1D("pfMetTurnOn_den",
                                        "Denominator;Offline MET [GeV];",
                                        20, 0.0, 500.0 );
  }

  if(!theHLTJetCollectionTag_.label().empty() && !theHLTJetTagCollectionTag_.label().empty()){
    h_CSVTurnOn_num_ = ibooker.book1D("CSVTurnOn_num",
                                      "Numerator;Offline Max CSV Discriminant;",
                                      20, 0.0, 1.0);
    h_CSVTurnOn_den_ = ibooker.book1D("CSVTurnOn_den",
                                      "Denominator;Offline Max CSV Discriminant;",
                                      20, 0.0, 1.0);

    h_btagTurnOn_num_ = ibooker.book1D("btagTurnOn_num",
                                       "Numerator;Offline CSV Requirement;",
                                       13, -0.5, 12.5);
    h_btagTurnOn_den_ = ibooker.book1D("btagTurnOn_den",
                                       "Denominator;Offline CSV Requirements;",
                                       13, -0.5, 12.5);

    SetBinLabels(h_btagTurnOn_num_);
    SetBinLabels(h_btagTurnOn_den_);
  }
  ibooker.cd();
}


void SUSY_HLT_SingleLepton::analyze(const edm::Event &e, const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::analyze\n";


  //HLT HT
  edm::Handle<reco::METCollection> HLTHT;
  if(!theHLTHTTag_.label().empty()){
    e.getByToken(theHLTHT_, HLTHT);
    if( !HLTHT.isValid() ){
      edm::LogInfo("SUSY_HLT_SingleLepton")
        << "Invalid METCollection: " << theHLTHTTag_.label() << '\n';
    }
  }

  //HLT MET
  edm::Handle<reco::METCollection> HLTMET;
  if(!theHLTMETTag_.label().empty()){
    e.getByToken(theHLTMET_, HLTMET);
    if( !HLTMET.isValid() ){
      edm::LogInfo("SUSY_HLT_SingleLepton")
        << "Invalid METCollection: " << theHLTMETTag_.label() << '\n';
    }
  }

  //HLT Jets
  edm::Handle<reco::CaloJetCollection> HLTJetCollection;
  if(!theHLTJetCollectionTag_.label().empty()){
    e.getByToken(theHLTJetCollection_, HLTJetCollection);
    if( !HLTJetCollection.isValid() ){
      edm::LogInfo("SUSY_HLT_SingleLepton")
        << "Invalid CaloJetCollection: " << theHLTJetCollectionTag_.label() << '\n';
    }
  }

  //HLT Jet Tags
  edm::Handle<reco::JetTagCollection> HLTJetTagCollection;
  if(!theHLTJetTagCollectionTag_.label().empty()){
    e.getByToken(theHLTJetTagCollection_, HLTJetTagCollection);
    if( !HLTJetTagCollection.isValid() ){
      edm::LogInfo("SUSY_HLT_SingleLepton")
        << "Invalid JetTagCollection: " << theHLTJetTagCollectionTag_.label() << '\n';
    }
  }

  //Vertex
  edm::Handle<reco::VertexCollection> VertexCollection;
  if(!theVertexCollectionTag_.label().empty()){
    e.getByToken(theVertexCollection_, VertexCollection);
    if( !VertexCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid VertexCollection: " << theVertexCollectionTag_.label() << '\n';
    }
  }

  //Conversions
  edm::Handle<reco::ConversionCollection> ConversionCollection;
  if(!theConversionCollectionTag_.label().empty()){
    e.getByToken(theConversionCollection_, ConversionCollection);
    if( !ConversionCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid ConversionCollection: " << theConversionCollectionTag_.label() << '\n';
    }
  }

  //Beam Spot
  edm::Handle<reco::BeamSpot> BeamSpot;
  if(!theBeamSpotTag_.label().empty()){
    e.getByToken(theBeamSpot_, BeamSpot);
    if( !BeamSpot.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid BeamSpot: " << theBeamSpotTag_.label() << '\n';
    }
  }

  //MET
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  if(!thePfMETTag_.label().empty()){
    e.getByToken(thePfMETCollection_, pfMETCollection);
    if( !pfMETCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid PFMETCollection: " << thePfMETTag_.label() << '\n';
    }
  }

  //Jets
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  if(!thePfJetTag_.label().empty()){
    e.getByToken (thePfJetCollection_,pfJetCollection);
    if( !pfJetCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid PFJetCollection: " << thePfJetTag_.label() << '\n';
    }
  }

  //b-tags
  edm::Handle<reco::JetTagCollection> jetTagCollection;
  if(!theJetTagTag_.label().empty()){
    e.getByToken(theJetTagCollection_, jetTagCollection);
    if( !jetTagCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid JetTagCollection: " << theJetTagTag_.label() << '\n';
    }
  }

  //Electron
  edm::Handle<reco::GsfElectronCollection> ElectronCollection;
  if(!theElectronTag_.label().empty()){
    e.getByToken (theElectronCollection_, ElectronCollection);
    if( !ElectronCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid GsfElectronCollection: " << theElectronTag_.label() << '\n';
    }
  }

  //Muon
  edm::Handle<reco::MuonCollection> MuonCollection;
  if(!theMuonTag_.label().empty()){
    e.getByToken (theMuonCollection_, MuonCollection);
    if( !MuonCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid MuonCollection: " << theMuonTag_.label() << '\n';
    }
  }

  //Trigger
  edm::Handle<edm::TriggerResults> hltresults;
  if(!theTriggerResultsTag_.label().empty()){
    e.getByToken(theTriggerResults_, hltresults);
    if( !hltresults.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid TriggerResults: " << theTriggerResultsTag_.label() << '\n';
    }
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  if(!theTrigSummaryTag_.label().empty()){
    e.getByToken(theTrigSummary_, triggerSummary);
    if( !triggerSummary.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
        << "Invalid TriggerEvent: " << theTrigSummaryTag_.label() << '\n';
    }
  }

  //Get online leptons
  std::vector<float> ptLepton, etaLepton, phiLepton;
  if(triggerSummary.isValid()){
    //Leptons
    size_t filterIndex = triggerSummary->filterIndex(theLeptonFilterTag_);
    trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
    if( !(filterIndex >= triggerSummary->sizeFilters()) ){
      size_t ilep = 0, num_keys = triggerSummary->filterKeys(filterIndex).size();
      ptLepton.resize(num_keys);
      etaLepton.resize(num_keys);
      phiLepton.resize(num_keys);
      for(const auto &key: triggerSummary->filterKeys(filterIndex)){
        const trigger::TriggerObject &foundObject = triggerObjects[key];

        if(h_triggerLepPt_) h_triggerLepPt_->Fill(foundObject.pt());
        if(h_triggerLepEta_) h_triggerLepEta_->Fill(foundObject.eta());
        if(h_triggerLepPhi_) h_triggerLepPhi_->Fill(foundObject.phi());

        ptLepton.at(ilep)=foundObject.pt();
        etaLepton.at(ilep)=foundObject.eta();
        phiLepton.at(ilep)=foundObject.phi();
	++ilep;
      }
    }
  }

  //Get online ht and met
  const float hlt_ht = ((HLTHT.isValid() && !HLTHT->empty())?HLTHT->front().sumEt():-1.0);
  if(h_HT_) h_HT_->Fill(hlt_ht);
  const float hlt_met = ((HLTMET.isValid() && !HLTMET->empty())?HLTMET->front().pt():-1.0);
  if(h_MET_) h_MET_->Fill(hlt_met);

  //Get online csv and fill plot
  float hlt_csv = -1.0;
  if(HLTJetCollection.isValid() && HLTJetTagCollection.isValid()){
    for(const auto &jet: *HLTJetTagCollection){
      if(jet.second>hlt_csv) hlt_csv = jet.second;
    }
  }
  if(h_maxCSV_) h_maxCSV_->Fill(hlt_csv);

  //Test whether main and auxilliary triggers fired
  bool hasFired = false;
  bool hasFiredAuxiliary = false;
  bool hasFiredLeptonAuxiliary = false;
  if(hltresults.isValid()){
    const edm::TriggerNames &trigNames = e.triggerNames(*hltresults);
    for( unsigned int hltIndex = 0; hltIndex < trigNames.size(); ++hltIndex ){
      if(hltresults->wasrun(hltIndex) && hltresults->accept(hltIndex)){
        const std::string& name = trigNames.triggerName(hltIndex);
        if(Contains(name, triggerPath_)) hasFired=true;
        if(Contains(name, triggerPathAuxiliary_)) hasFiredAuxiliary=true;
	if(Contains(name, triggerPathLeptonAuxiliary_)) hasFiredLeptonAuxiliary=true;
      }
    }
  }

  //Get offline HT
  double pfHT = -1.0;
  if(pfJetCollection.isValid()){
    pfHT=0.0;
    for(const auto &pfjet: *pfJetCollection){
      if(pfjet.pt() < jetPtCut_) continue;
      if(fabs(pfjet.eta()) > jetEtaCut_) continue;
      pfHT += pfjet.pt();
    }
  }

  //Get offline MET
  double pfMET = -1.0;
  if(pfMETCollection.isValid() && !pfMETCollection->empty()){
    pfMET = pfMETCollection->front().et();
  }

  //Get offline b-tagging info
  float maxCSV = -1.0;
  unsigned num_csvl = 0;
  unsigned num_csvm = 0;
  unsigned num_csvt = 0;
  if(jetTagCollection.isValid()){
    for(const auto &jet: *jetTagCollection){
      const float CSV = jet.second;
      if(jet.first->pt()>jetPtCut_){
        if(CSV>maxCSV){
          maxCSV=CSV;
        }
        if(CSV>csvlCut_){
          ++num_csvl;
          if(CSV>csvmCut_){
            ++num_csvm;
            if(CSV>csvtCut_){
              ++num_csvt;
            }
          }
        }
      }
    }
  }
  if(h_maxCSV_) h_maxCSV_->Fill(maxCSV);

  //Fill lepton pt efficiency plot
  double lep_max_pt = -1.0;
  if(VertexCollection.isValid() && !VertexCollection->empty()){//for quality checks
    //Try to find a reco electron
    if(ElectronCollection.isValid()
       && ConversionCollection.isValid()
       && BeamSpot.isValid()){
      for(const auto &electron: *ElectronCollection){
        if(IsGood(electron, VertexCollection->front().position(),
                  BeamSpot->position(), ConversionCollection)){
          if(electron.pt()>lep_max_pt) lep_max_pt=electron.pt();
        }
      }
    }

    //Try to find a reco muon
    if(MuonCollection.isValid()){
      for(const auto &muon: *MuonCollection){
        if(IsGood(muon, VertexCollection->front().position())){
          if(muon.pt()>lep_max_pt){
            lep_max_pt=muon.pt();
          }
        }
      }
    }
  }

  const bool lep_plateau = lep_max_pt>lep_pt_threshold_ || lep_pt_threshold_<0.0;
  const bool ht_plateau = pfHT>ht_threshold_ || ht_threshold_<0.0;
  const bool met_plateau = pfMET>met_threshold_ || met_threshold_<0.0;
  const bool csv_plateau = maxCSV>csv_threshold_ || csv_threshold_<0.0;

  //Fill lepton turn-on histograms
  if(hasFiredLeptonAuxiliary || triggerPathLeptonAuxiliary_.empty() || !e.isRealData()){
    //Fill histograms using highest pt reco lepton
    if(ht_plateau && met_plateau && csv_plateau
       && (pfMET>metCut_ || metCut_<0.0)
       && (pfHT>htCut_ || htCut_<0.0)){
      if(h_leptonTurnOn_den_) h_leptonTurnOn_den_->Fill(lep_max_pt);
      if(h_leptonTurnOn_num_ && hasFired) h_leptonTurnOn_num_->Fill(lep_max_pt);
    }
  }

  //Fill remaining turn-on histograms
  if(hasFiredAuxiliary || triggerPathAuxiliary_.empty() || !e.isRealData()){
    //Fill HT efficiency plot
    if(lep_plateau && met_plateau && csv_plateau){
      if(h_pfHTTurnOn_den_) h_pfHTTurnOn_den_->Fill(pfHT);
      if(h_pfHTTurnOn_num_ && hasFired) h_pfHTTurnOn_num_->Fill(pfHT);
    }

    //Fill MET efficiency plot
    if(lep_plateau && ht_plateau && csv_plateau){
      if(h_pfMetTurnOn_den_) h_pfMetTurnOn_den_->Fill(pfMET);
      if(h_pfMetTurnOn_num_ && hasFired) h_pfMetTurnOn_num_->Fill(pfMET);
    }

    //Fill CSV efficiency plot
    if(lep_plateau && ht_plateau && met_plateau){
      if(h_CSVTurnOn_den_) h_CSVTurnOn_den_->Fill(maxCSV);
      if(h_CSVTurnOn_num_ && hasFired) h_CSVTurnOn_num_->Fill(maxCSV);

      if(h_btagTurnOn_den_){
        switch(num_csvl){
        default: h_btagTurnOn_den_->Fill(4);
        case 3 : h_btagTurnOn_den_->Fill(3);
        case 2 : h_btagTurnOn_den_->Fill(2);
        case 1 : h_btagTurnOn_den_->Fill(1);
        case 0 : h_btagTurnOn_den_->Fill(0);
        }
        switch(num_csvm){
        default: h_btagTurnOn_den_->Fill(8);
        case 3 : h_btagTurnOn_den_->Fill(7);
        case 2 : h_btagTurnOn_den_->Fill(6);
        case 1 : h_btagTurnOn_den_->Fill(5);
        case 0 : break;//Don't double count in the no tag bin
        }
        switch(num_csvt){
        default: h_btagTurnOn_den_->Fill(12);
        case 3 : h_btagTurnOn_den_->Fill(11);
        case 2 : h_btagTurnOn_den_->Fill(10);
        case 1 : h_btagTurnOn_den_->Fill(9);
        case 0 : break;//Don't double count in the no tag bin
        }
      }
      if(h_btagTurnOn_num_ && hasFired){
        switch(num_csvl){
        default: h_btagTurnOn_num_->Fill(4);
        case 3 : h_btagTurnOn_num_->Fill(3);
        case 2 : h_btagTurnOn_num_->Fill(2);
        case 1 : h_btagTurnOn_num_->Fill(1);
        case 0 : h_btagTurnOn_num_->Fill(0);
        }
        switch(num_csvm){
        default: h_btagTurnOn_num_->Fill(8);
        case 3 : h_btagTurnOn_num_->Fill(7);
        case 2 : h_btagTurnOn_num_->Fill(6);
        case 1 : h_btagTurnOn_num_->Fill(5);
        case 0 : break;//Don't double count in the no tag bin
        }
        switch(num_csvt){
        default: h_btagTurnOn_num_->Fill(12);
        case 3 : h_btagTurnOn_num_->Fill(11);
        case 2 : h_btagTurnOn_num_->Fill(10);
        case 1 : h_btagTurnOn_num_->Fill(9);
        case 0 : break;//Don't double count in the no tag bin
        }
      }
    }
  }
}


void SUSY_HLT_SingleLepton::endRun(const edm::Run &run, const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::endRun\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_SingleLepton);
