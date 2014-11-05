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

  double GetMass(const double E, const double px, const double py, const double pz){
    const double rx=px/E;
    const double ry=py/E;
    const double rz=pz/E;
    return E*sqrt(1.0-rx*rx-ry*ry-rz*rz);
  }

  bool IsGood(const reco::GsfElectron &el, const reco::Vertex::Point &pv_position,
	      const reco::BeamSpot::Point &bs_position,
	      const edm::Handle<reco::ConversionCollection> &convs){
    const float dEtaIn = el.deltaEtaSuperClusterTrackAtVtx();
    const float dPhiIn = el.deltaPhiSuperClusterTrackAtVtx();
    const float sigmaietaieta = el.full5x5_sigmaIetaIeta();
    const float hOverE = el.hcalOverEcal();
    const float d0 = -1.0*el.gsfTrack()->dxy(pv_position);
    const float dz = el.gsfTrack()->dz(pv_position);
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
      pass_conversion = !ConversionTools::hasMatchedConversion(el, convs, bs_position);
    }

    const float etasc = el.superCluster()->eta();
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
    if(mu.globalTrack()->normalizedChi2()>=10.) return false;
    if(mu.globalTrack()->hitPattern().numberOfValidMuonHits()<=0) return false;
    if(mu.numberOfMatchedStations()<=1) return false;
    if(fabs(mu.muonBestTrack()->dxy(pv_position))>=0.2) return false;
    if(fabs(mu.muonBestTrack()->dz(pv_position))>=0.5) return false;
    if(mu.innerTrack()->hitPattern().numberOfValidPixelHits()<=0) return false;
    if(mu.innerTrack()->hitPattern().trackerLayersWithMeasurement()<=5) return false;
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
  theHLTHT_(consumes<reco::MET>(theHLTHTTag_)),
  theHLTMETTag_(ps.getParameter<edm::InputTag>("hltMet")),
  theHLTMET_(consumes<reco::MET>(theHLTMETTag_)),
  theHLTJetCollectionTag_(ps.getParameter<edm::InputTag>("hltJets")),
  theHLTJetCollection_(consumes<reco::CaloJetCollection>(theHLTJetCollectionTag_)),
  theHLTJetTagCollectionTag_(ps.getParameter<edm::InputTag>("hltJetTags")),
  theHLTJetTagCollection_(consumes<reco::JetTagCollection>(theHLTJetTagCollectionTag_)),

  theTriggerResultsTag_(ps.getParameter<edm::InputTag>("TriggerResults")),
  theTriggerResults_(consumes<edm::TriggerResults>(theTriggerResultsTag_)),
  theTrigSummaryTag_(ps.getParameter<edm::InputTag>("trigSummary")),
  theTrigSummary_(consumes<trigger::TriggerEvent>(theTrigSummaryTag_)),

  fHltConfig_(),

  HLTProcess_(ps.getParameter<std::string>("HLTProcess")),

  triggerPath_(ps.getParameter<std::string>("TriggerPath")),
  triggerPathAuxiliary_(ps.getParameter<std::string>("TriggerPathAuxiliary")),

  jetPtCut_(ps.getUntrackedParameter<double>("JetPtCut")),
  jetEtaCut_(ps.getUntrackedParameter<double>("JetEtaCut")),
  metCut_(ps.getUntrackedParameter<double>("MetCut")),
  
  lep_pt_threshold_(ps.getUntrackedParameter<double>("LeptonPtThreshold")),
  ht_threshold_(ps.getUntrackedParameter<double>("HtThreshold")),
  met_threshold_(ps.getUntrackedParameter<double>("MetThreshold")),
  csv_threshold_(ps.getUntrackedParameter<double>("CSVThreshold")),

  h_triggerLepPt_(nullptr),
  h_triggerLepEta_(nullptr),
  h_triggerLepPhi_(nullptr),
  h_CSVTurnOn_num_(nullptr),
  h_CSVTurnOn_den_(nullptr),
  h_pfMetTurnOn_num_(nullptr),
  h_pfMetTurnOn_den_(nullptr),
  h_pfHTTurnOn_num_(nullptr),
  h_pfHTTurnOn_den_(nullptr),
  h_leptonPtTurnOn_num_(nullptr),
  h_leptonPtTurnOn_den_(nullptr),
  h_leptonIsoTurnOn_num_(nullptr),
  h_leptonIsoTurnOn_den_(nullptr){
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
    if(trig_name.find(triggerPath_) != std::string::npos) pathFound = true;
  }

  if(!pathFound){
    edm::LogError ("SUSY_HLT_SingleLepton") << "Path not found: " << triggerPath_ << '\n';
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
  if(theElectronTag_.label() == "" && theMuonTag_.label() != ""){
    is_mu=true;
  }else if(theElectronTag_.label() != "" && theMuonTag_.label() == ""){
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
				   (";"+Lepton+" p_{T} [GeV]").c_str(),
				   20, 0.0, 500.0);
  h_triggerLepEta_ = ibooker.book1D("triggerLepEta",
				    (";"+Lepton+" #eta").c_str(),
				    20, -3.0, 3.0);
  h_triggerLepPhi_ = ibooker.book1D("triggerLepPhi",
				    (";"+Lepton+" #phi").c_str(),
				    20, -3.5, 3.5);

  //num and den hists to be divided in harvesting step to make turn on curves
  h_leptonPtTurnOn_num_ = ibooker.book1D("leptonPtTurnOn_num",
				       ("Numerator;Offline "+lepton+" p_{T} [GeV]").c_str(),
				       30, 0.0, 150);
  h_leptonPtTurnOn_den_ = ibooker.book1D("leptonPtTurnOn_den",
				       ("Denominator;Offline "+lepton+" p_{T} [GeV]").c_str(),
				       30, 0.0, 150.0);
  h_leptonIsoTurnOn_num_ = ibooker.book1D("leptonIsoTurnOn_num",
				       ("Numerator;Offline "+lepton+" rel. iso.").c_str(),
				       30, 0.0, 3.0);
  h_leptonIsoTurnOn_den_ = ibooker.book1D("leptonIsoTurnOn_den",
				       ("Denominator;Offline "+lepton+" rel. iso.").c_str(),
				       30, 0.0, 3.0);
  h_pfHTTurnOn_num_ = ibooker.book1D("pfHTTurnOn_num",
				     "Numerator;Offline H_{T} [GeV]",
				     30, 0.0, 1500.0 );
  h_pfHTTurnOn_den_ = ibooker.book1D("pfHTTurnOn_den",
				     "Denominator;Offline H_{T} [GeV]",
				     30, 0.0, 1500.0 );

  if(theHLTMETTag_.label()!=""){
    h_pfMetTurnOn_num_ = ibooker.book1D("pfMetTurnOn_num",
					"Numerator;Offline MET [GeV]",
					20, 0.0, 500.0 );
    h_pfMetTurnOn_den_ = ibooker.book1D("pfMetTurnOn_den",
					"Denominator;Offline MET [GeV]",
					20, 0.0, 500.0 );
  }

  if(theHLTJetCollectionTag_.label()!="" && theHLTJetTagCollectionTag_.label()!=""){
    h_CSVTurnOn_num_ = ibooker.book1D("CSVTurnOn_num",
				      "Numerator;Offline CSV Requirements",
				      13, -0.5, 12.5);
    h_CSVTurnOn_den_ = ibooker.book1D("CSVTurnOn_den",
				      "Denominator;Offline CSV Requirements",
				      13, -0.5, 12.5);

    SetBinLabels(h_CSVTurnOn_num_);
    SetBinLabels(h_CSVTurnOn_den_);
  }
  ibooker.cd();
}

void SUSY_HLT_SingleLepton::beginLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
						 const edm::EventSetup &context){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::beginLuminosityBlock\n";
}

void SUSY_HLT_SingleLepton::analyze(const edm::Event &e, const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::analyze\n";

  //HLT HT
  edm::Handle<reco::METCollection> HLTHT;
  if(theHLTHTTag_.label() != ""){
    e.getByToken(theHLTHT_, HLTHT);
    if( !HLTHT.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid METCollection: " << theHLTHTTag_.label() << '\n';
    }
  }

  //HLT MET
  edm::Handle<reco::METCollection> HLTMET;
  if(theHLTMETTag_.label() != ""){
    e.getByToken(theHLTMET_, HLTMET);
    if( !HLTMET.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid METCollection: " << theHLTMETTag_.label() << '\n';
    }
  }

  //HLT Jets
  edm::Handle<reco::CaloJetCollection> HLTJetCollection;
  if(theHLTJetCollectionTag_.label() != ""){
    e.getByToken(theHLTJetCollection_, HLTJetCollection);
    if( !HLTJetCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid CaloJetCollection: " << theHLTJetCollectionTag_.label() << '\n';
    }
  }

  //HLT Jet Tags
  edm::Handle<reco::JetTagCollection> HLTJetTagCollection;
  if(theHLTJetTagCollectionTag_.label() != ""){
    e.getByToken(theHLTJetTagCollection_, HLTJetTagCollection);
    if( !HLTJetTagCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid JetTagCollection: " << theHLTJetTagCollectionTag_.label() << '\n';
    }
  }

  //Vertex
  edm::Handle<reco::VertexCollection> VertexCollection;
  if(theVertexCollectionTag_.label() != ""){
    e.getByToken(theVertexCollection_, VertexCollection);
    if( !VertexCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid VertexCollection: " << theVertexCollectionTag_.label() << '\n';
    }
  }

  //Conversions
  edm::Handle<reco::ConversionCollection> ConversionCollection;
  if(theConversionCollectionTag_.label() != ""){
    e.getByToken(theConversionCollection_, ConversionCollection);
    if( !ConversionCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid ConversionCollection: " << theConversionCollectionTag_.label() << '\n';
    }
  }

  //Beam Spot
  edm::Handle<reco::BeamSpot> BeamSpot;
  if(theBeamSpotTag_.label() != ""){
    e.getByToken(theBeamSpot_, BeamSpot);
    if( !BeamSpot.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid BeamSpot: " << theBeamSpotTag_.label() << '\n';
    }
  }

  //MET
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  if(thePfMETTag_.label() != ""){
    e.getByToken(thePfMETCollection_, pfMETCollection);
    if( !pfMETCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid PFMETCollection: " << thePfMETTag_.label() << '\n';
    }
  }

  //Jets
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  if(thePfJetTag_.label() != ""){
    e.getByToken (thePfJetCollection_,pfJetCollection);
    if( !pfJetCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid PFJetCollection: " << thePfJetTag_.label() << '\n';
    }
  }

  //b-tags
  edm::Handle<reco::JetTagCollection> jetTagCollection;
  if(theJetTagTag_.label() != ""){
    e.getByToken(theJetTagCollection_, jetTagCollection);
    if( !jetTagCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid JetTagCollection: " << theJetTagTag_.label() << '\n';
    }
  }
  
  //Electron
  edm::Handle<reco::GsfElectronCollection> ElectronCollection;
  if(theElectronTag_.label() != ""){
    e.getByToken (theElectronCollection_, ElectronCollection);
    if( !ElectronCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid GsfElectronCollection: " << theElectronTag_.label() << '\n';
    }
  }
  
  //Muon
  edm::Handle<reco::MuonCollection> MuonCollection;
  if(theMuonTag_.label() != ""){ 
    e.getByToken (theMuonCollection_, MuonCollection);
    if( !MuonCollection.isValid() ){
      edm::LogWarning("SUSY_HLT_SingleLepton")
	<< "Invalid MuonCollection: " << theMuonTag_.label() << '\n';
    }
  }
  
  //Trigger
  edm::Handle<edm::TriggerResults> hltresults;
  e.getByToken(theTriggerResults_, hltresults);
  if(!hltresults.isValid()){
    edm::LogWarning("SUSY_HLT_SingleLepton")
      << "Invalid TriggerResults: " << theTriggerResultsTag_.label() << '\n';
  }
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  e.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()){
    edm::LogWarning("SUSY_HLT_SingleLepton")
      << "Invalid TriggerEvent: " << theTrigSummaryTag_.label() << '\n';
  }

  //Get online leptons
  std::vector<float> ptLepton, etaLepton, phiLepton;
  if(triggerSummary.isValid()){
    //Leptons
    size_t filterIndex = triggerSummary->filterIndex(theLeptonFilterTag_);
    trigger::TriggerObjectCollection triggerObjects = triggerSummary->getObjects();
    if( !(filterIndex >= triggerSummary->sizeFilters()) ){
      for(const auto &key: triggerSummary->filterKeys(filterIndex)){
	trigger::TriggerObject foundObject = triggerObjects[key];
	
	h_triggerLepPt_->Fill(foundObject.pt());
	h_triggerLepEta_->Fill(foundObject.eta());
	h_triggerLepPhi_->Fill(foundObject.phi());
	
	ptLepton.push_back(foundObject.pt());
	etaLepton.push_back(foundObject.eta());
	phiLepton.push_back(foundObject.phi());
      }
    }
  }
  const float hlt_lep_pt = *std::max_element(ptLepton.begin(), ptLepton.end());

  //Get online ht and met
  const float hlt_ht = (HLTHT.isValid()?HLTHT->front().sumEt():0.0);
  const float hlt_met = (HLTMET.isValid()?HLTMET->front().sumEt():0.0);

  //Get online csv
  float hlt_csv = -1.0;
  if(HLTJetCollection.isValid() && HLTJetTagCollection.isValid()){
    hlt_csv=0.0;
    for(const auto &jet: *HLTJetTagCollection){
      if(jet.second>hlt_csv) hlt_csv = jet.second;
    }
  }

  //Test whether main and auxilliary triggers fired
  bool hasFired = false;
  bool hasFiredAuxiliary = false;
  if(hltresults.isValid()){
    const edm::TriggerNames &trigNames = e.triggerNames(*hltresults);
    for( unsigned int hltIndex = 0; hltIndex < trigNames.size(); ++hltIndex ){
      if(trigNames.triggerName(hltIndex)==triggerPath_
	  && hltresults->wasrun(hltIndex)
	  && hltresults->accept(hltIndex)){
	hasFired = true;
      }

      if(trigNames.triggerName(hltIndex)==triggerPathAuxiliary_
	  && hltresults->wasrun(hltIndex)
	  && hltresults->accept(hltIndex)){
	hasFiredAuxiliary = true;
      }
    }
  }

  //Fill DQM plots if event is of interest
  if(hasFiredAuxiliary || triggerPathAuxiliary_=="" || !e.isRealData()){

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
    if(pfMETCollection.isValid() && pfMETCollection->size()){
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
	  if(CSV>0.244){
	    ++num_csvl;
	    if(CSV>0.679){
	      ++num_csvm;
	      if(CSV>0.898){
		++num_csvt;
	      }
	    }
	  }
	}
      }
    }

    const bool lep_plateau = hlt_lep_pt>lep_pt_threshold_;
    const bool ht_plateau = hlt_ht>ht_threshold_;
    const bool met_plateau = hlt_met>met_threshold_;
    const bool csv_plateau = hlt_csv>csv_threshold_;

    //Fill lepton pt efficiency plot
    if(ht_plateau && met_plateau && csv_plateau //other legs on plateau
       && pfMET > metCut_ //improve lepton purity
       && h_leptonPtTurnOn_den_ && h_leptonPtTurnOn_num_ //have valid pt histos
       && h_leptonIsoTurnOn_den_ && h_leptonIsoTurnOn_num_ //have valid iso histos
       && VertexCollection.isValid() && VertexCollection->size()){//for quality checks
      double maxpt = -1.0, maxpt_iso = -1.0;

      //Try to find a reco electron
      if(ElectronCollection.isValid()
	 && ConversionCollection.isValid()
	 && BeamSpot.isValid()){
	for(const auto &electron: *ElectronCollection){
	  if(IsGood(electron, VertexCollection->front().position(),
		    BeamSpot->position(), ConversionCollection)){
	    if(electron.pt()>lep_pt_threshold_ && electron.pt()>maxpt){
	      maxpt=electron.pt();
	      const auto &iso = electron.pfIsolationVariables();
	      const float absiso = iso.sumChargedHadronPt
		+ std::max(0.0, iso.sumNeutralHadronEt+iso.sumPhotonEt-0.5*iso.sumPUPt);
	      maxpt_iso = absiso/electron.pt();
	    }
	  }
	}
      }
      
      //Try to find a reco muon
      if(MuonCollection.isValid()){
	for(const auto &muon: *MuonCollection){
	  if(IsGood(muon, VertexCollection->front().position())){
	    if(muon.pt()>lep_pt_threshold_ && muon.pt()>maxpt){
	      maxpt=muon.pt();
	      const auto &iso = muon.pfIsolationR03();//QQQ change this?
	      const float absiso = iso.sumChargedHadronPt
		+ std::max(0.0, iso.sumNeutralHadronEt+iso.sumPhotonEt-0.5*iso.sumPUPt);
	      maxpt_iso = absiso/muon.pt();
	    }
	  }
	}
      }

      //Fill histograms using highest pt reco lepton
      if(maxpt>0.0){
	h_leptonPtTurnOn_den_->Fill(maxpt);
	if(hasFired) h_leptonPtTurnOn_num_->Fill(maxpt);
	h_leptonIsoTurnOn_den_->Fill(maxpt_iso);
	if(hasFired) h_leptonIsoTurnOn_num_->Fill(maxpt_iso);
      }
    }

    //Fill HT efficiency plot
    if(lep_plateau && met_plateau && csv_plateau
       && h_pfHTTurnOn_den_ && h_pfHTTurnOn_num_){
      h_pfHTTurnOn_den_->Fill(pfHT);
      if(hasFired) h_pfHTTurnOn_num_->Fill(pfHT);
    }

    //Fill MET efficiency plot
    if(lep_plateau && ht_plateau && csv_plateau
       && h_pfMetTurnOn_den_ && h_pfMetTurnOn_num_){
      h_pfMetTurnOn_den_->Fill(pfMET);
      if(hasFired) h_pfMetTurnOn_num_->Fill(pfMET);
    }

    //Fill CSV efficiency plot
    if(lep_plateau && ht_plateau && met_plateau
       && h_CSVTurnOn_den_ && h_CSVTurnOn_num_){
      switch(num_csvl){
      default: h_CSVTurnOn_den_->Fill(4);
      case 3 : h_CSVTurnOn_den_->Fill(3);
      case 2 : h_CSVTurnOn_den_->Fill(2);
      case 1 : h_CSVTurnOn_den_->Fill(1);
      case 0 : h_CSVTurnOn_den_->Fill(0);
      }
      switch(num_csvm){
      default: h_CSVTurnOn_den_->Fill(8);
      case 3 : h_CSVTurnOn_den_->Fill(7);
      case 2 : h_CSVTurnOn_den_->Fill(6);
      case 1 : h_CSVTurnOn_den_->Fill(5);
      case 0 : break;//Don't double count in the no tag bin
      }
      switch(num_csvt){
      default: h_CSVTurnOn_den_->Fill(12);
      case 3 : h_CSVTurnOn_den_->Fill(11);
      case 2 : h_CSVTurnOn_den_->Fill(10);
      case 1 : h_CSVTurnOn_den_->Fill(9);
      case 0 : break;//Don't double count in the no tag bin
      }
      if(hasFired){
	switch(num_csvl){
	default: h_CSVTurnOn_num_->Fill(4);
	case 3 : h_CSVTurnOn_num_->Fill(3);
	case 2 : h_CSVTurnOn_num_->Fill(2);
	case 1 : h_CSVTurnOn_num_->Fill(1);
	case 0 : h_CSVTurnOn_num_->Fill(0);
	}
	switch(num_csvm){
	default: h_CSVTurnOn_num_->Fill(8);
	case 3 : h_CSVTurnOn_num_->Fill(7);
	case 2 : h_CSVTurnOn_num_->Fill(6);
	case 1 : h_CSVTurnOn_num_->Fill(5);
	case 0 : break;//Don't double count in the no tag bin
	}
	switch(num_csvt){
	default: h_CSVTurnOn_num_->Fill(12);
	case 3 : h_CSVTurnOn_num_->Fill(11);
	case 2 : h_CSVTurnOn_num_->Fill(10);
	case 1 : h_CSVTurnOn_num_->Fill(9);
	case 0 : break;//Don't double count in the no tag bin
	}
      }
    }
  }
}

void SUSY_HLT_SingleLepton::endLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
					       const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton")
    << "SUSY_HLT_SingleLepton::endLuminosityBlock\n";
}

void SUSY_HLT_SingleLepton::endRun(const edm::Run &run, const edm::EventSetup &eSetup){
  edm::LogInfo("SUSY_HLT_SingleLepton") << "SUSY_HLT_SingleLepton::endRun\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(SUSY_HLT_SingleLepton);
