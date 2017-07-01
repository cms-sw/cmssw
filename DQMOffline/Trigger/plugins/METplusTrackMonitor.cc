#include "DQMOffline/Trigger/plugins/METplusTrackMonitor.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQM/TrackingMonitor/interface/GetLumi.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

// -----------------------------
//  constructors and destructor
// -----------------------------

METplusTrackMonitor::METplusTrackMonitor( const edm::ParameterSet& iConfig ) :
  folderName_             ( iConfig.getParameter<std::string>("FolderName") )
  , metToken_             ( consumes<reco::CaloMETCollection>      (iConfig.getParameter<edm::InputTag>("met")            ) )
  , hltMetToken_          ( consumes<reco::CaloMETCollection>      (iConfig.getParameter<edm::InputTag>("onlineMet")      ) )
  , hltMetCleanToken_     ( consumes<reco::CaloMETCollection>      (iConfig.getParameter<edm::InputTag>("onlineMetClean") ) )
  , muonToken_            ( consumes<reco::MuonCollection>         (iConfig.getParameter<edm::InputTag>("muons")          ) )
  , jetToken_             ( consumes<reco::PFJetCollection>        (iConfig.getParameter<edm::InputTag>("jets")           ) )
  , vtxToken_             ( consumes<reco::VertexCollection>       (iConfig.getParameter<edm::InputTag>("vertices")       ) )
  , theTrigSummary_       ( consumes<trigger::TriggerEvent>        (iConfig.getParameter<edm::InputTag>("trigSummary")    ) )
  , trackLegFilterTag_       ( iConfig.getParameter<edm::InputTag>("trackLegFilter") )
  , met_variable_binning_    ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("metBinning") )
  , muonPt_variable_binning_ ( iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<std::vector<double> >("ptBinning")  )
  , met_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("metPSet")    ) )
  , ls_binning_           ( getHistoLSPSet (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("lsPSet")     ) )
  , pt_binning_           ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("ptPSet")     ) )
  , eta_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("etaPSet")    ) )
  , phi_binning_          ( getHistoPSet   (iConfig.getParameter<edm::ParameterSet>("histoPSet").getParameter<edm::ParameterSet>("phiPSet")    ) )
  , num_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"), consumesCollector(), *this))
  , den_genTriggerEventFlag_(new GenericTriggerEventFlag(iConfig.getParameter<edm::ParameterSet>("denGenericTriggerEventPSet"), consumesCollector(), *this))
  , metSelection_         ( iConfig.getParameter<std::string>("metSelection")         )
  , hltMetSelection_      ( iConfig.getParameter<std::string>("hltMetSelection")      )
  , hltMetCleanSelection_ ( iConfig.getParameter<std::string>("hltMetCleanSelection") )
  , muonSelection_        ( iConfig.getParameter<std::string>("muonSelection")        )
  , jetSelection_         ( iConfig.getParameter<std::string>("jetSelection")         )
  , vtxSelection_         ( iConfig.getParameter<std::string>("vtxSelection")         )
  , nmuons_          ( iConfig.getParameter<unsigned>("nmuons") )
  , njets_           ( iConfig.getParameter<unsigned>("njets") )
  , leadJetEtaCut_   ( iConfig.getParameter<double>("leadJetEtaCut") )
  , hltMetCut_       ( iConfig.getParameter<double>("hltMetCut") )
  , hltMetCleanCut_  ( iConfig.getParameter<double>("hltMetCleanCut") )
{

  metME_variableBinning_.numerator   = nullptr;
  metME_variableBinning_.denominator = nullptr;
  metVsLS_.numerator   = nullptr;
  metVsLS_.denominator = nullptr;
  metPhiME_.numerator   = nullptr;
  metPhiME_.denominator = nullptr;
  deltaphimetj1ME_.numerator   = nullptr;
  deltaphimetj1ME_.denominator = nullptr;
  metVsHltMet_.denominator = nullptr;
  metVsHltMetClean_.denominator = nullptr;

  muonPtME_variableBinning_.numerator   = nullptr;
  muonPtME_variableBinning_.denominator = nullptr;
  muonPtVsLS_.numerator   = nullptr;
  muonPtVsLS_.denominator = nullptr;
  deltaphimetmuonME_.numerator = nullptr;
  deltaphimetmuonME_.denominator = nullptr;
  muonEtaVsPhi_.numerator = nullptr;
  muonEtaVsPhi_.denominator = nullptr;

}

METplusTrackMonitor::~METplusTrackMonitor()
{
}

METplusTrackMonitor::MEbinning METplusTrackMonitor::getHistoPSet(edm::ParameterSet pset)
{
  return METplusTrackMonitor::MEbinning {
    pset.getParameter<unsigned int>("nbins"),
    pset.getParameter<double>("xmin"),
    pset.getParameter<double>("xmax")
  };
}

METplusTrackMonitor::MEbinning METplusTrackMonitor::getHistoLSPSet(edm::ParameterSet pset)
{
  return METplusTrackMonitor::MEbinning {
    pset.getParameter<unsigned int>("nbins"),
    0.0,
    double(pset.getParameter<unsigned int>("nbins"))
  };
}

void METplusTrackMonitor::setMETitle(METplusTrackME& me, std::string titleX, std::string titleY, bool bookNumerator = true)
{
  if(bookNumerator) {
    me.numerator->setAxisTitle(titleX, 1);
    me.numerator->setAxisTitle(titleY, 2);
  }
  me.denominator->setAxisTitle(titleX, 1);
  me.denominator->setAxisTitle(titleY, 2);
}

void METplusTrackMonitor::bookME(DQMStore::IBooker &ibooker, METplusTrackME& me, const std::string& histname, const std::string& histtitle, int nbins, double min, double max)
{
  me.numerator   = ibooker.book1D(histname + "_numerator",   histtitle + " (numerator)",   nbins, min, max);
  me.denominator = ibooker.book1D(histname + "_denominator", histtitle + " (denominator)", nbins, min, max);
}

void METplusTrackMonitor::bookME(DQMStore::IBooker &ibooker, METplusTrackME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binning)
{
  int nbins = binning.size() - 1;
  std::vector<float> fbinning(binning.begin(), binning.end());
  float * arr = &fbinning[0];
  me.numerator   = ibooker.book1D(histname + "_numerator",   histtitle + " (numerator)",   nbins, arr);
  me.denominator = ibooker.book1D(histname + "_denominator", histtitle + " (denominator)", nbins, arr);
}

void METplusTrackMonitor::bookME(DQMStore::IBooker &ibooker, METplusTrackME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, double ymin, double ymax)
{
  me.numerator   = ibooker.bookProfile(histname + "_numerator",   histtitle + " (numerator)",   nbinsX, xmin, xmax, ymin, ymax);
  me.denominator = ibooker.bookProfile(histname + "_denominator", histtitle + " (denominator)", nbinsX, xmin, xmax, ymin, ymax);
}

void METplusTrackMonitor::bookME(DQMStore::IBooker &ibooker, METplusTrackME& me, const std::string& histname, const std::string& histtitle, int nbinsX, double xmin, double xmax, int nbinsY, double ymin, double ymax, bool bookNumerator = true)
{
  if(bookNumerator) me.numerator   = ibooker.book2D(histname + "_numerator",   histtitle + " (numerator)",   nbinsX, xmin, xmax, nbinsY, ymin, ymax);
  me.denominator = ibooker.book2D(histname + "_denominator", histtitle + " (denominator)", nbinsX, xmin, xmax, nbinsY, ymin, ymax);
}

void METplusTrackMonitor::bookME(DQMStore::IBooker &ibooker, METplusTrackME& me, const std::string& histname, const std::string& histtitle, const std::vector<double>& binningX, const std::vector<double>& binningY)
{
  int nbinsX = binningX.size() - 1;
  std::vector<float> fbinningX(binningX.begin(), binningX.end());
  float * arrX = &fbinningX[0];
  int nbinsY = binningY.size() - 1;
  std::vector<float> fbinningY(binningY.begin(), binningY.end());
  float * arrY = &fbinningY[0];

  me.numerator   = ibooker.book2D(histname+"_numerator",   histtitle+" (numerator)",   nbinsX, arrX, nbinsY, arrY);
  me.denominator = ibooker.book2D(histname+"_denominator", histtitle+" (denominator)", nbinsX, arrX, nbinsY, arrY);
}

void METplusTrackMonitor::bookHistograms(DQMStore::IBooker     &ibooker,
                                         edm::Run        const &iRun,
                                         edm::EventSetup const &iSetup)
 {
   std::string histname, histtitle;

   std::string currentFolder = folderName_;
   ibooker.setCurrentFolder(currentFolder.c_str());

   // MET leg histograms

   histname = "met_variable"; histtitle = "CaloMET";
   bookME(ibooker, metME_variableBinning_, histname, histtitle, met_variable_binning_);
   setMETitle(metME_variableBinning_, "CaloMET [GeV]", "events / [GeV]");

   histname = "metVsLS"; histtitle = "CaloMET vs LS";
   bookME(ibooker, metVsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, met_binning_.xmin, met_binning_.xmax);
   setMETitle(metVsLS_, "LS", "CaloMET [GeV]");

   histname = "metPhi"; histtitle = "CaloMET phi";
   bookME(ibooker, metPhiME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
   setMETitle(metPhiME_, "CaloMET #phi","events / 0.2 rad");

   histname = "deltaphi_metjet1"; histtitle = "dPhi(CaloMET, jet1)";
   bookME(ibooker, deltaphimetj1ME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
   setMETitle(deltaphimetj1ME_, "#Delta#phi (CaloMET, j1)", "events / 0.2 rad");

   histname = "metVsHltMet"; histtitle = "CaloMET vs hltMet";
   bool bookNumerator = false;
   bookME(ibooker, metVsHltMet_, histname, histtitle, met_binning_.nbins, met_binning_.xmin, met_binning_.xmax, met_binning_.nbins, met_binning_.xmin, met_binning_.xmax, bookNumerator);
   setMETitle(metVsHltMet_, "hltMet (online) [GeV]", "CaloMET (offline) [GeV]", bookNumerator);

   histname = "metVsHltMetClean"; histtitle = "CaloMET vs hltMetClean";
   bookNumerator = false;
   bookME(ibooker, metVsHltMetClean_, histname, histtitle, met_binning_.nbins, met_binning_.xmin, met_binning_.xmax, met_binning_.nbins, met_binning_.xmin, met_binning_.xmax, bookNumerator);
   setMETitle(metVsHltMetClean_, "hltMetClean (online) [GeV]", "CaloMET (offline) [GeV]", bookNumerator);

   // Track leg histograms

   histname = "muonPt_variable"; histtitle = "Muon PT";
   bookME(ibooker, muonPtME_variableBinning_, histname, histtitle, muonPt_variable_binning_);
   setMETitle(muonPtME_variableBinning_, "Muon p_{T} [GeV]", "events / [GeV]");

   histname = "muonEta"; histtitle = "Muon eta";
   bookME(ibooker, muonEtaME_, histname, histtitle, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
   setMETitle(muonEtaME_, "Muon #eta", "events / 0.2");

   histname = "deltaphi_muonmet"; histtitle = "dPhi(Muon, CaloMET)";
   bookME(ibooker, deltaphimetmuonME_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax);
   setMETitle(deltaphimetmuonME_, "#Delta#phi (Muon, CaloMET)", "events / 0.2 rad");

   histname = "muonEtaVsPhi"; histtitle = "Muon eta vs phi";
   bookME(ibooker, muonEtaVsPhi_, histname, histtitle, phi_binning_.nbins, phi_binning_.xmin, phi_binning_.xmax, eta_binning_.nbins, eta_binning_.xmin, eta_binning_.xmax);
   setMETitle(muonEtaVsPhi_, "Muon #phi", "Muon #eta");

   histname = "muonPtVsLS"; histtitle = "Muon PT vs LS";
   bookME(ibooker, muonPtVsLS_, histname, histtitle, ls_binning_.nbins, ls_binning_.xmin, ls_binning_.xmax, pt_binning_.xmin, pt_binning_.xmax);
   setMETitle(muonPtVsLS_, "LS", "Muon p_{T} [GeV]");

   // Initialize the GenericTriggerEventFlag
  if(num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on()) num_genTriggerEventFlag_->initRun(iRun, iSetup);
  if(den_genTriggerEventFlag_ && den_genTriggerEventFlag_->on()) den_genTriggerEventFlag_->initRun(iRun, iSetup);

 }

 void METplusTrackMonitor::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup)
 {

   // Filter out events if Trigger Filtering is requested
  if(den_genTriggerEventFlag_->on() && !den_genTriggerEventFlag_->accept(iEvent, iSetup)) return;

  edm::Handle<reco::CaloMETCollection> metHandle;
  iEvent.getByToken(metToken_, metHandle);
  reco::CaloMET caloMet = metHandle->front();
  if(!metSelection_(caloMet)) return;

  float met = caloMet.pt();
  float metPhi = caloMet.phi();

  edm::Handle<reco::PFJetCollection> jetsHandle;
  iEvent.getByToken(jetToken_, jetsHandle);
  if(jetsHandle->size() < njets_) return;
  std::vector<reco::PFJet> jets;
  jets.clear();
  for(auto const & j : *jetsHandle) {
    if(jetSelection_(j)) jets.push_back(j);
  }
  if(jets.size() < njets_) return;
  if(njets_ > 0 && jets.size() > 0 && fabs(jets[0].eta()) > leadJetEtaCut_) return;
  float deltaphi_metjet1 = (jets.size() > 0) ? fabs(deltaPhi(caloMet.phi(), jets[0].phi())) : 10.0;

  edm::Handle<reco::VertexCollection> primaryVertices;
  iEvent.getByToken(vtxToken_, primaryVertices);
  if(!primaryVertices->size()) return;
  const reco::Vertex* pv = nullptr;
  for(auto const& v: *primaryVertices) {
    if (!vtxSelection_(v)) continue;
    pv = &v;
    break;
  }
  if(pv == nullptr) return;

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByToken(muonToken_, muonHandle);
  if(muonHandle->size() < nmuons_) return;
  std::vector<reco::Muon> muons;
  muons.clear();
  for(auto const & m : *muonHandle) {
    bool passTightID = muon::isTightMuon(m, *pv);
    bool passMissHits = m.innerTrack()->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS) == 0 &&
                        m.innerTrack()->hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS) == 0;
    if(muonSelection_(m) && passTightID && passMissHits) muons.push_back(m);
  }
  if(muons.size() < nmuons_) return;

  edm::Handle<reco::CaloMETCollection> hltMetHandle;
  iEvent.getByToken(hltMetToken_, hltMetHandle);
  if(!hltMetHandle.isValid()) {
    edm::LogError("METplusTrackMonitor") << "Invalid collection: hltMet" << "\n";
    return;
  }
  reco::CaloMET hltMet = hltMetHandle->front();

  edm::Handle<reco::CaloMETCollection> hltMetCleanHandle;
  iEvent.getByToken(hltMetCleanToken_, hltMetCleanHandle);
  if(!hltMetCleanHandle.isValid()) {
    edm::LogError("METplusTrackMonitor") << "Invalid collection: hltMetClean" << "\n";
    return;
  }
  reco::CaloMET hltMetClean = hltMetCleanHandle->front();

  // Filling MET leg histograms (denominator)
  metME_variableBinning_.denominator->Fill(met);
  metPhiME_.denominator->Fill(metPhi);
  deltaphimetj1ME_.denominator->Fill(deltaphi_metjet1);
  metVsHltMet_.denominator->Fill(hltMet.pt(), met);
  metVsHltMetClean_.denominator->Fill(hltMetClean.pt(), met);

  int ls = iEvent.id().luminosityBlock();
  metVsLS_.denominator->Fill(ls, met);

  // Apply the selection for the MET leg numerator
  if(hltMet.pt() <= hltMetCut_ || hltMetClean.pt() <= hltMetCleanCut_) return;

  // Filling MET leg histograms (numerator)
  metME_variableBinning_.numerator->Fill(met);
  metPhiME_.numerator->Fill(metPhi);
  deltaphimetj1ME_.numerator->Fill(deltaphi_metjet1);
  metVsLS_.numerator->Fill(ls, met);

  // Filling track leg histograms (denominator)
  double leadMuonPt = muons.size() ? muons[0].pt() : -1.0;
  double leadMuonEta = muons.size() ? muons[0].eta() : 10.0;
  double leadMuonPhi = muons.size() ? muons[0].phi() : 10.0;
  float deltaphi_metmuon = fabs(deltaPhi(caloMet.phi(), muons[0].phi()));

  muonPtME_variableBinning_.denominator->Fill(leadMuonPt);
  muonPtVsLS_.denominator->Fill(ls, leadMuonPt);
  muonEtaME_.denominator->Fill(leadMuonEta);
  deltaphimetmuonME_.denominator->Fill(deltaphi_metmuon);
  muonEtaVsPhi_.denominator->Fill(leadMuonPhi, leadMuonEta);

  // Apply the selection for the track leg numerator
  edm::Handle<trigger::TriggerEvent> triggerSummary;
  iEvent.getByToken(theTrigSummary_, triggerSummary);
  if(!triggerSummary.isValid()) {
    edm::LogError("METplusTrackMonitor") << "Invalid collection: TriggerSummary" << "\n";
    return;
  }
  trigger::TriggerObject isoTrk;
  bool passesTrackLegFilter = getHLTObj(triggerSummary, trackLegFilterTag_, isoTrk);

  bool leadMuonMatchToHLTTrack = muons.size() > 0 &&
                                 passesTrackLegFilter &&
                                 deltaR(muons[0], isoTrk) < 0.1;
  if(!leadMuonMatchToHLTTrack) return;

  if(num_genTriggerEventFlag_->on() && !num_genTriggerEventFlag_->accept(iEvent, iSetup)) return;

  // Filling track leg histograms (denominator)
  muonPtME_variableBinning_.numerator->Fill(leadMuonPt);
  muonPtVsLS_.numerator->Fill(ls, leadMuonPt);
  muonEtaME_.numerator->Fill(leadMuonEta);
  deltaphimetmuonME_.numerator->Fill(deltaphi_metmuon);
  muonEtaVsPhi_.numerator->Fill(leadMuonPhi, leadMuonEta);

}

void METplusTrackMonitor::fillHistoPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>   ("nbins");
  pset.add<double>         ("xmin");
  pset.add<double>         ("xmax");
}

void METplusTrackMonitor::fillHistoLSPSetDescription(edm::ParameterSetDescription & pset)
{
  pset.add<unsigned int>("nbins", 2500);
}

void METplusTrackMonitor::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>  ( "FolderName", "HLT/MET" );

  desc.add<edm::InputTag>( "met",            edm::InputTag("caloMet") );
  desc.add<edm::InputTag>( "jets",           edm::InputTag("ak4PFJetsCHS") );
  desc.add<edm::InputTag>( "electrons",      edm::InputTag("gedGsfElectrons") );
  desc.add<edm::InputTag>( "muons",          edm::InputTag("muons") );
  desc.add<edm::InputTag>( "vertices",       edm::InputTag("offlinePrimaryVertices") );
  desc.add<edm::InputTag>( "trigSummary",    edm::InputTag("hltTriggerSummaryAOD") );
  desc.add<edm::InputTag>( "onlineMet",      edm::InputTag("hltMet", "", "HLT") );
  desc.add<edm::InputTag>( "onlineMetClean", edm::InputTag("hltMetClean", "", "HLT") );
  desc.add<edm::InputTag>( "trackLegFilter", edm::InputTag("hltTrk50Filter", "", "HLT") );

  desc.add<std::string>("metSelection", "pt > 0");
  desc.add<std::string>("hltMetSelection", "pt > 0");
  desc.add<std::string>("hltMetCleanSelection", "pt > 0");
  desc.add<std::string>("jetSelection", "pt > 0");
  desc.add<std::string>("muonSelection", "pt > 0");
  desc.add<std::string>("vtxSelection",  "!isFake");
  desc.add<unsigned>("njets",  0);
  desc.add<unsigned>("nmuons", 0);
  desc.add<double>("leadJetEtaCut", 2.4);
  desc.add<double>("hltMetCut", 105.0);
  desc.add<double>("hltMetCleanCut", 65.0);

  edm::ParameterSetDescription genericTriggerEventPSet;
  genericTriggerEventPSet.add<bool>("andOr");
  genericTriggerEventPSet.add<edm::InputTag>("dcsInputTag", edm::InputTag("scalersRawToDigi") );
  genericTriggerEventPSet.add<std::vector<int> >("dcsPartitions", {});
  genericTriggerEventPSet.add<bool>("andOrDcs", false);
  genericTriggerEventPSet.add<bool>("errorReplyDcs", true);
  genericTriggerEventPSet.add<std::string>("dbLabel", "");
  genericTriggerEventPSet.add<bool>("andOrHlt", true);
  genericTriggerEventPSet.add<edm::InputTag>("hltInputTag", edm::InputTag("TriggerResults::HLT") );
  genericTriggerEventPSet.add<std::vector<std::string> >("hltPaths", {});
  genericTriggerEventPSet.add<std::string>("hltDBKey", "");
  genericTriggerEventPSet.add<bool>("errorReplyHlt", false);
  genericTriggerEventPSet.add<unsigned int>("verbosityLevel", 1);

  desc.add<edm::ParameterSetDescription>("numGenericTriggerEventPSet", genericTriggerEventPSet);
  desc.add<edm::ParameterSetDescription>("denGenericTriggerEventPSet", genericTriggerEventPSet);

  edm::ParameterSetDescription histoPSet;
  edm::ParameterSetDescription metPSet;
  edm::ParameterSetDescription phiPSet;
  edm::ParameterSetDescription ptPSet;
  edm::ParameterSetDescription etaPSet;

  fillHistoPSetDescription(metPSet);
  fillHistoPSetDescription(phiPSet);
  fillHistoPSetDescription(ptPSet);
  fillHistoPSetDescription(etaPSet);

  histoPSet.add<edm::ParameterSetDescription>("metPSet", metPSet);
  histoPSet.add<edm::ParameterSetDescription>("phiPSet", phiPSet);
  histoPSet.add<edm::ParameterSetDescription>("ptPSet",  ptPSet);
  histoPSet.add<edm::ParameterSetDescription>("etaPSet", etaPSet);

  std::vector<double> bins = {0.,20.,40.,60.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.,180.,190.,200.,220.,240.,260.,280.,300.,350.,400.,450.,1000.};

  histoPSet.add<std::vector<double> >("metBinning", bins);
  histoPSet.add<std::vector<double> >("ptBinning",  bins);

  edm::ParameterSetDescription lsPSet;
  fillHistoLSPSetDescription(lsPSet);
  histoPSet.add<edm::ParameterSetDescription>("lsPSet", lsPSet);

  desc.add<edm::ParameterSetDescription>("histoPSet", histoPSet);

  descriptions.add("metPlusTrackMonitoring", desc);
}

bool METplusTrackMonitor::getHLTObj(const edm::Handle<trigger::TriggerEvent> &trigSummary, const edm::InputTag filterTag, trigger::TriggerObject &obj) const
{

 double leadingPt = -1.0;

 size_t filterIndex = trigSummary->filterIndex( filterTag );
 trigger::TriggerObjectCollection triggerObjects = trigSummary->getObjects();

 if( !(filterIndex >= trigSummary->sizeFilters()) ) {
   const trigger::Keys& keys = trigSummary->filterKeys( filterIndex );
   for(size_t j = 0; j < keys.size(); ++j) {
     trigger::TriggerObject foundObject = triggerObjects[keys[j]];
     if(foundObject.pt() > leadingPt) {
       obj = foundObject;
       leadingPt = obj.pt();
     }
   }
 }

 return (leadingPt > 0.0);
}

// Define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(METplusTrackMonitor);

