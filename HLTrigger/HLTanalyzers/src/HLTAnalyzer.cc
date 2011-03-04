// File: HLTAnalyzer.cc
// Description:  Example of Analysis driver originally from Jeremy Mans, 
// Date:  13-October-2006

#include <boost/foreach.hpp>

#include "HLTrigger/HLTanalyzers/interface/HLTAnalyzer.h"
#include "HLTMessages.h"

typedef std::pair<const char *, const edm::InputTag *> MissingCollectionInfo;

template <class T>
static inline
bool getCollection(const edm::Event & event, std::vector<MissingCollectionInfo> & missing, edm::Handle<T> & handle, const edm::InputTag & name, const char * description) 
{
    event.getByLabel(name, handle);
    bool valid = handle.isValid();
    if (not valid) {
        missing.push_back( std::make_pair(description, & name) );
        handle.clear();
    }
    return valid;
}

// Boiler-plate constructor definition of an analyzer module:
HLTAnalyzer::HLTAnalyzer(edm::ParameterSet const& conf) {
    
    // If your module takes parameters, here is where you would define
    // their names and types, and access them to initialize internal
    // variables. Example as follows:
    std::cout << " Beginning HLTAnalyzer Analysis " << std::endl;
    
    recjets_          = conf.getParameter<edm::InputTag> ("recjets");
    reccorjets_       = conf.getParameter<edm::InputTag> ("reccorjets");
    genjets_          = conf.getParameter<edm::InputTag> ("genjets");
    recmet_           = conf.getParameter<edm::InputTag> ("recmet");
    genmet_           = conf.getParameter<edm::InputTag> ("genmet");
    ht_               = conf.getParameter<edm::InputTag> ("ht");
    recoPFJets_       = conf.getParameter<edm::InputTag> ("recoPFJets"); 
    calotowers_       = conf.getParameter<edm::InputTag> ("calotowers");
    muon_             = conf.getParameter<edm::InputTag> ("muon");
    mctruth_          = conf.getParameter<edm::InputTag> ("mctruth");
    genEventInfo_     = conf.getParameter<edm::InputTag> ("genEventInfo");
    simhits_          = conf.getParameter<edm::InputTag> ("simhits");
    xSection_         = conf.getUntrackedParameter<double> ("xSection",1.);
    filterEff_        = conf.getUntrackedParameter<double> ("filterEff",1.);
    firstLumi_        = conf.getUntrackedParameter<int> ("firstLumi",0);
    lastLumi_         = conf.getUntrackedParameter<int> ("lastLumi",-1);
    towerThreshold_   = conf.getParameter<double>("caloTowerThreshold");
       
    // keep this separate from l1extramc_ as needed by FastSim:
    //    This is purposefully done this way to allow FastSim to run with OpenHLT: 
    //    The {FastSim+OpenHLT} package runs on the head of HLTrigger/HLTanalyzers 
    //    where there is purposefully this duplication because FastSim does the 
    //    simulation of muons seperately, and needs the same collection. 
    l1extramu_        = conf.getParameter<std::string>   ("l1extramu");
    m_l1extramu       = edm::InputTag(l1extramu_, "");
    
    // read the L1Extra collection name, and add the instance names as needed
    l1extramc_        = conf.getParameter<std::string>   ("l1extramc");
    m_l1extraemi      = edm::InputTag(l1extramc_, "Isolated");
    m_l1extraemn      = edm::InputTag(l1extramc_, "NonIsolated");
    m_l1extrajetc     = edm::InputTag(l1extramc_, "Central");
    m_l1extrajetf     = edm::InputTag(l1extramc_, "Forward");
    m_l1extrataujet   = edm::InputTag(l1extramc_, "Tau");
    m_l1extramet      = edm::InputTag(l1extramc_, "MET");
    m_l1extramht      = edm::InputTag(l1extramc_, "MHT");
    
    hltresults_       = conf.getParameter<edm::InputTag> ("hltresults");
    gtReadoutRecord_  = conf.getParameter<edm::InputTag> ("l1GtReadoutRecord");
    
    gctBitCounts_        = edm::InputTag( conf.getParameter<edm::InputTag>("l1GctHFBitCounts").label(), "" );
    gctRingSums_         = edm::InputTag( conf.getParameter<edm::InputTag>("l1GctHFRingSums").label(), "" );
    
    MuCandTag2_          = conf.getParameter<edm::InputTag> ("MuCandTag2");
    MuIsolTag2_          = conf.getParameter<edm::InputTag> ("MuIsolTag2");
    MuNoVtxCandTag2_     = conf.getParameter<edm::InputTag> ("MuNoVtxCandTag2");
    MuCandTag3_          = conf.getParameter<edm::InputTag> ("MuCandTag3");
    MuIsolTag3_          = conf.getParameter<edm::InputTag> ("MuIsolTag3");
    oniaPixelTag_        = conf.getParameter<edm::InputTag> ("OniaPixelTag");
    oniaTrackTag_        = conf.getParameter<edm::InputTag> ("OniaTrackTag");
    HLTTau_              = conf.getParameter<edm::InputTag> ("HLTTau");
    PFTau_               = conf.getParameter<edm::InputTag> ("HLTPFTau");
    _MinPtChargedHadrons = conf.getParameter<double>("minPtChargedHadronsForTaus");
    _MinPtGammas         = conf.getParameter<double>("minPtGammassForTaus");

    PFJets_          = conf.getParameter<edm::InputTag> ("HLTPFJet");
    
    // offline reco tau collection and discriminators
    RecoPFTau_                          = conf.getParameter<edm::InputTag> ("RecoPFTau");
    RecoPFTauDiscrByTanCOnePercent_     = conf.getParameter<edm::InputTag> ("RecoPFTauDiscrByTanCOnePercent"); 
    RecoPFTauDiscrByTanCHalfPercent_    = conf.getParameter<edm::InputTag> ("RecoPFTauDiscrByTanCHalfPercent");  
    RecoPFTauDiscrByTanCQuarterPercent_ = conf.getParameter<edm::InputTag> ("RecoPFTauDiscrByTanCQuarterPercent");
    RecoPFTauDiscrByTanCTenthPercent_   = conf.getParameter<edm::InputTag> ("RecoPFTauDiscrByTanCTenthPercent");
    RecoPFTauDiscrByIso_                = conf.getParameter<edm::InputTag> ("RecoPFTauDiscrByIso");  
    RecoPFTauAgainstMuon_               = conf.getParameter<edm::InputTag> ("RecoPFTauAgainstMuon");  
    RecoPFTauAgainstElec_               = conf.getParameter<edm::InputTag> ("RecoPFTauAgainstElec");  
   
    
    // btag OpenHLT input collections
    m_rawBJets                = conf.getParameter<edm::InputTag>("CommonBJetsL2");
    m_correctedBJets          = conf.getParameter<edm::InputTag>("CorrectedBJetsL2");
    m_lifetimeBJetsL25        = conf.getParameter<edm::InputTag>("LifetimeBJetsL25");
    m_lifetimeBJetsL3         = conf.getParameter<edm::InputTag>("LifetimeBJetsL3");
    m_softmuonBJetsL25        = conf.getParameter<edm::InputTag>("SoftmuonBJetsL25");
    m_softmuonBJetsL3         = conf.getParameter<edm::InputTag>("SoftmuonBJetsL3");
    m_performanceBJetsL25     = conf.getParameter<edm::InputTag>("PerformanceBJetsL25");
    m_performanceBJetsL3      = conf.getParameter<edm::InputTag>("PerformanceBJetsL3");
    
    // egamma OpenHLT input collections
    Electron_                 = conf.getParameter<edm::InputTag> ("Electron");
    Photon_                   = conf.getParameter<edm::InputTag> ("Photon");
    CandIso_                  = conf.getParameter<edm::InputTag> ("CandIso");
    CandNonIso_               = conf.getParameter<edm::InputTag> ("CandNonIso");
    EcalIso_                  = conf.getParameter<edm::InputTag> ("EcalIso");
    EcalNonIso_               = conf.getParameter<edm::InputTag> ("EcalNonIso");
    HcalIsoPho_               = conf.getParameter<edm::InputTag> ("HcalIsoPho");
    HcalNonIsoPho_            = conf.getParameter<edm::InputTag> ("HcalNonIsoPho");
    IsoPhoTrackIsol_          = conf.getParameter<edm::InputTag> ("IsoPhoTrackIsol");
    NonIsoPhoTrackIsol_       = conf.getParameter<edm::InputTag> ("NonIsoPhoTrackIsol");
    IsoElectron_              = conf.getParameter<edm::InputTag> ("IsoElectrons");
    NonIsoElectron_           = conf.getParameter<edm::InputTag> ("NonIsoElectrons");
    IsoEleHcal_               = conf.getParameter<edm::InputTag> ("HcalIsoEle");
    NonIsoEleHcal_            = conf.getParameter<edm::InputTag> ("HcalNonIsoEle");
    IsoEleTrackIsol_          = conf.getParameter<edm::InputTag> ("IsoEleTrackIsol");
    NonIsoEleTrackIsol_       = conf.getParameter<edm::InputTag> ("NonIsoEleTrackIsol");
    L1IsoPixelSeeds_          = conf.getParameter<edm::InputTag> ("PixelSeedL1Iso");
    L1NonIsoPixelSeeds_       = conf.getParameter<edm::InputTag> ("PixelSeedL1NonIso");
    IsoR9_                    = conf.getParameter<edm::InputTag> ("SpikeCleaningIsol");  
    NonIsoR9_                 = conf.getParameter<edm::InputTag> ("SpikeCleaningNonIsol");   
    IsoHoverEH_               = conf.getParameter<edm::InputTag> ("HcalForHoverEIsol");
    NonIsoHoverEH_            = conf.getParameter<edm::InputTag> ("HcalForHoverENonIsol"); 
    IsoR9ID_                  = conf.getParameter<edm::InputTag> ("R9IDIsol");
    NonIsoR9ID_               = conf.getParameter<edm::InputTag> ("R9IDNonIsol");
    HFECALClusters_           = conf.getParameter<edm::InputTag> ("HFECALClusters"); 
    HFElectrons_              = conf.getParameter<edm::InputTag> ("HFElectrons"); 

    // AlCa OpenHLT input collections  
    EERecHitTag_              = conf.getParameter<edm::InputTag> ("EERecHits"); 
    EBRecHitTag_              = conf.getParameter<edm::InputTag> ("EBRecHits"); 
    pi0EBRecHitTag_           = conf.getParameter<edm::InputTag> ("pi0EBRecHits");  
    pi0EERecHitTag_           = conf.getParameter<edm::InputTag> ("pi0EERecHits");  
    HBHERecHitTag_            = conf.getParameter<edm::InputTag> ("HBHERecHits");  
    HORecHitTag_              = conf.getParameter<edm::InputTag> ("HORecHits");  
    HFRecHitTag_              = conf.getParameter<edm::InputTag> ("HFRecHits");  
    IsoPixelTrackTagL3_       = conf.getParameter<edm::InputTag> ("IsoPixelTracksL3"); 
    IsoPixelTrackTagL2_       = conf.getParameter<edm::InputTag> ("IsoPixelTracksL2");
    IsoPixelTrackVerticesTag_       = conf.getParameter<edm::InputTag> ("IsoPixelTrackVertices");
    
    // Track OpenHLT input collections
    PixelTracksTagL3_         = conf.getParameter<edm::InputTag> ("PixelTracksL3"); 
    
    // Reco Vertex collection
    VertexTag_                = conf.getParameter<edm::InputTag> ("PrimaryVertices");  
    
    m_file = 0;   // set to null
    errCnt = 0;
    
    // read run parameters with a default value 
    edm::ParameterSet runParameters = conf.getParameter<edm::ParameterSet>("RunParameters");
    _HistName = runParameters.getUntrackedParameter<std::string>("HistogramFile", "test.root");
    _EtaMin   = runParameters.getUntrackedParameter<double>("EtaMin", -5.2);
    _EtaMax   = runParameters.getUntrackedParameter<double>("EtaMax",  5.2);
    
    
    
    // open the tree file
    m_file = new TFile(_HistName.c_str(), "RECREATE");
    if (m_file)
        m_file->cd();
    
    // Initialize the tree
    HltTree = new TTree("HltTree", "");
    
    treeWeight=xSection_*filterEff_;
    std::cout << "\n Setting HltTree weight to " << treeWeight << " = " << xSection_ << "*" << filterEff_ << " (cross section * gen filter efficiency)\n" << std::endl;
    
    // Setup the different analysis
    jet_analysis_.setup(conf, HltTree);
    bjet_analysis_.setup(conf, HltTree);
    elm_analysis_.setup(conf, HltTree);
    muon_analysis_.setup(conf, HltTree);
    track_analysis_.setup(conf, HltTree);
    mct_analysis_.setup(conf, HltTree);
    hlt_analysis_.setup(conf, HltTree);
    vrt_analysis_.setup(conf, HltTree);
    evt_header_.setup(HltTree);
}

void HLTAnalyzer::beginRun(const edm::Run& run, const edm::EventSetup& c){ 
    
    hlt_analysis_.beginRun(run, c);
}

// Boiler-plate "analyze" method declaration for an analyzer module.
void HLTAnalyzer::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    
    // To get information from the event setup, you must request the "Record"
    // which contains it and then extract the object you need
    //edm::ESHandle<CaloGeometry> geometry;
    //iSetup.get<IdealGeometryRecord>().get(geometry);
    
    int iLumi = iEvent.luminosityBlock();
    if (iLumi<firstLumi_) return;
    if (lastLumi_ != -1 && iLumi>lastLumi_) return;
    
    // These declarations create handles to the types of records that you want
    // to retrieve from event "iEvent".
    edm::Handle<reco::CaloJetCollection>              recjets;
    edm::Handle<reco::CaloJetCollection>              reccorjets;
    edm::Handle<reco::GenJetCollection>               genjets;
    edm::Handle<CaloTowerCollection>                  caloTowers;
    edm::Handle<reco::CaloMETCollection>              recmet;
    edm::Handle<reco::GenMETCollection>               genmet;
    edm::Handle<reco::METCollection>                  ht;
    edm::Handle<reco::PFJetCollection>                recoPFJets; 
    edm::Handle<reco::CandidateView>                  mctruth;
    edm::Handle<GenEventInfoProduct>                  genEventInfo;
    edm::Handle<std::vector<SimTrack> >               simTracks;
    edm::Handle<std::vector<SimVertex> >              simVertices;
    edm::Handle<reco::MuonCollection>                 muon;
    edm::Handle<edm::TriggerResults>                  hltresults;
    edm::Handle<l1extra::L1EmParticleCollection>      l1extemi, l1extemn;
    edm::Handle<l1extra::L1MuonParticleCollection>    l1extmu;
    edm::Handle<l1extra::L1JetParticleCollection>     l1extjetc, l1extjetf, l1exttaujet;
    edm::Handle<l1extra::L1EtMissParticleCollection>  l1extmet,l1extmht;
    edm::Handle<L1GlobalTriggerReadoutRecord>         l1GtRR;
    edm::Handle< L1GctHFBitCountsCollection >         gctBitCounts ;
    edm::Handle< L1GctHFRingEtSumsCollection >        gctRingSums ;
    
    edm::Handle<reco::RecoChargedCandidateCollection> mucands2, mucands3, munovtxcands2;
    edm::Handle<reco::RecoChargedCandidateCollection> oniaPixelCands, oniaTrackCands;
    edm::Handle<edm::ValueMap<bool> >                 isoMap2,  isoMap3;
    edm::Handle<reco::HLTTauCollection>               taus;
    edm::Handle<reco::PFTauCollection>               pftaus;
    edm::Handle<reco::PFJetCollection>               pfjets;
    
    // offline reco tau collection and discriminators
    edm::Handle<reco::PFTauCollection>  recoPftaus;
    edm::Handle<reco::PFTauDiscriminator> theRecoPFTauDiscrByTanCOnePercent;
    edm::Handle<reco::PFTauDiscriminator> theRecoPFTauDiscrByTanCHalfPercent; 
    edm::Handle<reco::PFTauDiscriminator> theRecoPFTauDiscrByTanCQuarterPercent;
    edm::Handle<reco::PFTauDiscriminator> theRecoPFTauDiscrByTanCTenthPercent;
    edm::Handle<reco::PFTauDiscriminator> theRecoPFTauDiscrByIsolation;
    edm::Handle<reco::PFTauDiscriminator> theRecoPFTauDiscrAgainstMuon;
    edm::Handle<reco::PFTauDiscriminator> theRecoPFTauDiscrAgainstElec;
   
    
    // btag OpenHLT input collections
    edm::Handle<edm::View<reco::Jet> >                hRawBJets;
    edm::Handle<edm::View<reco::Jet> >                hCorrectedBJets;
    edm::Handle<reco::JetTagCollection>               hLifetimeBJetsL25;
    edm::Handle<reco::JetTagCollection>               hLifetimeBJetsL3;
    edm::Handle<reco::JetTagCollection>               hSoftmuonBJetsL25;
    edm::Handle<reco::JetTagCollection>               hSoftmuonBJetsL3;
    edm::Handle<reco::JetTagCollection>               hPerformanceBJetsL25;
    edm::Handle<reco::JetTagCollection>               hPerformanceBJetsL3;
    
    // egamma OpenHLT input collections
    edm::Handle<reco::GsfElectronCollection>          electrons;
    edm::Handle<reco::PhotonCollection>               photons;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    photonR9IsoHandle; 
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    photonR9NonIsoHandle;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    photonR9IDIsoHandle;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    photonR9IDNonIsoHandle;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  photonHoverEHIsoHandle;   
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  photonHoverEHNonIsoHandle;    
    edm::Handle<reco::ElectronCollection>             electronIsoHandle;
    edm::Handle<reco::ElectronCollection>             electronNonIsoHandle;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    electronR9IsoHandle; 
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    electronR9NonIsoHandle;  
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    electronR9IDIsoHandle;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>    electronR9IDNonIsoHandle;
    edm::Handle<reco::ElectronIsolationMap>           NonIsoTrackEleIsolMap;
    edm::Handle<reco::ElectronIsolationMap>           TrackEleIsolMap;
    edm::Handle<reco::ElectronSeedCollection>         L1IsoPixelSeedsMap;
    edm::Handle<reco::ElectronSeedCollection>         L1NonIsoPixelSeedsMap;
    edm::Handle<reco::RecoEcalCandidateCollection>    recoIsolecalcands;
    edm::Handle<reco::RecoEcalCandidateCollection>    recoNonIsolecalcands;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  EcalIsolMap;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  EcalNonIsolMap;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  HcalEleIsolMap;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  HcalEleNonIsolMap;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  HcalIsolMap;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  HcalNonIsolMap;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  TrackIsolMap;
    edm::Handle<reco::RecoEcalCandidateIsolationMap>  TrackNonIsolMap;
    edm::Handle<reco::SuperClusterCollection>         electronHFClusterHandle; 
    edm::Handle<reco::RecoEcalCandidateCollection>    electronHFElectronHandle;  
    
    // AlCa OpenHLT input collections   
    edm::Handle<EBRecHitCollection>             ebrechits;  
    edm::Handle<EERecHitCollection>             eerechits;   
    edm::Handle<EBRecHitCollection>             pi0ebrechits;   
    edm::Handle<EERecHitCollection>             pi0eerechits;    
    edm::Handle<HBHERecHitCollection>           hbherechits;   
    edm::Handle<HORecHitCollection>             horechits;   
    edm::Handle<HFRecHitCollection>             hfrechits;   
    edm::Handle<reco::IsolatedPixelTrackCandidateCollection> isopixeltracksL3; 
    edm::Handle<reco::IsolatedPixelTrackCandidateCollection> isopixeltracksL2;	
    edm::Handle<reco::VertexCollection>         isopixeltrackPixVertices;
    edm::Handle<reco::RecoChargedCandidateCollection> pixeltracksL3; 
    
    // Reco vertex collection
    edm::Handle<reco::VertexCollection> recoVertexs;
    
    // new stuff for the egamma EleId
    edm::InputTag ecalRechitEBTag (std::string("hltEcalRegionalEgammaRecHit:EcalRecHitsEB"));
    edm::InputTag ecalRechitEETag (std::string("hltEcalRegionalEgammaRecHit:EcalRecHitsEE"));
    EcalClusterLazyTools lazyTools( iEvent, iSetup, ecalRechitEBTag, ecalRechitEETag);
    
    edm::ESHandle<MagneticField>                theMagField;
    iSetup.get<IdealMagneticFieldRecord>().get(theMagField);
    
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    edm::InputTag BSProducer_(std::string("hltOnlineBeamSpot"));
    
    // get EventSetup stuff needed for the AlCa pi0 path
    edm::ESHandle< EcalElectronicsMapping > ecalmapping;
    iSetup.get< EcalMappingRcd >().get(ecalmapping);
    
    edm::ESHandle<CaloGeometry> geoHandle;
    iSetup.get<CaloGeometryRecord>().get(geoHandle); 
    
    edm::ESHandle<CaloTopology> pTopology;
    iSetup.get<CaloTopologyRecord>().get(pTopology);
    
    edm::ESHandle<L1CaloGeometry> l1CaloGeom ;
    iSetup.get<L1CaloGeometryRecord>().get(l1CaloGeom) ;
    
    
    // extract the collections from the event, check their validity and log which are missing
    std::vector<MissingCollectionInfo> missing;
    
    //get the BeamSpot
    getCollection( iEvent, missing, recoBeamSpotHandle,       BSProducer_ ,          "Beam Spot handle");
    // gets its position
    reco::BeamSpot::Point BSPosition(0,0,0);
    BSPosition = recoBeamSpotHandle->position();
    
    getCollection( iEvent, missing, recjets,         recjets_,           kRecjets );
    getCollection( iEvent, missing, reccorjets,      reccorjets_,        kRecCorjets );
    getCollection( iEvent, missing, genjets,         genjets_,           kGenjets );
    getCollection( iEvent, missing, recmet,          recmet_,            kRecmet );
    getCollection( iEvent, missing, genmet,          genmet_,            kGenmet );
    getCollection( iEvent, missing, caloTowers,      calotowers_,        kCaloTowers );
    getCollection( iEvent, missing, ht,              ht_,                kHt );
    getCollection( iEvent, missing, recoPFJets,      recoPFJets_,        kRecoPFJets );   
    getCollection( iEvent, missing, muon,            muon_,              kMuon );
    getCollection( iEvent, missing, taus,            HLTTau_,            kTaus );
    getCollection( iEvent, missing, pftaus,          PFTau_,		 kPFTaus );
    getCollection( iEvent, missing, pfjets,          PFJets_,		 kPFJets );  
    getCollection( iEvent, missing, recoPftaus,                            RecoPFTau_,                          kRecoPFTaus );
    getCollection( iEvent, missing, theRecoPFTauDiscrByTanCOnePercent,     RecoPFTauDiscrByTanCOnePercent_,     ktheRecoPFTauDiscrByTanCOnePercent); 
    getCollection( iEvent, missing, theRecoPFTauDiscrByTanCHalfPercent,    RecoPFTauDiscrByTanCHalfPercent_,    ktheRecoPFTauDiscrByTanCHalfPercent); 
    getCollection( iEvent, missing, theRecoPFTauDiscrByTanCQuarterPercent, RecoPFTauDiscrByTanCQuarterPercent_, ktheRecoPFTauDiscrByTanCQuarterPercent); 
    getCollection( iEvent, missing, theRecoPFTauDiscrByTanCTenthPercent,   RecoPFTauDiscrByTanCTenthPercent_,   ktheRecoPFTauDiscrByTanCTenthPercent);     
    getCollection( iEvent, missing, theRecoPFTauDiscrByIsolation,          RecoPFTauDiscrByIso_,                ktheRecoPFTauDiscrByIsolation); 
    getCollection( iEvent, missing, theRecoPFTauDiscrAgainstMuon,          RecoPFTauAgainstMuon_,               ktheRecoPFTauDiscrAgainstMuon); 
    getCollection( iEvent, missing, theRecoPFTauDiscrAgainstElec,          RecoPFTauAgainstElec_,               ktheRecoPFTauDiscrAgainstElec); 
    getCollection( iEvent, missing, hltresults,      hltresults_,        kHltresults );
    getCollection( iEvent, missing, l1extemi,        m_l1extraemi,       kL1extemi );
    getCollection( iEvent, missing, l1extemn,        m_l1extraemn,       kL1extemn );
    getCollection( iEvent, missing, l1extmu,         m_l1extramu,        kL1extmu );
    getCollection( iEvent, missing, l1extjetc,       m_l1extrajetc,      kL1extjetc );
    getCollection( iEvent, missing, l1extjetf,       m_l1extrajetf,      kL1extjetf );
    getCollection( iEvent, missing, l1exttaujet,     m_l1extrataujet,    kL1exttaujet );
    getCollection( iEvent, missing, l1extmet,        m_l1extramet,       kL1extmet );
    getCollection( iEvent, missing, l1extmht,        m_l1extramht,       kL1extmht );
    getCollection( iEvent, missing, l1GtRR,          gtReadoutRecord_,   kL1GtRR );
    getCollection( iEvent, missing, gctBitCounts,     gctBitCounts_,      kL1GctBitCounts );
    getCollection( iEvent, missing, gctRingSums,      gctRingSums_,       kL1GctRingSums );
    getCollection( iEvent, missing, mctruth,         mctruth_,           kMctruth );
    getCollection( iEvent, missing, simTracks,       simhits_,           kSimhit );
    getCollection( iEvent, missing, simVertices,     simhits_,           kSimhit );
    getCollection( iEvent, missing, genEventInfo,    genEventInfo_,      kGenEventInfo );
    getCollection( iEvent, missing, mucands2,        MuCandTag2_,        kMucands2 );
    getCollection( iEvent, missing, munovtxcands2,   MuNoVtxCandTag2_,   kMunovtxcands2 ); 
    getCollection( iEvent, missing, mucands3,        MuCandTag3_,        kMucands3 );
    getCollection( iEvent, missing, oniaPixelCands,        oniaPixelTag_,        kOniaPixelCands );
    getCollection( iEvent, missing, oniaTrackCands,        oniaTrackTag_,        kOniaTrackCands );
    getCollection( iEvent, missing, isoMap2,         MuIsolTag2_,        kIsoMap2 );
    getCollection( iEvent, missing, isoMap3,         MuIsolTag3_,        kIsoMap3 );
    getCollection( iEvent, missing, hRawBJets,                m_rawBJets,                 kBTagJets );
    getCollection( iEvent, missing, hCorrectedBJets,          m_correctedBJets,           kBTagCorrectedJets );
    getCollection( iEvent, missing, hLifetimeBJetsL25,        m_lifetimeBJetsL25,         kBTagLifetimeBJetsL25 );
    getCollection( iEvent, missing, hLifetimeBJetsL3,         m_lifetimeBJetsL3,          kBTagLifetimeBJetsL3 );
    getCollection( iEvent, missing, hSoftmuonBJetsL25,        m_softmuonBJetsL25,         kBTagSoftmuonBJetsL25 );
    getCollection( iEvent, missing, hSoftmuonBJetsL3,         m_softmuonBJetsL3,          kBTagSoftmuonBJetsL3 );
    getCollection( iEvent, missing, hPerformanceBJetsL25,     m_performanceBJetsL25,      kBTagPerformanceBJetsL25 );
    getCollection( iEvent, missing, hPerformanceBJetsL3,      m_performanceBJetsL3,       kBTagPerformanceBJetsL3 );
    getCollection( iEvent, missing, electrons,                Electron_,                  kElectrons );
    getCollection( iEvent, missing, photons,                  Photon_,                    kPhotons );
    
    //Read offline eleID results
    std::vector<edm::Handle<edm::ValueMap<float> > > eIDValueMap(4); 
    //   edm::InputTag electronLabelRobustTight_(std::string("eidRobustTight"));
    //   edm::InputTag electronLabelTight_(std::string("eidTight"));
    //   edm::InputTag electronLabelRobustLoose_(std::string("eidRobustLoose"));
    //   edm::InputTag electronLabelLoose_(std::string("eidLoose"));
    //   getCollection( iEvent, missing, eIDValueMap[0],   electronLabelRobustLoose_      ,       "EleId Robust-Loose");
    //   getCollection( iEvent, missing, eIDValueMap[1],   electronLabelRobustTight_      ,       "EleId Robust-Tight");
    //   getCollection( iEvent, missing, eIDValueMap[2],   electronLabelLoose_      ,       "EleId Loose");
    //   getCollection( iEvent, missing, eIDValueMap[3],   electronLabelTight_      ,       "EleId Tight");
    
    //read all the OpenHLT egamma collections
    getCollection( iEvent, missing, recoIsolecalcands,        CandIso_,                   kCandIso);
    getCollection( iEvent, missing, recoNonIsolecalcands,     CandNonIso_,                kCandNonIso);
    getCollection( iEvent, missing, EcalIsolMap,              EcalIso_,                   kEcalIso);
    getCollection( iEvent, missing, EcalNonIsolMap,           EcalNonIso_,                kEcalNonIso);
    getCollection( iEvent, missing, HcalIsolMap,              HcalIsoPho_,                kHcalIsoPho);
    getCollection( iEvent, missing, HcalNonIsolMap,           HcalNonIsoPho_,             kHcalNonIsoPho);
    getCollection( iEvent, missing, photonR9IsoHandle,        IsoR9_,                     kIsoR9); 
    getCollection( iEvent, missing, photonR9NonIsoHandle,     NonIsoR9_,                  kNonIsoR9);  
    getCollection( iEvent, missing, photonR9IDIsoHandle,      IsoR9ID_,                   kIsoR9ID);
    getCollection( iEvent, missing, photonR9IDNonIsoHandle,   NonIsoR9ID_,                kNonIsoR9ID);
    getCollection( iEvent, missing, photonHoverEHIsoHandle,   IsoHoverEH_,                kIsoHoverEH);    
    getCollection( iEvent, missing, photonHoverEHNonIsoHandle,NonIsoHoverEH_,             kNonIsoHoverEH);   
    getCollection( iEvent, missing, electronIsoHandle,        IsoElectron_,               kIsoElectron);
    getCollection( iEvent, missing, HcalEleIsolMap,           IsoEleHcal_,                kIsoEleHcal);
    getCollection( iEvent, missing, TrackEleIsolMap,          IsoEleTrackIsol_,           kIsoEleTrackIsol);
    getCollection( iEvent, missing, L1IsoPixelSeedsMap,       L1IsoPixelSeeds_,           kL1IsoPixelSeeds);
    getCollection( iEvent, missing, L1NonIsoPixelSeedsMap,    L1NonIsoPixelSeeds_,        kL1NonIsoPixelSeeds);
    getCollection( iEvent, missing, electronNonIsoHandle,     NonIsoElectron_,            kNonIsoElectron);
    getCollection( iEvent, missing, HcalEleNonIsolMap,        NonIsoEleHcal_,             kIsoEleHcal);
    getCollection( iEvent, missing, NonIsoTrackEleIsolMap,    NonIsoEleTrackIsol_,        kNonIsoEleTrackIsol);
    getCollection( iEvent, missing, TrackNonIsolMap,          NonIsoPhoTrackIsol_,        kNonIsoPhoTrackIsol);
    getCollection( iEvent, missing, TrackIsolMap,             IsoPhoTrackIsol_,           kIsoPhoTrackIsol);
    getCollection( iEvent, missing, electronR9IsoHandle,      IsoR9_,                     kIsoR9);  
    getCollection( iEvent, missing, electronR9NonIsoHandle,   NonIsoR9_,                  kNonIsoR9);   
    getCollection( iEvent, missing, electronR9IDIsoHandle,    IsoR9ID_,                   kIsoR9ID);
    getCollection( iEvent, missing, electronR9IDNonIsoHandle, NonIsoR9ID_,                kNonIsoR9ID);
    getCollection( iEvent, missing, electronHFClusterHandle,  HFECALClusters_,            kHFECALClusters); 
    getCollection( iEvent, missing, electronHFElectronHandle, HFElectrons_,               kHFElectrons); 
    getCollection( iEvent, missing, eerechits,                EERecHitTag_,               kEErechits ); 
    getCollection( iEvent, missing, ebrechits,                EBRecHitTag_,               kEBrechits );  
    getCollection( iEvent, missing, pi0eerechits,             pi0EERecHitTag_,            kpi0EErechits );  
    getCollection( iEvent, missing, pi0ebrechits,             pi0EBRecHitTag_,            kpi0EBrechits );   
    getCollection( iEvent, missing, hbherechits,              HBHERecHitTag_,             kHBHErechits );   
    getCollection( iEvent, missing, horechits,                HORecHitTag_,               kHOrechits );   
    getCollection( iEvent, missing, hfrechits,                HFRecHitTag_,               kHFrechits );   
    getCollection( iEvent, missing, isopixeltracksL3,         IsoPixelTrackTagL3_,        kIsoPixelTracksL3 ); 
    getCollection( iEvent, missing, isopixeltracksL2,         IsoPixelTrackTagL2_,        kIsoPixelTracksL2 );
    getCollection( iEvent, missing, isopixeltrackPixVertices, IsoPixelTrackVerticesTag_,   kIsoPixelTrackVertices );
    getCollection( iEvent, missing, pixeltracksL3,            PixelTracksTagL3_,          kPixelTracksL3 ); 
    getCollection( iEvent, missing, recoVertexs,              VertexTag_,                 kRecoVertices ); 
    
  
    
    double ptHat=-1.;
    if (genEventInfo.isValid()) {ptHat=genEventInfo->qScale();}
    
    
    // print missing collections
    if (not missing.empty() and (errCnt < errMax())) {
        errCnt++;
        std::stringstream out;       
        out <<  "OpenHLT analyser - missing collections:";
        BOOST_FOREACH(const MissingCollectionInfo & entry, missing)
        out << "\n\t" << entry.first << ": " << entry.second->encode();
        edm::LogPrint("OpenHLT") << out.str() << std::endl; 
        if (errCnt == errMax())
            edm::LogWarning("OpenHLT") << "Maximum error count reached -- No more messages will be printed.";
    }
    
    // run the analysis, passing required event fragments
    jet_analysis_.analyze(
                          recjets,
                          reccorjets,
                          genjets,
                          recmet,
                          genmet,
                          ht,
                          taus,
                          pftaus,
                          pfjets,
			  recoPftaus,
			  theRecoPFTauDiscrByTanCOnePercent,
			  theRecoPFTauDiscrByTanCHalfPercent,
			  theRecoPFTauDiscrByTanCQuarterPercent,
			  theRecoPFTauDiscrByTanCTenthPercent,
			  theRecoPFTauDiscrByIsolation,
			  theRecoPFTauDiscrAgainstMuon,
			  theRecoPFTauDiscrAgainstElec,
                          recoPFJets, 
                          caloTowers,
                          towerThreshold_,
                          _MinPtGammas,
                          _MinPtChargedHadrons,
                          HltTree);
    
    muon_analysis_.analyze(
                           muon,
                           l1extmu,
                           mucands2,
                           isoMap2,
                           mucands3,
                           isoMap3,
                           oniaPixelCands,
                           oniaTrackCands,
			   munovtxcands2,
                           BSPosition,
                           HltTree);
    
    elm_analysis_.analyze(
                          electrons,
                          photons,
                          electronIsoHandle,
                          electronNonIsoHandle,
                          NonIsoTrackEleIsolMap,
                          TrackEleIsolMap,
                          L1IsoPixelSeedsMap,
                          L1NonIsoPixelSeedsMap,
                          recoIsolecalcands,
                          recoNonIsolecalcands,
                          EcalIsolMap,
                          EcalNonIsolMap,
                          HcalEleIsolMap,
                          HcalEleNonIsolMap,
                          HcalIsolMap,
                          HcalNonIsolMap,
                          TrackIsolMap,
                          TrackNonIsolMap,
                          lazyTools,
                          theMagField,
                          BSPosition,
                          eIDValueMap,
                          photonR9IsoHandle, 
                          photonR9NonIsoHandle, 
                          electronR9IsoHandle, 
                          electronR9NonIsoHandle, 
			  photonHoverEHIsoHandle,  
			  photonHoverEHNonIsoHandle,  
                          photonR9IDIsoHandle,
                          photonR9IDNonIsoHandle,
                          electronR9IDIsoHandle,
                          electronR9IDNonIsoHandle,
			  electronHFClusterHandle,
			  electronHFElectronHandle,
                          HltTree);
    
    mct_analysis_.analyze(
                          mctruth,
                          ptHat,
                          simTracks,
                          simVertices,
                          HltTree);
    
    track_analysis_.analyze( 
                            isopixeltracksL3, 
                            isopixeltracksL2,
                            isopixeltrackPixVertices,			  
                            pixeltracksL3, 
                            HltTree); 
    
    hlt_analysis_.analyze(
                          hltresults,
                          l1extemi,
                          l1extemn,
                          l1extmu,
                          l1extjetc,
                          l1extjetf,
                          l1exttaujet,
                          l1extmet,
                          l1extmht,
                          l1GtRR,
                          gctBitCounts,
                          gctRingSums,
                          iSetup,
                          iEvent,
                          HltTree);
    
    bjet_analysis_.analyze(
                           hRawBJets, 
                           hCorrectedBJets,
                           hLifetimeBJetsL25,
                           hLifetimeBJetsL3,
                           hSoftmuonBJetsL25,
                           hSoftmuonBJetsL3,
                           hPerformanceBJetsL25,
                           hPerformanceBJetsL3,
                           HltTree);
    
    vrt_analysis_.analyze(
                          recoVertexs, 
                          HltTree);
    
    evt_header_.analyze(iEvent, HltTree);
    
    
    // std::cout << " Ending Event Analysis" << std::endl;
    // After analysis, fill the variables tree
    if (m_file)
        m_file->cd();
    HltTree->Fill();
}

// "endJob" is an inherited method that you may implement to do post-EOF processing and produce final output.
void HLTAnalyzer::endJob() {
    
    if (m_file)
        m_file->cd();
    
    const edm::ParameterSet &thepset = edm::getProcessParameterSet();   
    TList *list = HltTree->GetUserInfo();   
    list->Add(new TObjString(thepset.dump().c_str()));   
    
    HltTree->SetWeight(treeWeight);
    HltTree->Write();
    delete HltTree;
    HltTree = 0;
    
    if (m_file) {         // if there was a tree file...
        m_file->Write();    // write out the branches
        delete m_file;      // close and delete the file
        m_file = 0;         // set to zero to clean up
    }
    
}
