#include "DQM/DataScouting/plugins/RazorVarAnalyzer.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

#include <cmath>

// A simple constructor which takes as inoput only the name of the PF jet collection
RazorVarAnalyzer::RazorVarAnalyzer( const edm::ParameterSet & conf ):
  ScoutingAnalyzerBase(conf),
  m_eleCollectionTag(conf.getUntrackedParameter<edm::InputTag>("eleCollectionName",edm::InputTag("hltPixelMatchElectronsActivity"))),
  m_jetCollectionTag(conf.getUntrackedParameter<edm::InputTag>("jetCollectionName",edm::InputTag("hltCaloJetIDPassed"))),
  m_muCollectionTag(conf.getUntrackedParameter<edm::InputTag>("muCollectionName",edm::InputTag("hltL3MuonCandidates"))),
  m_razorVarCollectionTag(conf.getUntrackedParameter<edm::InputTag>("razorVarCollectionName")) {
    //set Token(-s)
    m_jetCollectionTagToken_ = consumes<reco::CaloJetCollection>(conf.getUntrackedParameter<edm::InputTag>("jetCollectionName",edm::InputTag("hltCaloJetIDPassed")));
    m_muCollectionTagToken_ = consumes<std::vector<reco::RecoChargedCandidate> >(conf.getUntrackedParameter<edm::InputTag>("muCollectionName",edm::InputTag("hltL3MuonCandidates")));
    m_eleCollectionTagToken_ = consumes<reco::ElectronCollection>(conf.getUntrackedParameter<edm::InputTag>("eleCollectionName",edm::InputTag("hltPixelMatchElectronsActivity")));
    m_razorVarCollectionTagToken_ = consumes<std::vector<double> >(conf.getUntrackedParameter<edm::InputTag>("razorVarCollectionName"));
}

RazorVarAnalyzer::~RazorVarAnalyzer() {}

// Function to book the Monitoring Elements.
void RazorVarAnalyzer::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const &, edm::EventSetup const &) {
  ScoutingAnalyzerBase::prepareBooking(iBooker);
  //the full inclusive histograms
  m_rsqMRFullyInc = bookH2withSumw2(
      iBooker,
      "rsqMRFullyInc",
      "M_{R} vs R^{2} (All Events)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRInc4J = bookH2withSumw2(
      iBooker,
      "rsqMRInc4J",
      "M_{R} vs R^{2} (>= 4j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRInc6J = bookH2withSumw2(
      iBooker,
      "rsqMRInc6J",
      "M_{R} vs R^{2} (>= 6j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRInc8J = bookH2withSumw2(
      iBooker,
      "rsqMRInc8J",
      "M_{R} vs R^{2} (>= 8j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRInc10J = bookH2withSumw2(
      iBooker,
      "rsqMRInc10J",
      "M_{R} vs R^{2} (>= 10j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRInc12J = bookH2withSumw2(
      iBooker,
      "rsqMRInc12J",
      "M_{R} vs R^{2} (>= 12j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRInc14J = bookH2withSumw2(
      iBooker,
      "rsqMRInc14J",
      "M_{R} vs R^{2} (>= 14j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");

  //the by box histograms
  m_rsqMREleMu = bookH2withSumw2(
      iBooker,
      "rsqMREleMu",
      "M_{R} vs R^{2} (EleMu box)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRMuMu = bookH2withSumw2(
      iBooker,
      "rsqMRMuMu",
      "M_{R} vs R^{2} (MuMu box)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMREleEle = bookH2withSumw2(
      iBooker,
      "rsqMREleEle",
      "M_{R} vs R^{2} (EleEle box)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRMu = bookH2withSumw2(
      iBooker,
      "rsqMRMu",
      "M_{R} vs R^{2} (Mu box)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMREle = bookH2withSumw2(
      iBooker,
      "rsqMREle",
      "M_{R} vs R^{2} (Ele box)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRHad = bookH2withSumw2(
      iBooker,
      "rsqMRHad",
      "M_{R} vs R^{2} (Had box)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");

  //the by box histograms
  m_rsqMRMuMJ = bookH2withSumw2(
      iBooker,
      "rsqMRMuMJ",
      "M_{R} vs R^{2} (Mu box >= 4j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMREleMJ = bookH2withSumw2(
      iBooker,
      "rsqMREleMJ",
      "M_{R} vs R^{2} (Ele box >= 5j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
  m_rsqMRHadMJ = bookH2withSumw2(
      iBooker,
      "rsqMRHadMJ",
      "M_{R} vs R^{2} (Had box >= 6j)",
      400,0.,4000.,
      50,0.,1.,
      "M_{R} [GeV]",
      "R^{2}");
}

// Usual analyze method
void RazorVarAnalyzer::analyze( const edm::Event & iEvent, const edm::EventSetup & c ) {
  //count the number of jets with a minimal selection
  edm::Handle<reco::CaloJetCollection> calojets_handle;
  iEvent.getByToken(m_jetCollectionTagToken_, calojets_handle);

  unsigned int njets = 0;
  for(reco::CaloJetCollection::const_iterator it = calojets_handle->begin(); it != calojets_handle->end(); ++it){
    if(it->pt() >= 30. && fabs(it->eta()) <= 3.0){
      njets++;
    }
  }

  //count the number of muons
  edm::Handle<std::vector<reco::RecoChargedCandidate> > muon_handle;
  iEvent.getByToken(m_muCollectionTagToken_, muon_handle);

  unsigned int nmu_loose = 0;
  unsigned int nmu_tight = 0;
  if(muon_handle.isValid()){
    for(std::vector<reco::RecoChargedCandidate>::const_iterator it = muon_handle->begin(); it != muon_handle->end(); ++it){
      if(it->pt() >= 15 && fabs(it->eta()) <= 2.1) nmu_tight++;
      if(it->pt() >= 10 && fabs(it->eta()) <= 2.4) nmu_loose++;
    }
  }

  //count the number of electrons
  edm::Handle<reco::ElectronCollection> ele_handle;
  iEvent.getByToken(m_eleCollectionTagToken_, ele_handle);

  unsigned int nele_loose = 0;
  unsigned int nele_tight = 0;
  if(ele_handle.isValid()){
    for(reco::ElectronCollection::const_iterator it = ele_handle->begin(); it != ele_handle->end(); ++it){
      if(it->pt() >= 20 && fabs(it->eta()) <= 2.5) nele_tight++;
      if(it->pt() >= 10 && fabs(it->eta()) <= 2.5) nele_loose++;
    }
  }

  //now get the box number: {'MuEle':0,'MuMu':1,'EleEle':2,'Mu':3,'Ele':4,'Had':5}
  unsigned int box_num = 5;
  if(nmu_tight > 0 && nele_tight > 0){
    box_num = 0;
  }else if(nmu_tight > 0 && nmu_loose > 1){
    box_num = 1;
  }else if(nele_tight > 0 && nele_loose > 1){
    box_num = 2;
  }else if(nmu_tight > 0){
    box_num = 3;
  }else if(nele_tight > 0){
    box_num = 4;
  }

  edm::Handle<std::vector<double> > razorvar_handle;
  iEvent.getByToken(m_razorVarCollectionTagToken_, razorvar_handle);
  if(razorvar_handle->size() > 1){
    const double MR = razorvar_handle->at(0);
    const double R = razorvar_handle->at(1);
    m_rsqMRFullyInc->Fill(MR,R*R);
    if(njets >= 4) m_rsqMRInc4J->Fill(MR,R*R);
    if(njets >= 6) m_rsqMRInc6J->Fill(MR,R*R);
    if(njets >= 8) m_rsqMRInc8J->Fill(MR,R*R);
    if(njets >= 10) m_rsqMRInc10J->Fill(MR,R*R);
    if(njets >= 12) m_rsqMRInc12J->Fill(MR,R*R);
    if(njets >= 14) m_rsqMRInc14J->Fill(MR,R*R);

    //now fill the boxes
    if(box_num == 0) m_rsqMREleMu->Fill(MR,R*R);
    if(box_num == 1) m_rsqMRMuMu->Fill(MR,R*R);
    if(box_num == 2) m_rsqMREleEle->Fill(MR,R*R);
    if(box_num == 3) m_rsqMRMu->Fill(MR,R*R);
    if(box_num == 4) m_rsqMREle->Fill(MR,R*R);
    if(box_num == 5) m_rsqMRHad->Fill(MR,R*R);

    //finally the multijet boxes - think ttbar
    //muon boxes: muons are not in jets
    if( box_num == 3 && njets >= 4) m_rsqMRMuMJ->Fill(MR,R*R);
    //ele boxes: electrons are in jets
    else if( box_num == 4 && njets >= 5) m_rsqMREleMJ->Fill(MR,R*R);
    //fill the Had box
    else if( box_num == 5 && njets >= 6) m_rsqMRHadMJ->Fill(MR,R*R);
  }
} 
