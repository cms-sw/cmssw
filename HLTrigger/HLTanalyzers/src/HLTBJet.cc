#include <cmath>
#include <algorithm>
#include <utility>
#include <boost/foreach.hpp>

#include <TTree.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "HLTrigger/HLTanalyzers/interface/HLTBJet.h"

static const size_t kMaxBJets = 10;

#include "HLTMessages.h"

HLTBJet::HLTBJet() 
{
  // set of variables for uncorrected L2 jets
  NohBJetL2                 = 0;
  ohBJetL2Energy            = new float[kMaxBJets];
  ohBJetL2Et                = new float[kMaxBJets];
  ohBJetL2Pt                = new float[kMaxBJets];
  ohBJetL2Eta               = new float[kMaxBJets];
  ohBJetL2Phi               = new float[kMaxBJets];
              
  // set of variables for corrected L2 jets
  NohBJetL2Corrected        = 0;
  ohBJetL2CorrectedEnergy   = new float[kMaxBJets];
  ohBJetL2CorrectedEt       = new float[kMaxBJets];
  ohBJetL2CorrectedPt       = new float[kMaxBJets];
  ohBJetL2CorrectedEta      = new float[kMaxBJets];
  ohBJetL2CorrectedPhi      = new float[kMaxBJets];
  
  // set of variables for lifetime-based b-tag
  ohBJetIPL25Tag            = new float[kMaxBJets];
  ohBJetIPL3Tag             = new float[kMaxBJets];
  
  // set of variables for lifetime-based relaxed b-tag
  ohBJetIPLooseL25Tag       = new float[kMaxBJets];
  ohBJetIPLooseL3Tag        = new float[kMaxBJets];
  
  // set of variables for soft-muon-based b-tag
  ohBJetMuL25Tag            = new int[kMaxBJets];          // do not optimize
  ohBJetMuL3Tag             = new float[kMaxBJets];
  
  // set of variables for b-tagging performance measurements
  ohBJetPerfL25Tag          = new int[kMaxBJets];          // do not optimize 
  ohBJetPerfL3Tag           = new int[kMaxBJets];          // do not optimize
}

void HLTBJet::clear() 
{
  NohBJetL2          = 0;
  NohBJetL2Corrected = 0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2Energy[i]          = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2Et[i]              = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2Et[i]              = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2Pt[i]              = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2Eta[i]             = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2Phi[i]             = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2CorrectedEnergy[i] = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2CorrectedEt[i]     = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2CorrectedPt[i]     = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2CorrectedEta[i]    = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetL2CorrectedPhi[i]    = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetIPL25Tag[i]          = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetIPL3Tag[i]           = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetIPLooseL25Tag[i]     = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetIPLooseL3Tag[i]      = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetMuL25Tag[i]          = 0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetMuL3Tag[i]           = 0.0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetPerfL25Tag[i]        = 0;
  for (size_t i = 0; i < kMaxBJets; ++i) ohBJetPerfL3Tag[i]         = 0;
}

HLTBJet::~HLTBJet() 
{ }

void HLTBJet::setup(const edm::ParameterSet & config, TTree * tree)
{
  m_jets                    = config.getParameter<edm::InputTag>("CommonBJetsL2");
  m_correctedJets           = config.getParameter<edm::InputTag>("CorrectedBJetsL2");
  m_lifetimeBJetsL25        = config.getParameter<edm::InputTag>("LifetimeBJetsL25");
  m_lifetimeBJetsL3         = config.getParameter<edm::InputTag>("LifetimeBJetsL3");
  m_lifetimeBJetsL25Relaxed = config.getParameter<edm::InputTag>("LifetimeBJetsL25Relaxed");
  m_lifetimeBJetsL3Relaxed  = config.getParameter<edm::InputTag>("LifetimeBJetsL3Relaxed");
  m_softmuonBJetsL25        = config.getParameter<edm::InputTag>("SoftmuonBJetsL25");
  m_softmuonBJetsL3         = config.getParameter<edm::InputTag>("SoftmuonBJetsL3");
  m_performanceBJetsL25     = config.getParameter<edm::InputTag>("PerformanceBJetsL25");
  m_performanceBJetsL3      = config.getParameter<edm::InputTag>("PerformanceBJetsL3");
  
  // create the TTree branches
  if (tree) {
    tree->Branch("NohBJetL2",               & NohBJetL2,             "NohBJetL2/I");
    tree->Branch("ohBJetL2Energy",          ohBJetL2Energy,          "ohBJetL2Energy[NohBJetL2]/F");
    tree->Branch("ohBJetL2Et",              ohBJetL2Et,              "ohBJetL2Et[NohBJetL2]/F");
    tree->Branch("ohBJetL2Pt",              ohBJetL2Pt,              "ohBJetL2Pt[NohBJetL2]/F");
    tree->Branch("ohBJetL2Eta",             ohBJetL2Eta,             "ohBJetL2Eta[NohBJetL2]/F");
    tree->Branch("ohBJetL2Phi",             ohBJetL2Phi,             "ohBJetL2Phi[NohBJetL2]/F");
                
    tree->Branch("NohBJetL2Corrected",      & NohBJetL2Corrected,    "NohBJetL2Corrected/I");
    tree->Branch("ohBJetL2CorrectedEnergy", ohBJetL2CorrectedEnergy, "ohBJetL2CorrectedEnergy[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetL2CorrectedEt",     ohBJetL2CorrectedEt,     "ohBJetL2CorrectedEt[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetL2CorrectedPt",     ohBJetL2CorrectedPt,     "ohBJetL2CorrectedPt[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetL2CorrectedEta",    ohBJetL2CorrectedEta,    "ohBJetL2CorrectedEta[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetL2CorrectedPhi",    ohBJetL2CorrectedPhi,    "ohBJetL2CorrectedPhi[NohBJetL2Corrected]/F");
    
    tree->Branch("ohBJetIPL25Tag",          ohBJetIPL25Tag,          "ohBJetIPL25Tag[NohBJetL2]/F");
    tree->Branch("ohBJetIPL3Tag",           ohBJetIPL3Tag,           "ohBJetIPL3Tag[NohBJetL2]/F");
    tree->Branch("ohBJetIPLooseL25Tag",     ohBJetIPLooseL25Tag,     "ohBJetIPLooseL25Tag[NohBJetL2]/F");
    tree->Branch("ohBJetIPLooseL3Tag",      ohBJetIPLooseL3Tag,      "ohBJetIPLooseL3Tag[NohBJetL2]/F");
    tree->Branch("ohBJetMuL25Tag",          ohBJetMuL25Tag,          "ohBJetMuL25Tag[NohBJetL2]/I");
    tree->Branch("ohBJetMuL3Tag",           ohBJetMuL3Tag,           "ohBJetMuL3Tag[NohBJetL2]/F");
    tree->Branch("ohBJetPerfL25Tag",        ohBJetPerfL25Tag,        "ohBJetPerfL25Tag[NohBJetL2]/I");
    tree->Branch("ohBJetPerfL3Tag",         ohBJetPerfL3Tag,         "ohBJetPerfL3Tag[NohBJetL2]/I");
  }
}

void HLTBJet::analyze(const edm::Event & event, const edm::EventSetup & setup, TTree * tree)
{
  // reset the tree variables
  clear();
  
  // read the collections from the Event
  edm::Handle<edm::View<reco::Jet> >  h_jets;
  edm::Handle<edm::View<reco::Jet> >  h_correctedJets;
  edm::Handle<reco::JetTagCollection> h_lifetimeBJetsL25;
  edm::Handle<reco::JetTagCollection> h_lifetimeBJetsL3;
  edm::Handle<reco::JetTagCollection> h_lifetimeBJetsL25Relaxed;
  edm::Handle<reco::JetTagCollection> h_lifetimeBJetsL3Relaxed;
  edm::Handle<reco::JetTagCollection> h_softmuonBJetsL25;
  edm::Handle<reco::JetTagCollection> h_softmuonBJetsL3;
  edm::Handle<reco::JetTagCollection> h_performanceBJetsL25;
  edm::Handle<reco::JetTagCollection> h_performanceBJetsL3;

  event.getByLabel(m_jets,                      h_jets);
  event.getByLabel(m_correctedJets,             h_correctedJets);
  event.getByLabel(m_lifetimeBJetsL25,          h_lifetimeBJetsL25);
  event.getByLabel(m_lifetimeBJetsL3,           h_lifetimeBJetsL3);
  event.getByLabel(m_lifetimeBJetsL25Relaxed,   h_lifetimeBJetsL25Relaxed);
  event.getByLabel(m_lifetimeBJetsL3Relaxed,    h_lifetimeBJetsL3Relaxed);
  event.getByLabel(m_softmuonBJetsL25,          h_softmuonBJetsL25);
  event.getByLabel(m_softmuonBJetsL3,           h_softmuonBJetsL3);
  event.getByLabel(m_performanceBJetsL25,       h_performanceBJetsL25);
  event.getByLabel(m_performanceBJetsL3,        h_performanceBJetsL3);

  typedef std::pair<const char *, const edm::InputTag *> MissingCollectionInfo;
  std::vector<MissingCollectionInfo> missing;
  if (not h_jets.isValid())                     missing.push_back(std::make_pair(kBTagJets,                     & m_jets));
  if (not h_correctedJets.isValid())            missing.push_back(std::make_pair(kBTagCorrectedJets,            & m_correctedJets));
  if (not h_lifetimeBJetsL25.isValid())         missing.push_back(std::make_pair(kBTagLifetimeBJetsL25,         & m_lifetimeBJetsL25));
  if (not h_lifetimeBJetsL3.isValid())          missing.push_back(std::make_pair(kBTagLifetimeBJetsL3,          & m_lifetimeBJetsL3));
  if (not h_lifetimeBJetsL25Relaxed.isValid())  missing.push_back(std::make_pair(kBTagLifetimeBJetsL25Relaxed,  & m_lifetimeBJetsL25Relaxed));
  if (not h_lifetimeBJetsL3Relaxed.isValid())   missing.push_back(std::make_pair(kBTagLifetimeBJetsL3Relaxed,   & m_lifetimeBJetsL3Relaxed));
  if (not h_softmuonBJetsL25.isValid())         missing.push_back(std::make_pair(kBTagSoftmuonBJetsL25,         & m_softmuonBJetsL25));
  if (not h_softmuonBJetsL3.isValid())          missing.push_back(std::make_pair(kBTagSoftmuonBJetsL3,          & m_softmuonBJetsL3));
  if (not h_performanceBJetsL25.isValid())      missing.push_back(std::make_pair(kBTagPerformanceBJetsL25,      & m_performanceBJetsL25));
  if (not h_performanceBJetsL3.isValid())       missing.push_back(std::make_pair(kBTagPerformanceBJetsL3,       & m_performanceBJetsL3));
  if (not missing.empty()) {
    std::stringstream out;
    out <<  "BJet OpenHLT producer - missing collections:";
    BOOST_FOREACH(const MissingCollectionInfo & entry, missing)
      out << "\n\t" << entry.first << ' ' << entry.second->encode();
    edm::LogPrint("OpenHLT") << out.str() << std::endl;
  }

  // if the required collections are available, fill the corresponding tree branches
  if (h_jets.isValid())
    analyseJets(* h_jets);

  if (h_correctedJets.isValid())
    analyseCorrectedJets(* h_correctedJets);
 
  if (h_jets.isValid() and h_lifetimeBJetsL25.isValid() and h_lifetimeBJetsL3.isValid())
    analyseLifetime(* h_jets, * h_lifetimeBJetsL25, * h_lifetimeBJetsL3);

  if (h_jets.isValid() and h_lifetimeBJetsL25Relaxed.isValid() and h_lifetimeBJetsL3Relaxed.isValid())
    analyseLifetimeLoose(* h_jets, * h_lifetimeBJetsL25Relaxed, * h_lifetimeBJetsL3Relaxed);

  if (h_jets.isValid() and h_softmuonBJetsL25.isValid() and h_softmuonBJetsL3.isValid())
    analyseSoftmuon(* h_jets, * h_softmuonBJetsL25, * h_softmuonBJetsL3);
  
  if (h_jets.isValid() and h_performanceBJetsL25.isValid() and h_performanceBJetsL3.isValid())
    analysePerformance(* h_jets, * h_performanceBJetsL25, * h_performanceBJetsL3);
}

void HLTBJet::analyseJets(const edm::View<reco::Jet> & jets)
{
  size_t size = std::min(kMaxBJets, jets.size());
  NohBJetL2 = size;
  for (size_t i = 0; i < size; ++i) {
    ohBJetL2Energy[i] = jets[i].energy();
    ohBJetL2Et[i]     = jets[i].et();
    ohBJetL2Pt[i]     = jets[i].pt();
    ohBJetL2Eta[i]    = jets[i].eta();
    ohBJetL2Phi[i]    = jets[i].phi();
  }
}

void HLTBJet::analyseCorrectedJets(const edm::View<reco::Jet> & jets)
{
  size_t size = std::min(kMaxBJets, jets.size());
  NohBJetL2Corrected = size;
  for (size_t i = 0; i < size; ++i) {
    ohBJetL2CorrectedEnergy[i] = jets[i].energy();
    ohBJetL2CorrectedEt[i]     = jets[i].et();
    ohBJetL2CorrectedPt[i]     = jets[i].pt();
    ohBJetL2CorrectedEta[i]    = jets[i].eta();
    ohBJetL2CorrectedPhi[i]    = jets[i].phi();
  }
}

void HLTBJet::analyseLifetime(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL25 << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL3 << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  size_t size = std::min(kMaxBJets, jets.size());
  for (size_t i = 0; i < size; i++) {
    ohBJetIPL25Tag[i] = tagsL25[i].second;
    ohBJetIPL3Tag[i]  = tagsL3[i].second;
  }
}

void HLTBJet::analyseLifetimeLoose(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL25Relaxed << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL3Relaxed << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  size_t size = std::min(kMaxBJets, jets.size());
  for (size_t i = 0; i < size; i++) {
    ohBJetIPLooseL25Tag[i] = tagsL25[i].second;
    ohBJetIPLooseL3Tag[i]  = tagsL3[i].second;
  }
}

void HLTBJet::analyseSoftmuon(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagSoftmuonBJetsL25 << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagSoftmuonBJetsL3 << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  size_t size = std::min(kMaxBJets, jets.size());
  for (size_t i = 0; i < size; i++) {
    ohBJetMuL25Tag[i] = (tagsL25[i].second > 0.) ? 1 : 0;
    ohBJetMuL3Tag[i]  = tagsL3[i].second;
  }
}

void HLTBJet::analysePerformance(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagPerformanceBJetsL25 << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagPerformanceBJetsL3 << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  size_t size = std::min(kMaxBJets, jets.size());
  for (size_t i = 0; i < size; i++) {
    ohBJetPerfL25Tag[i] = (tagsL25[i].second > 0.) ? 1 : 0;
    ohBJetPerfL3Tag[i]  = (tagsL3[i].second  > 0.) ? 1 : 0;
  }
}
