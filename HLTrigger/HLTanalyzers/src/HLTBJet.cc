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

void HLTBJet::analyze(
        const edm::View<reco::Jet> *   rawBJets,
        const edm::View<reco::Jet> *   correctedBJets,
        const reco::JetTagCollection * lifetimeBJetsL25,
        const reco::JetTagCollection * lifetimeBJetsL3,
        const reco::JetTagCollection * lifetimeBJetsL25Relaxed,
        const reco::JetTagCollection * lifetimeBJetsL3Relaxed,
        const reco::JetTagCollection * softmuonBJetsL25,
        const reco::JetTagCollection * softmuonBJetsL3,
        const reco::JetTagCollection * performanceBJetsL25,
        const reco::JetTagCollection * performanceBJetsL3,
        TTree * tree) 
{
  // reset the tree variables
  clear();
  
  // if the required collections are available, fill the corresponding tree branches
  if (rawBJets)
    analyseJets(* rawBJets);

  if (correctedBJets)
    analyseCorrectedJets(* correctedBJets);
 
  if (rawBJets and lifetimeBJetsL25 and lifetimeBJetsL3)
    analyseLifetime(* rawBJets, * lifetimeBJetsL25, * lifetimeBJetsL3);

  if (rawBJets and lifetimeBJetsL25Relaxed and lifetimeBJetsL3Relaxed)
    analyseLifetimeLoose(* rawBJets, * lifetimeBJetsL25Relaxed, * lifetimeBJetsL3Relaxed);

  if (rawBJets and softmuonBJetsL25 and softmuonBJetsL3)
    analyseSoftmuon(* rawBJets, * softmuonBJetsL25, * softmuonBJetsL3);
  
  if (rawBJets and performanceBJetsL25 and performanceBJetsL3)
    analysePerformance(* rawBJets, * performanceBJetsL25, * performanceBJetsL3);
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
