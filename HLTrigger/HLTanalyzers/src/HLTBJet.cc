#include <cmath>
#include <algorithm>
#include <utility>
#include <cstring>
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
  
  // set of variables for corrected L2 jets L1FastJet
  NohBJetL2CorrectedL1FastJet        = 0;
  ohBJetL2CorrectedEnergyL1FastJet   = new float[kMaxBJets];
  ohBJetL2CorrectedEtL1FastJet       = new float[kMaxBJets];
  ohBJetL2CorrectedPtL1FastJet       = new float[kMaxBJets];
  ohBJetL2CorrectedEtaL1FastJet      = new float[kMaxBJets];
  ohBJetL2CorrectedPhiL1FastJet      = new float[kMaxBJets];
  
  // set of variables for uncorrected L2 jets
  NohpfBJetL2                 = 0;
  ohpfBJetL2Energy            = new float[kMaxBJets];
  ohpfBJetL2Et                = new float[kMaxBJets];
  ohpfBJetL2Pt                = new float[kMaxBJets];
  ohpfBJetL2Eta               = new float[kMaxBJets];
  ohpfBJetL2Phi               = new float[kMaxBJets];
              
  // set of variables for lifetime-based b-tag
  ohBJetIPL25Tag            = new float[kMaxBJets];
  ohBJetIPL3Tag             = new float[kMaxBJets];

  // set of variables for lifetime-based b-tag L1FastJet
  ohBJetIPL25TagL1FastJet           = new float[kMaxBJets];
  ohBJetIPL3TagL1FastJet             = new float[kMaxBJets];

  // set of variables for lifetime-based b-tag PF jets
  ohpfBJetIPL3Tag             = new float[kMaxBJets];

  // set of variables for lifetime-based b-tag Single Track
  ohBJetIPL25TagSingleTrack = new float[kMaxBJets];
  ohBJetIPL3TagSingleTrack  = new float[kMaxBJets];
  ohBJetIPL25TagSingleTrackL1FastJet = new float[kMaxBJets];
  ohBJetIPL3TagSingleTrackL1FastJet  = new float[kMaxBJets];
  
  // set of variables for b-tagging performance measurements
  // SoftMuonbyDR
  ohBJetPerfL25Tag          = new int[kMaxBJets];          // do not optimize 
  ohBJetPerfL3Tag           = new int[kMaxBJets];          // do not optimize
  // set of variables for b-tagging performance measurements L1FastJet
  // SoftMuonbyDR
  ohBJetPerfL25TagL1FastJet          = new int[kMaxBJets];          // do not optimize 
  ohBJetPerfL3TagL1FastJet           = new int[kMaxBJets];          // do not optimize
}

void HLTBJet::clear() 
{
  NohBJetL2          = 0;
  NohBJetL2Corrected = 0;
  std::memset(ohBJetL2Energy,            '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2Et,                '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2Et,                '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2Pt,                '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2Eta,               '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2Phi,               '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedEnergy,   '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedEt,       '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedPt,       '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedEta,      '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedPhi,      '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedEnergyL1FastJet, '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedEtL1FastJet,     '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedPtL1FastJet,     '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedEtaL1FastJet,    '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetL2CorrectedPhiL1FastJet,    '\0', kMaxBJets * sizeof(float));
  std::memset(ohpfBJetL2Energy,            '\0', kMaxBJets * sizeof(float));
  std::memset(ohpfBJetL2Et,                '\0', kMaxBJets * sizeof(float));
  std::memset(ohpfBJetL2Et,                '\0', kMaxBJets * sizeof(float));
  std::memset(ohpfBJetL2Pt,                '\0', kMaxBJets * sizeof(float));
  std::memset(ohpfBJetL2Eta,               '\0', kMaxBJets * sizeof(float));
  std::memset(ohpfBJetL2Phi,               '\0', kMaxBJets * sizeof(float));

  std::memset(ohBJetIPL25Tag,            '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL3Tag,             '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL25TagL1FastJet,   '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL3TagL1FastJet,    '\0', kMaxBJets * sizeof(float));
  std::memset(ohpfBJetIPL3Tag,           '\0', kMaxBJets * sizeof(float));

  std::memset(ohBJetIPL25TagSingleTrack, '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL3TagSingleTrack,  '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL25TagSingleTrackL1FastJet, '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL3TagSingleTrackL1FastJet,  '\0', kMaxBJets * sizeof(float));

  std::memset(ohBJetPerfL25Tag,          '\0', kMaxBJets * sizeof(int));
  std::memset(ohBJetPerfL3Tag,           '\0', kMaxBJets * sizeof(int));
  std::memset(ohBJetPerfL25TagL1FastJet, '\0', kMaxBJets * sizeof(int));
  std::memset(ohBJetPerfL3TagL1FastJet,  '\0', kMaxBJets * sizeof(int));
}

HLTBJet::~HLTBJet() 
{ 
  delete[] ohBJetL2Energy;
  delete[] ohBJetL2Et;
  delete[] ohBJetL2Pt;
  delete[] ohBJetL2Eta;
  delete[] ohBJetL2Phi;
  delete[] ohBJetL2CorrectedEnergy;
  delete[] ohBJetL2CorrectedEt;
  delete[] ohBJetL2CorrectedPt;
  delete[] ohBJetL2CorrectedEta;
  delete[] ohBJetL2CorrectedPhi;
  delete[] ohBJetL2CorrectedEnergyL1FastJet;
  delete[] ohBJetL2CorrectedEtL1FastJet;
  delete[] ohBJetL2CorrectedPtL1FastJet;
  delete[] ohBJetL2CorrectedEtaL1FastJet;
  delete[] ohBJetL2CorrectedPhiL1FastJet;
  delete[] ohpfBJetL2Energy;
  delete[] ohpfBJetL2Et;
  delete[] ohpfBJetL2Pt;
  delete[] ohpfBJetL2Eta;
  delete[] ohpfBJetL2Phi;
  delete[] ohBJetIPL25Tag;
  delete[] ohBJetIPL3Tag;
  delete[] ohBJetIPL25TagL1FastJet;
  delete[] ohBJetIPL3TagL1FastJet;
  delete[] ohpfBJetIPL3Tag;
  delete[] ohBJetIPL25TagSingleTrack;
  delete[] ohBJetIPL3TagSingleTrack;
  delete[] ohBJetIPL25TagSingleTrackL1FastJet;
  delete[] ohBJetIPL3TagSingleTrackL1FastJet;
  delete[] ohBJetPerfL25Tag;
  delete[] ohBJetPerfL3Tag;
  delete[] ohBJetPerfL25TagL1FastJet;
  delete[] ohBJetPerfL3TagL1FastJet;
}

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
    
    tree->Branch("NohBJetL2CorrectedL1FastJet",      & NohBJetL2CorrectedL1FastJet,    "NohBJetL2CorrectedL1FastJet/I");
    tree->Branch("ohBJetL2CorrectedEnergyL1FastJet", ohBJetL2CorrectedEnergyL1FastJet, "ohBJetL2CorrectedEnergyL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    tree->Branch("ohBJetL2CorrectedEtL1FastJet",     ohBJetL2CorrectedEtL1FastJet,     "ohBJetL2CorrectedEtL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    tree->Branch("ohBJetL2CorrectedPtL1FastJet",     ohBJetL2CorrectedPtL1FastJet,     "ohBJetL2CorrectedPtL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    tree->Branch("ohBJetL2CorrectedEtaL1FastJet",    ohBJetL2CorrectedEtaL1FastJet,    "ohBJetL2CorrectedEtaL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    tree->Branch("ohBJetL2CorrectedPhiL1FastJet",    ohBJetL2CorrectedPhiL1FastJet,    "ohBJetL2CorrectedPhiL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    
    tree->Branch("NohpfBJetL2",      & NohpfBJetL2,    "NohpfBJetL2/I");
    tree->Branch("ohpfBJetL2Energy", ohpfBJetL2Energy, "ohpfBJetL2Energy[NohpfBJetL2]/F");
    tree->Branch("ohpfBJetL2Et",     ohpfBJetL2Et,     "ohpfBJetL2Et[NohpfBJetL2]/F");
    tree->Branch("ohpfBJetL2Pt",     ohpfBJetL2Pt,     "ohpfBJetL2Pt[NohpfBJetL2]/F");
    tree->Branch("ohpfBJetL2Eta",    ohpfBJetL2Eta,    "ohpfBJetL2Eta[NohpfBJetL2]/F");
    tree->Branch("ohpfBJetL2Phi",    ohpfBJetL2Phi,    "ohpfBJetL2Phi[NohpfBJetL2]/F");
                
    tree->Branch("ohBJetIPL25Tag",          ohBJetIPL25Tag,          "ohBJetIPL25Tag[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetIPL3Tag",           ohBJetIPL3Tag,           "ohBJetIPL3Tag[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetIPL25TagL1FastJet", ohBJetIPL25TagL1FastJet, "ohBJetIPL25TagL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    tree->Branch("ohBJetIPL3TagL1FastJet",  ohBJetIPL3TagL1FastJet,  "ohBJetIPL3TagL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    tree->Branch("ohpfBJetIPL3Tag",         ohpfBJetIPL3Tag,         "ohpfBJetIPL3Tag[NohpfBJetL2]/F");

    tree->Branch("ohBJetIPL25TagSingleTrack", ohBJetIPL25TagSingleTrack,      "ohBJetIPL25TagSingleTrack[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetIPL3TagSingleTrack",  ohBJetIPL3TagSingleTrack,       "ohBJetIPL3TagSingleTrack[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetIPL25TagSingleTrackL1FastJet", ohBJetIPL25TagSingleTrackL1FastJet,      "ohBJetIPL25TagSingleTrackL1FastJet[NohBJetL2CorrectedL1FastJet]/F");
    tree->Branch("ohBJetIPL3TagSingleTrackL1FastJet",  ohBJetIPL3TagSingleTrackL1FastJet,       "ohBJetIPL3TagSingleTrackL1FastJet[NohBJetL2CorrectedL1FastJet]/F");

    tree->Branch("ohBJetPerfL25Tag",        ohBJetPerfL25Tag,        "ohBJetPerfL25Tag[NohBJetL2Corrected]/I");
    tree->Branch("ohBJetPerfL3Tag",         ohBJetPerfL3Tag,         "ohBJetPerfL3Tag[NohBJetL2Corrected]/I");
    tree->Branch("ohBJetPerfL25TagL1FastJet", ohBJetPerfL25TagL1FastJet, "ohBJetPerfL25TagL1FastJet[NohBJetL2CorrectedL1FastJet]/I");
    tree->Branch("ohBJetPerfL3TagL1FastJet",  ohBJetPerfL3TagL1FastJet,  "ohBJetPerfL3TagL1FastJet[NohBJetL2CorrectedL1FastJet]/I");
  }
}

void HLTBJet::analyze(
        const edm::Handle<edm::View<reco::Jet> >  & rawBJets,
        const edm::Handle<edm::View<reco::Jet> >  & correctedBJets,
        const edm::Handle<edm::View<reco::Jet> >  & correctedBJetsL1FastJet,
        const edm::Handle<edm::View<reco::Jet> >  & pfBJets,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25L1FastJet,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3L1FastJet,
        const edm::Handle<reco::JetTagCollection> & lifetimePFBJetsL3,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25SingleTrack,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3SingleTrack,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25SingleTrackL1FastJet,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3SingleTrackL1FastJet,
        const edm::Handle<reco::JetTagCollection> & performanceBJetsL25,
        const edm::Handle<reco::JetTagCollection> & performanceBJetsL3,
        const edm::Handle<reco::JetTagCollection> & performanceBJetsL25L1FastJet,
        const edm::Handle<reco::JetTagCollection> & performanceBJetsL3L1FastJet,
        TTree * tree) 
{
  // reset the tree variables
  clear();
  
  // if the required collections are available, fill the corresponding tree branches
  if (rawBJets.isValid())
    analyseJets(* rawBJets);

  if (correctedBJets.isValid())
    analyseCorrectedJets(* correctedBJets);
 
  if (correctedBJetsL1FastJet.isValid())
    analyseCorrectedJetsL1FastJet(* correctedBJetsL1FastJet);
 
  if (pfBJets.isValid())
    analysePFJets(* pfBJets);

  if (correctedBJets.isValid() and lifetimeBJetsL25.isValid() and lifetimeBJetsL3.isValid())
    analyseLifetime(* correctedBJets, * lifetimeBJetsL25, * lifetimeBJetsL3);

  if (correctedBJetsL1FastJet.isValid() and lifetimeBJetsL25L1FastJet.isValid() and lifetimeBJetsL3L1FastJet.isValid())
    analyseLifetimeL1FastJet(* correctedBJetsL1FastJet, * lifetimeBJetsL25L1FastJet, * lifetimeBJetsL3L1FastJet);

  if (pfBJets.isValid() and lifetimePFBJetsL3.isValid())
    analyseLifetimePF(* pfBJets, * lifetimePFBJetsL3);

  if (correctedBJets.isValid() and lifetimeBJetsL25SingleTrack.isValid() and lifetimeBJetsL3SingleTrack.isValid())
    analyseLifetimeSingleTrack(* correctedBJets, * lifetimeBJetsL25SingleTrack, * lifetimeBJetsL3SingleTrack);

  if (correctedBJetsL1FastJet.isValid() and lifetimeBJetsL25SingleTrackL1FastJet.isValid() and lifetimeBJetsL3SingleTrackL1FastJet.isValid())
    analyseLifetimeSingleTrackL1FastJet(* correctedBJetsL1FastJet, * lifetimeBJetsL25SingleTrackL1FastJet, * lifetimeBJetsL3SingleTrackL1FastJet);

  if (correctedBJets.isValid() and performanceBJetsL25.isValid() and performanceBJetsL3.isValid())
    analysePerformance(* correctedBJets, * performanceBJetsL25, * performanceBJetsL3);

  if (correctedBJetsL1FastJet.isValid() and performanceBJetsL25L1FastJet.isValid() and performanceBJetsL3L1FastJet.isValid())
    analysePerformanceL1FastJet(* correctedBJetsL1FastJet, * performanceBJetsL25L1FastJet, * performanceBJetsL3L1FastJet);

}

void HLTBJet::analyseJets(const edm::View<reco::Jet> & jets)
{
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) ); 
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
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  NohBJetL2Corrected = size;
  for (size_t i = 0; i < size; ++i) {
    ohBJetL2CorrectedEnergy[i] = jets[i].energy();
    ohBJetL2CorrectedEt[i]     = jets[i].et();
    ohBJetL2CorrectedPt[i]     = jets[i].pt();
    ohBJetL2CorrectedEta[i]    = jets[i].eta();
    ohBJetL2CorrectedPhi[i]    = jets[i].phi();
  }
}

void HLTBJet::analyseCorrectedJetsL1FastJet(const edm::View<reco::Jet> & jets)
{
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  NohBJetL2CorrectedL1FastJet = size;
  for (size_t i = 0; i < size; ++i) {
    ohBJetL2CorrectedEnergyL1FastJet[i] = jets[i].energy();
    ohBJetL2CorrectedEtL1FastJet[i]     = jets[i].et();
    ohBJetL2CorrectedPtL1FastJet[i]     = jets[i].pt();
    ohBJetL2CorrectedEtaL1FastJet[i]    = jets[i].eta();
    ohBJetL2CorrectedPhiL1FastJet[i]    = jets[i].phi();
  }
}

void HLTBJet::analysePFJets(const edm::View<reco::Jet> & jets)
{
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) ); 
  NohpfBJetL2 = size;
  for (size_t i = 0; i < size; ++i) {
    ohpfBJetL2Energy[i] = jets[i].energy();
    ohpfBJetL2Et[i]     = jets[i].et();
    ohpfBJetL2Pt[i]     = jets[i].pt();
    ohpfBJetL2Eta[i]    = jets[i].eta();
    ohpfBJetL2Phi[i]    = jets[i].phi();
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
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  for (size_t i = 0; i < size; i++) {
    ohBJetIPL25Tag[i] = tagsL25[i].second;
    ohBJetIPL3Tag[i]  = tagsL3[i].second;
  }
}

void HLTBJet::analyseLifetimeL1FastJet(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL25L1FastJet << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL3L1FastJet << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  for (size_t i = 0; i < size; i++) {
    ohBJetIPL25TagL1FastJet[i] = tagsL25[i].second;
    ohBJetIPL3TagL1FastJet[i]  = tagsL3[i].second;
  }
}

void HLTBJet::analyseLifetimePF(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimePFBJetsL3 << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  for (size_t i = 0; i < size; i++) {
    ohpfBJetIPL3Tag[i]  = tagsL3[i].second;
  }
}

void HLTBJet::analyseLifetimeSingleTrack(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL25SingleTrack << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL3SingleTrack << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  for (size_t i = 0; i < size; i++) {
    ohBJetIPL25TagSingleTrack[i] = tagsL25[i].second;
    ohBJetIPL3TagSingleTrack[i]  = tagsL3[i].second;
  }
}

void HLTBJet::analyseLifetimeSingleTrackL1FastJet(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL25SingleTrackL1FastJet << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagLifetimeBJetsL3SingleTrackL1FastJet << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  for (size_t i = 0; i < size; i++) {
    ohBJetIPL25TagSingleTrackL1FastJet[i] = tagsL25[i].second;
    ohBJetIPL3TagSingleTrackL1FastJet[i]  = tagsL3[i].second;
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
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  for (size_t i = 0; i < size; i++) {
    ohBJetPerfL25Tag[i] = (tagsL25[i].second > 0.) ? 1 : 0;
    ohBJetPerfL3Tag[i]  = (tagsL3[i].second  > 0.) ? 1 : 0;
  }
}

void HLTBJet::analysePerformanceL1FastJet(
    const edm::View<reco::Jet>   & jets, 
    const reco::JetTagCollection & tagsL25, 
    const reco::JetTagCollection & tagsL3)
{
  if (tagsL25.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagPerformanceBJetsL25L1FastJet << " collection has " << tagsL25.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  if (tagsL3.size() != jets.size()) {
    edm::LogWarning("OpenHLT") << kBTagPerformanceBJetsL3L1FastJet << " collection has " << tagsL3.size() << " elements, but " << jets.size() << " where expected from L2" << std::endl;
    return;
  }
  // the jets need to be persistable, so .size() returns an 'unsigned int' to be stable across the architectures
  // so, for the comparison, we cast back to size_t
  size_t size = std::min(kMaxBJets, size_t(jets.size()) );
  for (size_t i = 0; i < size; i++) {
     
    ohBJetPerfL25TagL1FastJet[i] = (tagsL25[i].second > 0.) ? 1 : 0;
    ohBJetPerfL3TagL1FastJet[i]  = (tagsL3[i].second  > 0.) ? 1 : 0;
  }
}
