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
  
  // set of variables for lifetime-based b-tag
  ohBJetIPL25Tag            = new float[kMaxBJets];
  ohBJetIPL3Tag             = new float[kMaxBJets];

  // set of variables for lifetime-based b-tag Single Track
  ohBJetIPL25TagSingleTrack = new float[kMaxBJets];
  ohBJetIPL3TagSingleTrack  = new float[kMaxBJets];
  
  // set of variables for b-tagging performance measurements
  // SoftMuonbyDR
  ohBJetPerfL25Tag          = new int[kMaxBJets];          // do not optimize 
  ohBJetPerfL3Tag           = new int[kMaxBJets];          // do not optimize
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

  std::memset(ohBJetIPL25Tag,            '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL3Tag,             '\0', kMaxBJets * sizeof(float));

  std::memset(ohBJetIPL25TagSingleTrack, '\0', kMaxBJets * sizeof(float));
  std::memset(ohBJetIPL3TagSingleTrack,  '\0', kMaxBJets * sizeof(float));

  std::memset(ohBJetPerfL25Tag,          '\0', kMaxBJets * sizeof(int));
  std::memset(ohBJetPerfL3Tag,           '\0', kMaxBJets * sizeof(int));
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
  delete[] ohBJetIPL25Tag;
  delete[] ohBJetIPL3Tag;
  delete[] ohBJetIPL25TagSingleTrack;
  delete[] ohBJetIPL3TagSingleTrack;
  delete[] ohBJetPerfL25Tag;
  delete[] ohBJetPerfL3Tag;
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
    
    tree->Branch("ohBJetIPL25Tag",          ohBJetIPL25Tag,          "ohBJetIPL25Tag[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetIPL3Tag",           ohBJetIPL3Tag,           "ohBJetIPL3Tag[NohBJetL2Corrected]/F");

    tree->Branch("ohBJetIPL25TagSingleTrack", ohBJetIPL25TagSingleTrack,      "ohBJetIPL25TagSingleTrack[NohBJetL2Corrected]/F");
    tree->Branch("ohBJetIPL3TagSingleTrack",  ohBJetIPL3TagSingleTrack,       "ohBJetIPL3TagSingleTrack[NohBJetL2Corrected]/F");

    tree->Branch("ohBJetPerfL25Tag",        ohBJetPerfL25Tag,        "ohBJetPerfL25Tag[NohBJetL2Corrected]/I");
    tree->Branch("ohBJetPerfL3Tag",         ohBJetPerfL3Tag,         "ohBJetPerfL3Tag[NohBJetL2Corrected]/I");
  }
}

void HLTBJet::analyze(
        const edm::Handle<edm::View<reco::Jet> >  & rawBJets,
        const edm::Handle<edm::View<reco::Jet> >  & correctedBJets,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL25SingleTrack,
        const edm::Handle<reco::JetTagCollection> & lifetimeBJetsL3SingleTrack,
        const edm::Handle<reco::JetTagCollection> & performanceBJetsL25,
        const edm::Handle<reco::JetTagCollection> & performanceBJetsL3,
        TTree * tree) 
{
  // reset the tree variables
  clear();
  
  // if the required collections are available, fill the corresponding tree branches
  if (rawBJets.isValid())
    analyseJets(* rawBJets);

  if (correctedBJets.isValid())
    analyseCorrectedJets(* correctedBJets);
 
  if (correctedBJets.isValid() and lifetimeBJetsL25.isValid() and lifetimeBJetsL3.isValid())
    analyseLifetime(* correctedBJets, * lifetimeBJetsL25, * lifetimeBJetsL3);

  if (correctedBJets.isValid() and lifetimeBJetsL25SingleTrack.isValid() and lifetimeBJetsL3SingleTrack.isValid())
    analyseLifetimeSingleTrack(* correctedBJets, * lifetimeBJetsL25SingleTrack, * lifetimeBJetsL3SingleTrack);

  if (correctedBJets.isValid() and performanceBJetsL25.isValid() and performanceBJetsL3.isValid())
    analysePerformance(* correctedBJets, * performanceBJetsL25, * performanceBJetsL3);

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
