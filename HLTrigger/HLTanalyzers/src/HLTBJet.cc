#include <cmath>

#include <TTree.h>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "HLTrigger/HLTanalyzers/interface/HLTBJet.h"

const int kMaxBJets = 4;

HLTBJet::HLTBJet() 
{
  m_lifetimeBJets = 0;
  m_lifetimeBJetL2Energy         = new float[kMaxBJets];
  m_lifetimeBJetL2ET             = new float[kMaxBJets];
  m_lifetimeBJetL2Eta            = new float[kMaxBJets];
  m_lifetimeBJetL2Phi            = new float[kMaxBJets];
  m_lifetimeBJetL25Discriminator = new float[kMaxBJets];
  m_lifetimeBJetL3Discriminator  = new float[kMaxBJets];

  m_softmuonBJets = 0;
  m_softmuonBJetL2Energy         = new float[kMaxBJets];
  m_softmuonBJetL2ET             = new float[kMaxBJets];
  m_softmuonBJetL2Eta            = new float[kMaxBJets];
  m_softmuonBJetL2Phi            = new float[kMaxBJets];
  m_softmuonBJetL25Discriminator = new int[kMaxBJets];
  m_softmuonBJetL3Discriminator  = new float[kMaxBJets];

  m_performanceBJets = 0;
  m_performanceBJetL2Energy         = new float[kMaxBJets];
  m_performanceBJetL2ET             = new float[kMaxBJets];
  m_performanceBJetL2Eta            = new float[kMaxBJets];
  m_performanceBJetL2Phi            = new float[kMaxBJets];
  m_performanceBJetL25Discriminator = new int[kMaxBJets];
  m_performanceBJetL3Discriminator  = new int[kMaxBJets];
}

HLTBJet::~HLTBJet() 
{ }

void HLTBJet::setup(const edm::ParameterSet & config, TTree * tree)
{
  // read the product names
  m_lifetimeBjetL2     = config.getParameter<edm::InputTag>("LifetimeBJetsL2");
  m_lifetimeBjetL25    = config.getParameter<edm::InputTag>("LifetimeBJetsL25");
  m_lifetimeBjetL3     = config.getParameter<edm::InputTag>("LifetimeBJetsL3");
  m_softmuonBjetL2     = config.getParameter<edm::InputTag>("SoftmuonBJetsL2");
  m_softmuonBjetL25    = config.getParameter<edm::InputTag>("SoftmuonBJetsL25");
  m_softmuonBjetL3     = config.getParameter<edm::InputTag>("SoftmuonBJetsL3");
  m_performanceBjetL2  = config.getParameter<edm::InputTag>("PerformanceBJetsL2");
  m_performanceBjetL25 = config.getParameter<edm::InputTag>("PerformanceBJetsL25");
  m_performanceBjetL3  = config.getParameter<edm::InputTag>("PerformanceBJetsL3");

  // create the TTree branches
  if (tree) {
    tree->Branch("NohBJetLife",                 & m_lifetimeBJets,                  "NohBJetLife/I");
    tree->Branch("ohBJetLifeL2E",               m_lifetimeBJetL2Energy,             "ohBJetLifeL2E[NohBJetLife]/F");
    tree->Branch("ohBJetLifeL2ET",              m_lifetimeBJetL2ET,                 "ohBJetLifeL2ET[NohBJetLife]/F");
    tree->Branch("ohBJetLifeL2Eta",             m_lifetimeBJetL2Eta,                "ohBJetLifeL2Eta[NohBJetLife]/F");
    tree->Branch("ohBJetLifeL2Phi",             m_lifetimeBJetL2Phi,                "ohBJetLifeL2Phi[NohBJetLife]/F");
    tree->Branch("ohBJetLifeL25Discriminator",  m_lifetimeBJetL25Discriminator,     "ohBJetLifeL25Discriminator[NohBJetLife]/F");
    tree->Branch("ohBJetLifeL3Discriminator",   m_lifetimeBJetL3Discriminator,      "ohBJetLifeL3Discriminator[NohBJetLife]/F");

    tree->Branch("NohBJetSoftm",                & m_softmuonBJets,                  "NohBJetSoftm/I");
    tree->Branch("ohBJetSoftmL2E",              m_softmuonBJetL2Energy,             "ohBJetSoftmL2E[NohBJetSoftm]/F");
    tree->Branch("ohBJetSoftmL2ET",             m_softmuonBJetL2ET,                 "ohBJetSoftmL2ET[NohBJetSoftm]/F");
    tree->Branch("ohBJetSoftmL2Eta",            m_softmuonBJetL2Eta,                "ohBJetSoftmL2Eta[NohBJetSoftm]/F");
    tree->Branch("ohBJetSoftmL2Phi",            m_softmuonBJetL2Phi,                "ohBJetSoftmL2Phi[NohBJetSoftm]/F");
    tree->Branch("ohBJetSoftmL25Discriminator", m_softmuonBJetL25Discriminator,     "ohBJetSoftmL25Discriminator[NohBJetSoftm]/I");
    tree->Branch("ohBJetSoftmL3Discriminator",  m_softmuonBJetL3Discriminator,      "ohBJetSoftmL3Discriminator[NohBJetSoftm]/F");

    tree->Branch("NohBJetPerf",                 & m_performanceBJets,               "NohBJetPerf/I");
    tree->Branch("ohBJetPerfL2E",               m_performanceBJetL2Energy,          "ohBJetPerfL2E[NohBJetPerf]/F");
    tree->Branch("ohBJetPerfL2ET",              m_performanceBJetL2ET,              "ohBJetPerfL2ET[NohBJetPerf]/F");
    tree->Branch("ohBJetPerfL2Eta",             m_performanceBJetL2Eta,             "ohBJetPerfL2Eta[NohBJetPerf]/F");
    tree->Branch("ohBJetPerfL2Phi",             m_performanceBJetL2Phi,             "ohBJetPerfL2Phi[NohBJetPerf]/F");
    tree->Branch("ohBJetPerfL25Discriminator",  m_performanceBJetL25Discriminator,  "ohBJetPerfL25Discriminator[NohBJetPerf]/I");
    tree->Branch("ohBJetPerfL3Discriminator",   m_performanceBJetL3Discriminator,   "ohBJetPerfL3Discriminator[NohBJetPerf]/I");
  }
}

void HLTBJet::analyze(const edm::Event & event, const edm::EventSetup & setup, TTree * tree)
{
  // read the collections from the Event
  edm::Handle<edm::View<reco::Jet> >  h_lifetimeBjetL2;
  edm::Handle<reco::JetTagCollection> h_lifetimeBjetL25;
  edm::Handle<reco::JetTagCollection> h_lifetimeBjetL3;
  edm::Handle<edm::View<reco::Jet> >  h_softmuonBjetL2;
  edm::Handle<reco::JetTagCollection> h_softmuonBjetL25;
  edm::Handle<reco::JetTagCollection> h_softmuonBjetL3;
  edm::Handle<edm::View<reco::Jet> >  h_performanceBjetL2;
  edm::Handle<reco::JetTagCollection> h_performanceBjetL25;
  edm::Handle<reco::JetTagCollection> h_performanceBjetL3;

  event.getByLabel(m_lifetimeBjetL2,     h_lifetimeBjetL2);
  event.getByLabel(m_lifetimeBjetL25,    h_lifetimeBjetL25);
  event.getByLabel(m_lifetimeBjetL3,     h_lifetimeBjetL3);
  event.getByLabel(m_softmuonBjetL2,     h_softmuonBjetL2);
  event.getByLabel(m_softmuonBjetL25,    h_softmuonBjetL25);
  event.getByLabel(m_softmuonBjetL3,     h_softmuonBjetL3);
  event.getByLabel(m_performanceBjetL2,  h_performanceBjetL2);
  event.getByLabel(m_performanceBjetL25, h_performanceBjetL25);
  event.getByLabel(m_performanceBjetL3,  h_performanceBjetL3);

  // FIXME do something - MessageLogger ?
  std::string log;
  if (! h_lifetimeBjetL2.isValid())     { log += "  -- No L2 lifetime b-jet candidates"; }
  if (! h_lifetimeBjetL25.isValid())    { log += "  -- No L2.5 lifetime b-jet candidates"; }
  if (! h_lifetimeBjetL3.isValid())     { log += "  -- No L3 lifetime b-jet candidates"; }
  if (! h_softmuonBjetL2.isValid())     { log += "  -- No L2 soft mu b-jet candidates"; }
  if (! h_softmuonBjetL25.isValid())    { log += "  -- No L2.5 soft mu b-jet candidates"; }
  if (! h_softmuonBjetL3.isValid())     { log += "  -- No L3 soft mu b-jet candidates"; }
  if (! h_performanceBjetL2.isValid())  { log += "  -- No L2 b-jet perf. measurement candidates"; }
  if (! h_performanceBjetL25.isValid()) { log += "  -- No L2.5 b-jet perf. measurement candidates"; }
  if (! h_performanceBjetL3.isValid())  { log += "  -- No L3 b-jet perf. measurement candidates"; }
  if (log.size()) {
    std::cerr << "OpenHLT: " << log << std::endl;
  }

  // if the required collections are available, fill the corresponding tree branches
  if (h_lifetimeBjetL2.isValid() and h_lifetimeBjetL25.isValid() and h_lifetimeBjetL3.isValid()) 
    analyzeLifetime(* h_lifetimeBjetL2, * h_lifetimeBjetL25, * h_lifetimeBjetL3);
  
  if (h_softmuonBjetL2.isValid() and h_softmuonBjetL25.isValid() and h_softmuonBjetL3.isValid()) 
    analyzeSoftmuon(* h_softmuonBjetL2, * h_softmuonBjetL25, * h_softmuonBjetL3);
  
  if (h_performanceBjetL2.isValid() and h_performanceBjetL25.isValid() and h_performanceBjetL3.isValid()) 
    analyzePerformance(* h_performanceBjetL2, * h_performanceBjetL25, * h_performanceBjetL3);
}

void HLTBJet::analyzeLifetime(
    const edm::View<reco::Jet>   & lifetimeBjetL2, 
    const reco::JetTagCollection & lifetimeBjetL25, 
    const reco::JetTagCollection & lifetimeBjetL3)
{
  m_lifetimeBJets = lifetimeBjetL2.size();
  if (lifetimeBjetL25.size() != m_lifetimeBJets) {
    std::cerr << "OpenHLT: L2.5 BJet collection has " << lifetimeBjetL25.size() << " elements, but " << m_lifetimeBJets << " where expected from L2" << std::endl;
    return;
  }
  if (lifetimeBjetL3.size() != m_lifetimeBJets) {
    std::cerr << "OpenHLT: L3 BJet collection has " << lifetimeBjetL3.size() << " elements, but " << m_lifetimeBJets << " where expected from L2" << std::endl;
    return;
  }
  if (m_lifetimeBJets > kMaxBJets) m_lifetimeBJets = kMaxBJets;
  for (int i = 0; i < m_lifetimeBJets; i++) {
    m_lifetimeBJetL2Energy[i]         = lifetimeBjetL2[i].energy();
    m_lifetimeBJetL2ET[i]             = lifetimeBjetL2[i].et();
    m_lifetimeBJetL2Eta[i]            = lifetimeBjetL2[i].eta();
    m_lifetimeBJetL2Phi[i]            = lifetimeBjetL2[i].phi();
    m_lifetimeBJetL25Discriminator[i] = lifetimeBjetL25[i].second;
    m_lifetimeBJetL3Discriminator[i]  = lifetimeBjetL3[i].second;
  }
}

void HLTBJet::analyzeSoftmuon(
    const edm::View<reco::Jet>   & softmuonBjetL2, 
    const reco::JetTagCollection & softmuonBjetL25, 
    const reco::JetTagCollection & softmuonBjetL3)
{
  m_softmuonBJets = softmuonBjetL2.size();
  if (m_softmuonBJets > kMaxBJets) m_softmuonBJets = kMaxBJets;
  for (int i = 0; i < m_softmuonBJets; i++) {
    m_softmuonBJetL2Energy[i]         = softmuonBjetL2[i].energy();
    m_softmuonBJetL2ET[i]             = softmuonBjetL2[i].et();
    m_softmuonBJetL2Eta[i]            = softmuonBjetL2[i].eta();
    m_softmuonBJetL2Phi[i]            = softmuonBjetL2[i].phi();
    m_softmuonBJetL25Discriminator[i] = (unsigned int) (softmuonBjetL25[i].second > 0.0);
    m_softmuonBJetL3Discriminator[i]  = softmuonBjetL3[i].second;
  }
}

void HLTBJet::analyzePerformance(
    const edm::View<reco::Jet>   & performanceBjetL2, 
    const reco::JetTagCollection & performanceBjetL25, 
    const reco::JetTagCollection & performanceBjetL3)
{
  m_performanceBJets = performanceBjetL2.size();
  if (m_performanceBJets > kMaxBJets) m_performanceBJets = kMaxBJets;
  for (int i = 0; i < m_performanceBJets; i++) {
    m_performanceBJetL2Energy[i]         = performanceBjetL2[i].energy();
    m_performanceBJetL2ET[i]             = performanceBjetL2[i].et();
    m_performanceBJetL2Eta[i]            = performanceBjetL2[i].eta();
    m_performanceBJetL2Phi[i]            = performanceBjetL2[i].phi();
    m_performanceBJetL25Discriminator[i] = (unsigned int) (performanceBjetL25[i].second > 0.0);
    m_performanceBJetL3Discriminator[i]  = (unsigned int) (performanceBjetL3[i].second  > 0.0);
  }
}
