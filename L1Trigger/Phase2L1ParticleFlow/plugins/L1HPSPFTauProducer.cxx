#include <vector>
#include <numeric>

////////////////////
// FRAMEWORK HEADERS
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
// bitwise emulation headers
#include "L1Trigger/Phase2L1ParticleFlow/interface/taus/L1HPSPFTauEmulator.h"


class L1HPSPFTauProducer : public edm::global::EDProducer<> {
    
public:
  explicit L1HPSPFTauProducer(const edm::ParameterSet&);
  ~L1HPSPFTauProducer() override {};
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  

private:
  
  
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;
  

  //various needed vars
  int _nTaus;
  bool _HW;
  bool fUseJets_;
  bool _debug;
  
  edm::InputTag srcL1PFCands_;
  edm::EDGetTokenT<l1t::PFCandidateCollection> tokenL1PFCands_;    
  //jets
  edm::InputTag srcL1PFJets_;
  edm::EDGetTokenT<std::vector<reco::CaloJet>> tokenL1PFJets_;
  //functions
  std::vector<l1t::PFTau> processEvent_HW(std::vector<edm::Ptr<l1t::PFCandidate>>& parts, std::vector<edm::Ptr<reco::CaloJet>>&  jets) const;

  static std::vector<L1HPSPFTauEmu::Particle> convertJetsToHW(std::vector<edm::Ptr<reco::CaloJet>>& edmJets); 
  static std::vector<L1HPSPFTauEmu::Particle> convertEDMToHW(std::vector<edm::Ptr<l1t::PFCandidate>>& edmParticles);
  static std::vector<l1t::PFTau> convertHWToEDM(std::vector<L1HPSPFTauEmu::Tau> hwTaus);
  
};

L1HPSPFTauProducer::L1HPSPFTauProducer(const edm::ParameterSet& cfg)
   : _nTaus(cfg.getParameter<int>("nTaus")),
     _HW(cfg.getParameter<bool>("HW")),
     fUseJets_(cfg.getParameter<bool>("useJets")),
     _debug(cfg.getParameter<bool>("debug")){ //,
      srcL1PFCands_ = cfg.getParameter<edm::InputTag>("srcL1PFCands");
      tokenL1PFCands_ = consumes<l1t::PFCandidateCollection>(srcL1PFCands_);   
      
      srcL1PFJets_ = cfg.getParameter<edm::InputTag>("srcL1PFJets");
      tokenL1PFJets_ = consumes<std::vector<reco::CaloJet>>(srcL1PFJets_); 
      produces<l1t::PFTauCollection>("HPSTaus");
      
      produces<l1t::PFCandidateCollection>("SelPFCands");

     }

void L1HPSPFTauProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions){
  edm::ParameterSetDescription desc;    
  
  desc.add<edm::InputTag>("srcL1PFCands", edm::InputTag("l1tLayer1","Puppi"));    
  desc.add<int>("nTaus", 16);
  desc.add<bool>("HW", true);
  desc.add<bool>("useJets",false);
  desc.add<bool>("debug", false);   
  desc.add<edm::InputTag>("srcL1PFJets", edm::InputTag("l1tPhase1JetProducer","UncalibratedPhase1L1TJetFromPfCandidates"));
  descriptions.add("L1HPSPFTauProducer", desc);
}


void L1HPSPFTauProducer::produce(edm::StreamID,
                                 edm::Event& iEvent,
                                 const edm::EventSetup& iSetup) const {

  std::unique_ptr<l1t::PFTauCollection> newPFTauCollection(new l1t::PFTauCollection);
  std::unique_ptr<l1t::PFCandidateCollection> selParticles(new l1t::PFCandidateCollection);
  edm::Handle<l1t::PFCandidateCollection> l1PFCandidates;
  iEvent.getByToken(tokenL1PFCands_, l1PFCandidates);
  //add jets even if not used, for simplicity
  edm::Handle<std::vector<reco::CaloJet>> l1PFJets;
  iEvent.getByToken(tokenL1PFJets_, l1PFJets);
  //

  //adding collection
  std::vector<edm::Ptr<l1t::PFCandidate>> particles;
  for(unsigned i = 0; i < (*l1PFCandidates).size(); i++) {

    particles.push_back(edm::Ptr<l1t::PFCandidate>(l1PFCandidates, i));  
    
  }


  //get the jets
  std::vector<edm::Ptr<reco::CaloJet>> jets;
  for(unsigned int i = 0; i < (*l1PFJets).size(); i++){
    jets.push_back(edm::Ptr<reco::CaloJet>(l1PFJets, i));  
  //
  }

  std::vector<l1t::PFTau> taus;
 
  taus = processEvent_HW(particles, jets);
  

  std::sort(taus.begin(), taus.end(), [](l1t::PFTau i, l1t::PFTau j) { return (i.pt() > j.pt()); });
  newPFTauCollection->swap(taus);
  iEvent.put(std::move(newPFTauCollection), "HPSTaus");

}


std::vector<l1t::PFTau> L1HPSPFTauProducer::processEvent_HW(std::vector<edm::Ptr<l1t::PFCandidate>>& work, std::vector<edm::Ptr<reco::CaloJet>>& jwork) const{
    //convert and call emulator
    
 
    using namespace L1HPSPFTauEmu;
    
    std::vector<Particle> particles = convertEDMToHW(work);
    
    std::vector<Particle> jets = convertJetsToHW(jwork);
    //also need to pass the jet enabler
    
    
    bool jEnable = fUseJets_;
    
    std::vector<Tau> taus = emulateEvent(particles, jets, jEnable);
    
    return convertHWToEDM(taus);
    
}

std::vector<L1HPSPFTauEmu::Particle> L1HPSPFTauProducer::convertJetsToHW(std::vector<edm::Ptr<reco::CaloJet>>& edmJets){
  using namespace L1HPSPFTauEmu;
  std::vector<Particle> hwJets;
  std::for_each(edmJets.begin(), edmJets.end(), [&](edm::Ptr<reco::CaloJet>& edmJet){
  	L1HPSPFTauEmu::Particle jPart;
        jPart.hwPt = l1ct::Scales::makePtFromFloat(edmJet->pt());
        jPart.hwEta = edmJet->eta() * etaphi_base;
        jPart.hwPhi = edmJet->phi() * etaphi_base;
        jPart.tempZ0 = 0.;
        hwJets.push_back(jPart);
  });
  return hwJets;
}


//conversion to and from HW bitwise 
std::vector<L1HPSPFTauEmu::Particle> L1HPSPFTauProducer::convertEDMToHW(std::vector<edm::Ptr<l1t::PFCandidate>>& edmParticles){
    using namespace L1HPSPFTauEmu;
    std::vector<Particle> hwParticles;
    
    std::for_each(edmParticles.begin(), edmParticles.end(), [&](edm::Ptr<l1t::PFCandidate>& edmParticle){
      Particle hwPart;
      hwPart.hwPt = l1ct::Scales::makePtFromFloat(edmParticle->pt());
      hwPart.hwEta = edmParticle->eta() * etaphi_base;
      hwPart.hwPhi = edmParticle->phi() * etaphi_base;
      hwPart.pID = edmParticle->id();
      if(edmParticle->z0()) {
        hwPart.tempZ0 = edmParticle->z0() / dz_base;
      }
      hwParticles.push_back(hwPart);

 });
  return hwParticles;
    
}

std::vector<l1t::PFTau> L1HPSPFTauProducer::convertHWToEDM(std::vector<L1HPSPFTauEmu::Tau> hwTaus){
    using namespace L1HPSPFTauEmu;
    std::vector<l1t::PFTau> edmTaus;

    //empty array for the PFTau format, since it's used for PuppiTaus but not here
    float tauArray[80] = {0};
    std::for_each(hwTaus.begin(), hwTaus.end(), [&](Tau tau){
    l1gt::Tau gtTau = tau.toGT();
    l1gt::PackedTau packTau = gtTau.pack();

           
    l1t::PFTau pTau(reco::Candidate::PolarLorentzVector(l1ct::Scales::floatPt(tau.hwPt),
                                 float(tau.hwEta) / etaphi_base,
                                 float(tau.hwPhi) / etaphi_base,
                                 0), tauArray,  0, 0, 0, tau.hwPt, tau.hwEta, tau.hwPhi);
    pTau.set_encodedTau(packTau);
    edmTaus.push_back(pTau);
  });
  return edmTaus;
}  


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1HPSPFTauProducer);
