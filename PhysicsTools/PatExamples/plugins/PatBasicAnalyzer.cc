#include <map>
#include <string>

#include "TH1.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

class PatBasicAnalyzer : public edm::EDAnalyzer {

public:
  explicit PatBasicAnalyzer(const edm::ParameterSet&);
  ~PatBasicAnalyzer();
  
private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // simple map to contain all histograms; 
  // histograms are booked in the beginJob() 
  // method
  std::map<std::string,TH1F*> histContainer_; 

  // input tags  
  edm::InputTag photonSrc_;
  edm::InputTag elecSrc_;
  edm::InputTag muonSrc_;
  edm::InputTag tauSrc_;
  edm::InputTag jetSrc_;
  edm::InputTag metSrc_;
};

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

PatBasicAnalyzer::PatBasicAnalyzer(const edm::ParameterSet& iConfig):
  histContainer_(),
  photonSrc_(iConfig.getUntrackedParameter<edm::InputTag>("photonSrc")),
  elecSrc_(iConfig.getUntrackedParameter<edm::InputTag>("electronSrc")),
  muonSrc_(iConfig.getUntrackedParameter<edm::InputTag>("muonSrc")),
  tauSrc_(iConfig.getUntrackedParameter<edm::InputTag>("tauSrc" )),
  jetSrc_(iConfig.getUntrackedParameter<edm::InputTag>("jetSrc" )),
  metSrc_(iConfig.getUntrackedParameter<edm::InputTag>("metSrc" ))
{
}

PatBasicAnalyzer::~PatBasicAnalyzer()
{
}

void
PatBasicAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // get electron collection
  edm::Handle<edm::View<pat::Electron> > electrons;
  iEvent.getByLabel(elecSrc_,electrons);

  // get muon collection
  edm::Handle<edm::View<pat::Muon> > muons;
  iEvent.getByLabel(muonSrc_,muons);

  // get tau collection  
  edm::Handle<edm::View<pat::Tau> > taus;
  iEvent.getByLabel(tauSrc_,taus);

  // get jet collection
  edm::Handle<edm::View<pat::Jet> > jets;
  iEvent.getByLabel(jetSrc_,jets);

  // get met collection  
  edm::Handle<edm::View<pat::MET> > mets;
  iEvent.getByLabel(metSrc_,mets);
  
  // get photon collection  
  edm::Handle<edm::View<pat::Photon> > photons;
  iEvent.getByLabel(photonSrc_,photons);
    
  // loop over jets
  size_t nJets=0;
  for(edm::View<pat::Jet>::const_iterator jet=jets->begin(); jet!=jets->end(); ++jet){
    if(jet->pt()>50)
      ++nJets;
  }
  histContainer_["jets"]->Fill(nJets);

  // do something similar for the other candidates
  histContainer_["photons"]->Fill(photons->size() );
  histContainer_["elecs" ]->Fill(electrons->size());
  histContainer_["muons"]->Fill(muons->size() );
  histContainer_["taus" ]->Fill(taus->size()  );
  histContainer_["met"  ]->Fill(mets->empty() ? 0 : (*mets)[0].et());
}

void 
PatBasicAnalyzer::beginJob(const edm::EventSetup&)
{
  // register to the TFileService
  edm::Service<TFileService> fs;
  
  // book histograms:
  histContainer_["photons"]=fs->make<TH1F>("photons", "photon multiplicity",   10, 0,  10);
  histContainer_["elecs"  ]=fs->make<TH1F>("elecs",   "electron multiplicity", 10, 0,  10);
  histContainer_["muons"  ]=fs->make<TH1F>("muons",   "muon multiplicity",     10, 0,  10);
  histContainer_["taus"   ]=fs->make<TH1F>("taus",    "tau multiplicity",      10, 0,  10);
  histContainer_["jets"   ]=fs->make<TH1F>("jets",    "jet multiplicity",      10, 0,  10);
  histContainer_["met"    ]=fs->make<TH1F>("met",     "missing E_{T}",         20, 0, 100);
}

void 
PatBasicAnalyzer::endJob() 
{
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PatBasicAnalyzer);
