//
//

#include "PhysicsTools/PatAlgos/plugins/PATLeptonCountFilter.h"

#include "DataFormats/Common/interface/Handle.h"


using namespace pat;


PATLeptonCountFilter::PATLeptonCountFilter(const edm::ParameterSet & iConfig) :
  electronToken_(mayConsume<edm::View<Electron> >(iConfig.getParameter<edm::InputTag>( "electronSource" ))),
  muonToken_(mayConsume<edm::View<Muon> >(iConfig.getParameter<edm::InputTag>( "muonSource" ))),
  tauToken_(mayConsume<edm::View<Tau> >(iConfig.getParameter<edm::InputTag>( "tauSource" ))),
  countElectrons_(iConfig.getParameter<bool>         ( "countElectrons" )),
  countMuons_(iConfig.getParameter<bool>         ( "countMuons" )),
  countTaus_(iConfig.getParameter<bool>         ( "countTaus" )),
  minNumber_(iConfig.getParameter<unsigned int> ( "minNumber" )),
  maxNumber_(iConfig.getParameter<unsigned int> ( "maxNumber" )) {
  
}


PATLeptonCountFilter::~PATLeptonCountFilter() {
}


bool PATLeptonCountFilter::filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {
  edm::Handle<edm::View<Electron> > electrons;
  if (countElectrons_) iEvent.getByToken(electronToken_, electrons);
  edm::Handle<edm::View<Muon> > muons;
  if (countMuons_) iEvent.getByToken(muonToken_, muons);
  edm::Handle<edm::View<Tau> > taus;
  if (countTaus_) iEvent.getByToken(tauToken_, taus);
  unsigned int nrLeptons = 0;
  nrLeptons += (countElectrons_ ? electrons->size() : 0);
  nrLeptons += (countMuons_     ? muons->size()     : 0);
  nrLeptons += (countTaus_      ? taus->size()      : 0);
  return nrLeptons >= minNumber_ && nrLeptons <= maxNumber_;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PATLeptonCountFilter);

