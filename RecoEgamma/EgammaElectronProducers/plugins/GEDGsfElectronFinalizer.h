
#ifndef GEDGsfElectronFinalizer_h
#define GEDGsfElectronFinalizer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <string>
#include <vector>

class GEDGsfElectronFinalizer : public edm::stream::EDProducer<>
{
 public:
  explicit GEDGsfElectronFinalizer (const edm::ParameterSet &);
  ~GEDGsfElectronFinalizer(); 
  
  virtual void produce(edm::Event &, const edm::EventSetup&);

 private:
  edm::EDGetTokenT<reco::GsfElectronCollection> previousGsfElectrons_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
  std::string outputCollectionLabel_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > tokenElectronIsoVals_;
  unsigned nDeps_;
  
};

#endif
