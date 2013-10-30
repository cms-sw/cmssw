
#ifndef PFIsolationFiller_h
#define PFIsolationFiller_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include <string>
#include <vector>

class PFIsolationFiller : public edm::EDProducer
{
 public:
  explicit PFIsolationFiller (const edm::ParameterSet &);
  ~PFIsolationFiller(); 
  
  virtual void produce(edm::Event &, const edm::EventSetup&);

 private:
  edm::EDGetTokenT<reco::GsfElectronCollection> previousGsfElectrons_;
  std::string outputCollectionLabel_;
  std::vector<edm::EDGetTokenT<edm::ValueMap<double> > > tokenElectronIsoVals_;
  unsigned nDeps_;
  
};

#endif
