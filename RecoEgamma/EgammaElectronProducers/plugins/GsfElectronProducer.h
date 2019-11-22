
#ifndef GsfElectronProducer_h
#define GsfElectronProducer_h

#include "GsfElectronBaseProducer.h"

class GsfElectronProducer : public GsfElectronBaseProducer {
public:
  explicit GsfElectronProducer(const edm::ParameterSet&, const GsfElectronAlgo::HeavyObjectCache*);
  void produce(edm::Event&, const edm::EventSetup&) override;

protected:
  void beginEvent(edm::Event&, const edm::EventSetup&);

private:
  reco::GsfElectronCollection clonePreviousElectrons(edm::Event const& event) const;
  void addPflowInfo(reco::GsfElectronCollection& electrons, edm::Event const& event) const;  // now deprecated
  void setPflowPreselectionFlag(reco::GsfElectron& ele) const;

  // check expected configuration of previous modules
  bool pfTranslatorParametersChecked_;
  void checkPfTranslatorParameters(edm::ParameterSet const&);
};

#endif
