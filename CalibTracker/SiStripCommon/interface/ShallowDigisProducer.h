#ifndef SHALLOW_DIGIS_PRODUCER
#define SHALLOW_DIGIS_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
class SiStripNoises;

class ShallowDigisProducer : public edm::EDProducer {

 public:

  explicit ShallowDigisProducer(const edm::ParameterSet&);

 private:  
  struct products {
    std::unique_ptr<std::vector<unsigned> > id;
    std::unique_ptr<std::vector<unsigned> > subdet;
    std::unique_ptr<std::vector<unsigned> > strip;
    std::unique_ptr<std::vector<unsigned> > adc;
    std::unique_ptr<std::vector<float> > noise;
    products() 
      : id(new std::vector<unsigned>()), 
	subdet(new std::vector<unsigned>()), 
	strip(new std::vector<unsigned>()), 
	adc(new std::vector<unsigned>()), 
	noise(new std::vector<float>()) {}
  };
  std::vector<edm::InputTag> inputTags;
  edm::ESHandle<SiStripNoises> noiseHandle;

  void produce(edm::Event&, const edm::EventSetup&) override;
  template<class T> bool findInput(edm::Handle<T>&, const edm::Event&);
  template<class T> void recordDigis(const T &, products&);
  void insert(products&, edm::Event&);  
  
};

#endif
