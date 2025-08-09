#ifndef SHALLOW_DIGIS_PRODUCER
#define SHALLOW_DIGIS_PRODUCER

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"

class ShallowDigisProducer : public edm::stream::EDProducer<> {
public:
  explicit ShallowDigisProducer(const edm::ParameterSet &);

private:
  struct products {
    std::unique_ptr<std::vector<unsigned>> id;
    std::unique_ptr<std::vector<unsigned>> subdet;
    std::unique_ptr<std::vector<unsigned>> strip;
    std::unique_ptr<std::vector<unsigned>> adc;
    std::unique_ptr<std::vector<float>> noise;
    products()
        : id(new std::vector<unsigned>()),
          subdet(new std::vector<unsigned>()),
          strip(new std::vector<unsigned>()),
          adc(new std::vector<unsigned>()),
          noise(new std::vector<float>()) {}
  };
  std::vector<edm::EDGetTokenT<edm::DetSetVector<SiStripDigi>>> oldTokens_;
  std::vector<edm::EDGetTokenT<edmNew::DetSetVector<SiStripDigi>>> newTokens_;
  edm::ESGetToken<SiStripNoises, SiStripNoisesRcd> noisesToken_;

  void produce(edm::Event &, const edm::EventSetup &) override;
  template <class T>
  bool findInput(edm::Handle<T> &, std::vector<edm::EDGetTokenT<T>> const &, const edm::Event &);
  template <class T>
  void recordDigis(const T &, products &, const SiStripNoises &noises);
  void insert(products &, edm::Event &);
};

#endif
