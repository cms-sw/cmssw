#ifndef SHALLOW_TRACKS_PRODUCER
#define SHALLOW_TRACKS_PRODUCER

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"

class ShallowTracksProducer : public edm::global::EDProducer<> {
public:
  explicit ShallowTracksProducer(const edm::ParameterSet &);

private:
  const edm::EDGetTokenT<edm::View<reco::Track>> tracks_token_;
  edm::InputTag theTracksLabel;
  std::string Prefix;
  std::string Suffix;
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  const edm::EDPutTokenT<unsigned int> numberPut_;
  const edm::EDPutTokenT<std::vector<double>> chi2Put_;
  const edm::EDPutTokenT<std::vector<double>> ndofPut_;
  const edm::EDPutTokenT<std::vector<double>> chi2ndofPut_;
  const edm::EDPutTokenT<std::vector<float>> chargePut_;
  const edm::EDPutTokenT<std::vector<float>> momentumPut_;
  const edm::EDPutTokenT<std::vector<float>> ptPut_;
  const edm::EDPutTokenT<std::vector<float>> pterrPut_;
  const edm::EDPutTokenT<std::vector<unsigned int>> hitsvalidPut_;
  const edm::EDPutTokenT<std::vector<unsigned int>> hitslostPut_;
  const edm::EDPutTokenT<std::vector<double>> thetaPut_;
  const edm::EDPutTokenT<std::vector<double>> thetaerrPut_;
  const edm::EDPutTokenT<std::vector<double>> phiPut_;
  const edm::EDPutTokenT<std::vector<double>> phierrPut_;
  const edm::EDPutTokenT<std::vector<double>> etaPut_;
  const edm::EDPutTokenT<std::vector<double>> etaerrPut_;
  const edm::EDPutTokenT<std::vector<double>> dxyPut_;
  const edm::EDPutTokenT<std::vector<double>> dxyerrPut_;
  const edm::EDPutTokenT<std::vector<double>> dszPut_;
  const edm::EDPutTokenT<std::vector<double>> dszerrPut_;
  const edm::EDPutTokenT<std::vector<double>> qoverpPut_;
  const edm::EDPutTokenT<std::vector<double>> qoverperrPut_;
  const edm::EDPutTokenT<std::vector<double>> vxPut_;
  const edm::EDPutTokenT<std::vector<double>> vyPut_;
  const edm::EDPutTokenT<std::vector<double>> vzPut_;
  const edm::EDPutTokenT<std::vector<int>> algoPut_;
};
#endif
