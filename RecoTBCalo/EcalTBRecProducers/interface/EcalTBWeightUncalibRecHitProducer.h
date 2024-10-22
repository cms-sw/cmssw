#ifndef RecoTBCalo_EcalTBRecProducers_EcalTBWeightUncalibRecHitProducer_HH
#define RecoTBCalo_EcalTBRecProducers_EcalTBWeightUncalibRecHitProducer_HH

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBTDCRecInfo.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"

// forward declaration
class EcalTBWeightUncalibRecHitProducer : public edm::stream::EDProducer<> {
public:
  typedef std::vector<double> EcalRecoAmplitudes;
  explicit EcalTBWeightUncalibRecHitProducer(const edm::ParameterSet& ps);
  ~EcalTBWeightUncalibRecHitProducer() override;
  void produce(edm::Event& evt, const edm::EventSetup& es) override;

private:
  const edm::InputTag ebDigiCollection_;
  const edm::InputTag eeDigiCollection_;
  const edm::InputTag tdcRecInfoCollection_;

  const std::string ebHitCollection_;  // secondary name to be given to collection of hit
  const std::string eeHitCollection_;  // secondary name to be given to collection of hit

  const edm::EDGetTokenT<EBDigiCollection> ebDigiToken_;
  const edm::EDGetTokenT<EEDigiCollection> eeDigiToken_;
  const edm::EDGetTokenT<EcalTBTDCRecInfo> tbTDCRecInfoToken_;
  const edm::ESGetToken<EcalWeightXtalGroups, EcalWeightXtalGroupsRcd> weightXtalGroupsToken_;
  const edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> gainRatiosToken_;
  const edm::ESGetToken<EcalTBWeights, EcalTBWeightsRcd> tbWeightsToken_;
  const edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalsToken_;

  EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> ebAlgo_;
  EcalUncalibRecHitRecWeightsAlgo<EEDataFrame> eeAlgo_;

  const EEShape testbeamEEShape;
  const EBShape testbeamEBShape;

  /*     HepMatrix makeMatrixFromVectors(const std::vector< std::vector<EcalWeight> >& vecvec); */
  /*     HepMatrix makeDummySymMatrix(int size); */

  const int nbTimeBin_;

  //use 2004 convention for the TDC
  const bool use2004OffsetConvention_;
};
#endif
