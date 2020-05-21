//-------------------------------------------------
//
//   Class: DTTrackFinderConfig
//
//   L1 DT Track Finder ESProducer
//
//
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTTrackFinderConfig.h"

#include <iostream>
#include <memory>

#include <string>

using namespace std;

DTTrackFinderConfig::DTTrackFinderConfig(const edm::ParameterSet& pset) {
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTExtLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTPhiLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTPtaLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTEtaPatternLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTQualPatternLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTTFParameters);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTTFMasks);
}

DTTrackFinderConfig::~DTTrackFinderConfig() {}

unique_ptr<L1MuDTExtLut> DTTrackFinderConfig::produceL1MuDTExtLut(const L1MuDTExtLutRcd& iRecord) {
  unique_ptr<L1MuDTExtLut> extlut = std::make_unique<L1MuDTExtLut>();

  if (extlut->load() != 0) {
    cout << "Can not open files to load  extrapolation look-up tables for DTTrackFinder!" << endl;
  }

  return extlut;
}

unique_ptr<L1MuDTPhiLut> DTTrackFinderConfig::produceL1MuDTPhiLut(const L1MuDTPhiLutRcd& iRecord) {
  unique_ptr<L1MuDTPhiLut> philut = std::make_unique<L1MuDTPhiLut>();

  if (philut->load() != 0) {
    cout << "Can not open files to load phi-assignment look-up tables for DTTrackFinder!" << endl;
  }

  return philut;
}

unique_ptr<L1MuDTPtaLut> DTTrackFinderConfig::produceL1MuDTPtaLut(const L1MuDTPtaLutRcd& iRecord) {
  unique_ptr<L1MuDTPtaLut> ptalut = std::make_unique<L1MuDTPtaLut>();

  if (ptalut->load() != 0) {
    cout << "Can not open files to load pt-assignment look-up tables for DTTrackFinder!" << endl;
  }

  return ptalut;
}

unique_ptr<L1MuDTEtaPatternLut> DTTrackFinderConfig::produceL1MuDTEtaPatternLut(const L1MuDTEtaPatternLutRcd& iRecord) {
  unique_ptr<L1MuDTEtaPatternLut> etalut = std::make_unique<L1MuDTEtaPatternLut>();

  if (etalut->load() != 0) {
    cout << "Can not open files to load eta track finder look-up tables for DTTrackFinder!" << endl;
  }

  return etalut;
}

unique_ptr<L1MuDTQualPatternLut> DTTrackFinderConfig::produceL1MuDTQualPatternLut(
    const L1MuDTQualPatternLutRcd& iRecord) {
  unique_ptr<L1MuDTQualPatternLut> qualut = std::make_unique<L1MuDTQualPatternLut>();

  if (qualut->load() != 0) {
    cout << "Can not open files to load eta matching look-up tables for DTTrackFinder!" << endl;
  }

  return qualut;
}

unique_ptr<L1MuDTTFParameters> DTTrackFinderConfig::produceL1MuDTTFParameters(const L1MuDTTFParametersRcd& iRecord) {
  unique_ptr<L1MuDTTFParameters> dttfpar = std::make_unique<L1MuDTTFParameters>();

  dttfpar->reset();

  return dttfpar;
}

unique_ptr<L1MuDTTFMasks> DTTrackFinderConfig::produceL1MuDTTFMasks(const L1MuDTTFMasksRcd& iRecord) {
  unique_ptr<L1MuDTTFMasks> dttfmsk = std::make_unique<L1MuDTTFMasks>();

  dttfmsk->reset();

  return dttfmsk;
}

DEFINE_FWK_EVENTSETUP_MODULE(DTTrackFinderConfig);
