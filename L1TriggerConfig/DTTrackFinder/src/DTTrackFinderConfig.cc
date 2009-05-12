//-------------------------------------------------
//
//   Class: DTTrackFinderConfig
//
//   L1 DT Track Finder ESProducer
//
//
//   $Date: 2008/10/13 03:26:13 $
//   $Revision: 1.4 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTTrackFinderConfig.h"

#include <iostream>
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


auto_ptr<L1MuDTExtLut> DTTrackFinderConfig::produceL1MuDTExtLut(const L1MuDTExtLutRcd& iRecord) {

   auto_ptr<L1MuDTExtLut> extlut = auto_ptr<L1MuDTExtLut>( new L1MuDTExtLut() );

   if ( extlut->load() != 0 ) {
     cout << "Can not open files to load  extrapolation look-up tables for DTTrackFinder!" << endl;
   }

   return extlut;
}

auto_ptr<L1MuDTPhiLut> DTTrackFinderConfig::produceL1MuDTPhiLut(const L1MuDTPhiLutRcd& iRecord) {

   auto_ptr<L1MuDTPhiLut> philut = auto_ptr<L1MuDTPhiLut>( new L1MuDTPhiLut() );

   if ( philut->load() != 0 ) {
     cout << "Can not open files to load phi-assignment look-up tables for DTTrackFinder!" << endl;
   }

   return philut;
}

auto_ptr<L1MuDTPtaLut> DTTrackFinderConfig::produceL1MuDTPtaLut(const L1MuDTPtaLutRcd& iRecord) {

   auto_ptr<L1MuDTPtaLut> ptalut = auto_ptr<L1MuDTPtaLut>( new L1MuDTPtaLut() );

   if ( ptalut->load() != 0 ) {
     cout << "Can not open files to load pt-assignment look-up tables for DTTrackFinder!" << endl;
   }

   return ptalut;
}

auto_ptr<L1MuDTEtaPatternLut> DTTrackFinderConfig::produceL1MuDTEtaPatternLut(const L1MuDTEtaPatternLutRcd& iRecord) {

   auto_ptr<L1MuDTEtaPatternLut> etalut = auto_ptr<L1MuDTEtaPatternLut>( new L1MuDTEtaPatternLut() );

   if ( etalut->load() != 0 ) {
     cout << "Can not open files to load eta track finder look-up tables for DTTrackFinder!" << endl;
   }

   return etalut;
}

auto_ptr<L1MuDTQualPatternLut> DTTrackFinderConfig::produceL1MuDTQualPatternLut(const L1MuDTQualPatternLutRcd& iRecord) {

   auto_ptr<L1MuDTQualPatternLut> qualut = auto_ptr<L1MuDTQualPatternLut>( new L1MuDTQualPatternLut() );

   if ( qualut->load() != 0 ) {
     cout << "Can not open files to load eta matching look-up tables for DTTrackFinder!" << endl;
   }

   return qualut;
}

auto_ptr<L1MuDTTFParameters> DTTrackFinderConfig::produceL1MuDTTFParameters(const L1MuDTTFParametersRcd& iRecord) {

   auto_ptr<L1MuDTTFParameters> dttfpar = auto_ptr<L1MuDTTFParameters>( new L1MuDTTFParameters() );

   dttfpar->reset();

   return dttfpar;
}

auto_ptr<L1MuDTTFMasks> DTTrackFinderConfig::produceL1MuDTTFMasks(const L1MuDTTFMasksRcd& iRecord) {

   auto_ptr<L1MuDTTFMasks> dttfmsk = auto_ptr<L1MuDTTFMasks>( new L1MuDTTFMasks() );

   dttfmsk->reset();

   return dttfmsk;
}

DEFINE_FWK_EVENTSETUP_MODULE(DTTrackFinderConfig);
