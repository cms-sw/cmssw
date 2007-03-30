//-------------------------------------------------
//
//   Class: DTTrackFinderConfig
//
//   L1 DT Track Finder ESProducer
//
//
//   $Date: 2007/02/27 11:44:00 $
//   $Revision: 1.5 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTTrackFinderConfig.h"

using namespace std;

DTTrackFinderConfig::DTTrackFinderConfig(const edm::ParameterSet& pset) {

  string lutdir_ = pset.getUntrackedParameter<string>("lutdir","L1TriggerConfig/DTTrackFinder/parameters/");
  setenv("DTTF_DATA_PATH",lutdir_.c_str(),1); 

  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTExtLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTPhiLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTPtaLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTEtaPatternLut);
  setWhatProduced(this, &DTTrackFinderConfig::produceL1MuDTQualPatternLut);
  
}


DTTrackFinderConfig::~DTTrackFinderConfig() {}


auto_ptr<L1MuDTExtLut> DTTrackFinderConfig::produceL1MuDTExtLut(const L1MuDTExtLutRcd& iRecord) {

   auto_ptr<L1MuDTExtLut> extlut = auto_ptr<L1MuDTExtLut>( new L1MuDTExtLut() );

   return extlut;
}

auto_ptr<L1MuDTPhiLut> DTTrackFinderConfig::produceL1MuDTPhiLut(const L1MuDTPhiLutRcd& iRecord) {

   auto_ptr<L1MuDTPhiLut> philut = auto_ptr<L1MuDTPhiLut>( new L1MuDTPhiLut() );

   return philut;
}

auto_ptr<L1MuDTPtaLut> DTTrackFinderConfig::produceL1MuDTPtaLut(const L1MuDTPtaLutRcd& iRecord) {

   auto_ptr<L1MuDTPtaLut> ptalut = auto_ptr<L1MuDTPtaLut>( new L1MuDTPtaLut() );

   return ptalut;
}

auto_ptr<L1MuDTEtaPatternLut> DTTrackFinderConfig::produceL1MuDTEtaPatternLut(const L1MuDTEtaPatternLutRcd& iRecord) {

   auto_ptr<L1MuDTEtaPatternLut> etalut = auto_ptr<L1MuDTEtaPatternLut>( new L1MuDTEtaPatternLut() );

   return etalut;
}

auto_ptr<L1MuDTQualPatternLut> DTTrackFinderConfig::produceL1MuDTQualPatternLut(const L1MuDTQualPatternLutRcd& iRecord) {

   auto_ptr<L1MuDTQualPatternLut> qualut = auto_ptr<L1MuDTQualPatternLut>( new L1MuDTQualPatternLut() );

   return qualut;
}
