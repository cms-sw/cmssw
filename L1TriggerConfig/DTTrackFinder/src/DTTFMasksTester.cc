//-------------------------------------------------
//
//   Class: DTTFMasksTester
//
//   L1 DT Track Finder Parameters Tester
//
//
//   $Date: 2009/05/12 09:53:37 $
//   $Revision: 1.1 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTTFMasksTester.h"


DTTFMasksTester::DTTFMasksTester(const edm::ParameterSet& ps) {}


DTTFMasksTester::~DTTFMasksTester() {}


void DTTFMasksTester::analyze(const edm::Event& e, const edm::EventSetup& c) {

  edm::ESHandle<L1MuDTTFMasks> dttfmsk;
  c.get<L1MuDTTFMasksRcd>().get( dttfmsk );
  dttfmsk->print();

}
