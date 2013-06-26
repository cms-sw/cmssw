//-------------------------------------------------
//
//   Class: DTTFParametersTester
//
//   L1 DT Track Finder Parameters Tester
//
//
//   $Date: 2009/05/04 09:26:10 $
//   $Revision: 1.1 $
//
//   Author :
//   J. Troconiz              UAM Madrid
//
//--------------------------------------------------

#include "L1TriggerConfig/DTTrackFinder/interface/DTTFParametersTester.h"


DTTFParametersTester::DTTFParametersTester(const edm::ParameterSet& ps) {}


DTTFParametersTester::~DTTFParametersTester() {}


void DTTFParametersTester::analyze(const edm::Event& e, const edm::EventSetup& c) {

  edm::ESHandle<L1MuDTTFParameters> dttfpar;
  c.get<L1MuDTTFParametersRcd>().get( dttfpar );
  dttfpar->print();

}
