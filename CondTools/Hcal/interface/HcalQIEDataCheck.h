#ifndef HcalQIEDataCheck_h
#define HcalQIEDataCheck_h

//
// R.Ofierzynski 9.12.2007
//

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/HcalObjects/interface/HcalQIEData.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"

#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "CondFormats/DataRecord/interface/HcalQIEDataRcd.h"
#include "CondFormats/DataRecord/interface/HcalElectronicsMapRcd.h"

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"

class HcalQIEDataCheck : public edm::EDAnalyzer {
public:
  HcalQIEDataCheck(edm::ParameterSet const& ps);

  ~HcalQIEDataCheck() override;

  void analyze(const edm::Event& ev, const edm::EventSetup& es) override;

private:
  std::string outfile;
  std::string dumprefs;
  std::string dumpupdate;
  bool checkemapflag;
  bool validateflag;
  //  double epsilon;
  //  vecDetId getMissingDetIds(std::vector<HcalPedestalWidths> &);
};
#endif
