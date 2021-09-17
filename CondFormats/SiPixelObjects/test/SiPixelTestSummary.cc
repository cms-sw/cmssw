//#include <memory>

#include <string.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/SiPixelObjects/interface/SiPixelDetSummary.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

using std::string;
using namespace std;

// ----------------------------------------------------------------------
class SiPixelTestSummary : public edm::EDAnalyzer {
public:
  explicit SiPixelTestSummary(const edm::ParameterSet&) {}
  ~SiPixelTestSummary();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
};

// ----------------------------------------------------------------------
SiPixelTestSummary::~SiPixelTestSummary() {}

// ----------------------------------------------------------------------
void SiPixelTestSummary::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  std::cout << "====== SiPixelTestSummary begin" << std::endl;

  SiPixelDetSummary a(1);

  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get(pDD);

  cout << "**********************************************************************" << endl;
  cout << " *** Geometry node for TrackerGeom is  " << &(*pDD) << std::endl;
  cout << " *** I have " << pDD->dets().size() << " detectors" << std::endl;
  cout << " *** I have " << pDD->detTypes().size() << " types" << std::endl;

  for (TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++) {
    if (dynamic_cast<PixelGeomDetUnit const*>((*it)) != 0) {
      DetId detId = (*it)->geographicalId();
      a.add(detId);
    }
  }

  bool toyExample(false);
  if (toyExample) {
    a.add(302055684, 2.3);  // BPix_BmI_SEC4_LYR1_LDR5_MOD4  302055684
    a.add(302125072, 2.4);  // BPix_BmO_SEC4_LYR2_LDR8_MOD1  302125072
    a.add(302188552, 2.1);  // BPix_BmI_SEC2_LYR3_LDR4_MOD3  302188552
    a.add(352388356, 1.5);  // FPix_BpI_D1_BLD6_PNL1_MOD1  352388356
    a.add(352390664, 2.2);  // FPix_BpI_D1_BLD4_PNL2_MOD2  352390664
    a.add(352453892, 2.5);  // FPix_BpI_D2_BLD6_PNL1_MOD1  352453892
    a.add(352457224, 2.1);  // FPix_BpI_D2_BLD3_PNL2_MOD2  352457224
    a.add(344005892, 2.6);  // FPix_BmO_D1_BLD1_PNL1_MOD1  344005892
    a.add(344008204, 2.0);  // FPix_BmO_D1_BLD3_PNL2_MOD3  344008204
    a.add(344071432, 2.4);  // FPix_BmO_D2_BLD1_PNL1_MOD2  344071432
    //  a.add(344073740);  // FPix_BmO_D2_BLD3_PNL2_MOD3  344073740
  }

  cout << endl;
  cout << "Testing printout" << endl;
  stringstream bla;
  a.print(bla, false);
  cout << bla.str() << endl;

  cout << endl;
  cout << "Testing map" << endl;
  map<int, int> b = a.getCounts();
  for (map<int, int>::const_iterator bIt = b.begin(); bIt != b.end(); ++bIt) {
    cout << bIt->first << " -> " << bIt->second << endl;
  }

  std::cout << "====== SiPixelTestSummary end" << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelTestSummary);
