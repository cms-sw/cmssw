#include <fstream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

//
//
// class declaration
//

class ComputeTransformation : public edm::EDAnalyzer {
public:
  explicit ComputeTransformation(const edm::ParameterSet&);
  ~ComputeTransformation() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  std::string m_fileName;
};

//
// constructors and destructor
//
ComputeTransformation::ComputeTransformation(const edm::ParameterSet& iConfig)
    : m_fileName(iConfig.getParameter<std::string>("fileName")) {}

ComputeTransformation::~ComputeTransformation() {}

void ComputeTransformation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<DTGeometry> dtGeometry;
  iSetup.get<MuonGeometryRecord>().get(dtGeometry);

  edm::ESHandle<CSCGeometry> cscGeometry;
  iSetup.get<MuonGeometryRecord>().get(cscGeometry);

  std::ofstream output;
  output.open(m_fileName.c_str());
  output << "rotation = {}" << std::endl;

  for (int wheel = -2; wheel <= 2; wheel++) {
    for (int station = 1; station <= 4; station++) {
      for (int sector = 1; sector <= 14; sector++) {
        if (station != 4 and sector > 12)
          continue;
        DTChamberId id(wheel, station, sector);
        // globalcoords = rot * localcoords
        Surface::RotationType rot = dtGeometry->idToDet(id)->surface().rotation();
        output << "rotation[\"DT\", " << wheel << ", " << station << ", 0, " << sector << "] = [[" << rot.xx() << ", "
               << rot.xy() << ", " << rot.xz() << "], [" << rot.yx() << ", " << rot.yy() << ", " << rot.yz() << "], ["
               << rot.zx() << ", " << rot.zy() << ", " << rot.zz() << "]]" << std::endl;
      }
    }
  }

  for (int endcap = 1; endcap <= 2; endcap++) {
    for (int station = 1; station <= 4; station++) {
      for (int ring = 1; ring <= 3; ring++) {
        if (station > 1 and ring == 3)
          continue;
        for (int sector = 1; sector <= 36; sector++) {
          if (station > 1 && ring == 1 && sector > 18)
            continue;
          CSCDetId id(endcap, station, ring, sector);
          // globalcoords = rot * localcoords
          Surface::RotationType rot = cscGeometry->idToDet(id)->surface().rotation();
          output << "rotation[\"CSC\", " << endcap << ", " << station << ", " << ring << ", " << sector << "] = [["
                 << rot.xx() << ", " << rot.xy() << ", " << rot.xz() << "], [" << rot.yx() << ", " << rot.yy() << ", "
                 << rot.yz() << "], [" << rot.zx() << ", " << rot.zy() << ", " << rot.zz() << "]]" << std::endl;
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ComputeTransformation);
