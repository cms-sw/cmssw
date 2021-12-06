/** Derived from DTSectorAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

using namespace std;

class RPCSectorAnalyzer : public edm::one::EDAnalyzer<> {
public:
  RPCSectorAnalyzer(const edm::ParameterSet& pset);

  ~RPCSectorAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

  const std::string& myName() { return myName_; }

private:
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> tokRPC_;
  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::ofstream ofos;
};

RPCSectorAnalyzer::RPCSectorAnalyzer(const edm::ParameterSet& /*iConfig*/)
    : tokRPC_{esConsumes<RPCGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
      dashedLineWidth_(104),
      dashedLine_(std::string(dashedLineWidth_, '-')),
      myName_("RPCSectorAnalyzer") {
  ofos.open("MytestOutput.out");
  std::cout << "======================== Opening output file" << std::endl;
}

RPCSectorAnalyzer::~RPCSectorAnalyzer() {
  ofos.close();
  std::cout << "======================== Closing output file" << std::endl;
}

void RPCSectorAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const RPCGeometry* pDD = &iSetup.getData(tokRPC_);

  std::cout << myName() << ": Analyzer..." << std::endl;
  std::cout << "start " << dashedLine_ << std::endl;

  std::cout << " Geometry node for RPCGeom is  " << &(*pDD) << std::endl;
  cout << " I have " << pDD->detTypes().size() << " detTypes" << endl;
  cout << " I have " << pDD->detUnits().size() << " detUnits" << endl;
  cout << " I have " << pDD->dets().size() << " dets" << endl;
  cout << " I have " << pDD->rolls().size() << " rolls" << endl;
  cout << " I have " << pDD->chambers().size() << " chambers" << endl;

  std::cout << myName() << ": Begin iteration over geometry..." << std::endl;
  std::cout << "iter " << dashedLine_ << std::endl;

  const double dPi = 3.14159265358;
  const double radToDeg = 180. / dPi;  //@@ Where to get pi from?

  for (auto it : pDD->dets()) {
    //      //----------------------- RPCCHAMBER TEST -------------------------------------------------------

    if (dynamic_cast<const RPCChamber*>(it) != nullptr) {
      const RPCChamber* ch = dynamic_cast<const RPCChamber*>(it);

      //RPCDetId detId=ch->id();

      std::vector<const RPCRoll*> rolls = (ch->rolls());
      for (auto& roll : rolls) {
        if (roll->id().region() == -1 && roll->id().station() > 0)  // &&
        // (*r)->id().sector() == 8)
        {
          std::cout << "RPCDetId = " << roll->id() << std::endl;
          RPCGeomServ geosvc(roll->id());
          LocalPoint centre(0., 0., 0.);
          GlobalPoint gc = roll->toGlobal(centre);
          double phifirs = double(roll->toGlobal(roll->centreOfStrip(1)).phi()) * radToDeg;
          if (phifirs < 0)
            phifirs = 360. + phifirs;
          double philast = double(roll->toGlobal(roll->centreOfStrip(roll->nstrips())).phi()) * radToDeg;
          if (philast < 0)
            philast = 360. + philast;
          double deltaphi = philast - phifirs;
          if (abs(deltaphi) > 180.) {
            if (deltaphi < 0) {
              deltaphi += 360.;
            } else if (deltaphi > 0) {
              deltaphi -= 360.;
            }
          }

          std::cout << " Position " << gc << " phi=" << double(gc.phi()) * radToDeg << " strip 1 " << phifirs
                    << " strip " << roll->nstrips() << " " << philast << std::endl;
          std::cout << geosvc.name() << " "
                    << "Anticlockwise ? ";
          if (geosvc.aclockwise()) {
            std::cout << "yes ";
          } else {
            std::cout << "no ";
          }
          if (deltaphi > 0. && geosvc.aclockwise()) {
            std::cout << " OK ";
          } else if (deltaphi < 0. && !geosvc.aclockwise()) {
            std::cout << " OK ";
          } else {
            std::cout << " NOTOK ";
          }
          std::cout << " /\\phi=" << deltaphi << std::endl;
        }
      }
    }
  }
  std::cout << dashedLine_ << " end" << std::endl;
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(RPCSectorAnalyzer);
