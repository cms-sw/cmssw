/** Derived from DTRadiiAnalyzer by Nicola Amapane
 *
 *  \author M. Maggi - INFN Bari
 */

#include <memory>
#include <fstream>
#include <FWCore/Framework/interface/Frameworkfwd.h>

#include <FWCore/Framework/interface/one/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCRollSpecs.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <Geometry/CommonTopologies/interface/RectangularStripTopology.h>
#include <Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h>

#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>

class RPCRadiiAnalyzer : public edm::one::EDAnalyzer<> {
public:
  RPCRadiiAnalyzer(const edm::ParameterSet& pset);

  ~RPCRadiiAnalyzer() override;

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

RPCRadiiAnalyzer::RPCRadiiAnalyzer(const edm::ParameterSet& /*iConfig*/)
    : tokRPC_{esConsumes<RPCGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
      dashedLineWidth_(104),
      dashedLine_(std::string(dashedLineWidth_, '-')),
      myName_("RPCRadiiAnalyzer") {
  ofos.open("MytestOutput.out");
  edm::LogVerbatim("RPCGeometry") << "======================== Opening output file";
}

RPCRadiiAnalyzer::~RPCRadiiAnalyzer() {
  ofos.close();
  edm::LogVerbatim("RPCGeometry") << "======================== Closing output file";
}

void RPCRadiiAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const RPCGeometry* pDD = &iSetup.getData(tokRPC_);

  ofos << myName() << ": Analyzer..." << std::endl;
  ofos << "start " << dashedLine_ << std::endl;

  ofos << " Geometry node for RPCGeom is  " << &(*pDD) << std::endl;
  ofos << " I have " << pDD->detTypes().size() << " detTypes" << std::endl;
  ofos << " I have " << pDD->detUnits().size() << " detUnits" << std::endl;
  ofos << " I have " << pDD->dets().size() << " dets" << std::endl;
  ofos << " I have " << pDD->rolls().size() << " rolls" << std::endl;
  ofos << " I have " << pDD->chambers().size() << " chambers" << std::endl;

  ofos << myName() << ": Begin iteration over geometry..." << std::endl;
  ofos << "iter " << dashedLine_ << std::endl;

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
                                                                    //	   (*r)->id().ring() == 2)
        {
          //	    ofos<<"RPCDetId = "<<(*r)->id()<<std::endl;
          RPCGeomServ geosvc(roll->id());
          LocalPoint centre(0., 0., 0.);
          GlobalPoint gc = roll->toGlobal(centre);
          double phic = double(gc.phi()) * radToDeg;
          double radii = double(gc.perp());
          ofos << geosvc.name() << " phi=" << phic << " r=" << radii << " detName " << roll->specs()->detName()
                    << " s=" << roll->id().sector() << " subs=" << roll->id().subsector() << std::endl;
        }
      }
    }
  }
  edm::LogVerbatim("RPCGeometry") << dashedLine_ << " end";
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(RPCRadiiAnalyzer);
