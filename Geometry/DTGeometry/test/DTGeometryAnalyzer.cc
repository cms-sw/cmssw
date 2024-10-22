/** \file
 *
 *  \author N. Amapane - CERN
 */

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/Rounding.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>  // for setw() etc.
#include <vector>

using namespace std;
using namespace cms_rounding;

class DTGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  DTGeometryAnalyzer(const edm::ParameterSet& pset);
  ~DTGeometryAnalyzer() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const string& myName() { return myName_; }
  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> tokDT_;
  const int dashedLineWidth_;
  const string dashedLine_;
  const string myName_;
  double tolerance_;
};

DTGeometryAnalyzer::DTGeometryAnalyzer(const edm::ParameterSet& iConfig)
    : tokDT_{esConsumes<DTGeometry, MuonGeometryRecord>(edm::ESInputTag{})},
      dashedLineWidth_(104),
      dashedLine_(string(dashedLineWidth_, '-')),
      myName_("DTGeometryAnalyzer"),
      tolerance_(iConfig.getUntrackedParameter<double>("tolerance", 1.e-23)) {}

DTGeometryAnalyzer::~DTGeometryAnalyzer() {}

void DTGeometryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& pDD = iSetup.getData(tokDT_);

  edm::LogVerbatim("DTGeometry") << myName() << ": Analyzer...";
  edm::LogVerbatim("DTGeometry") << "start " << dashedLine_;

  edm::LogVerbatim("DTGeometry") << " Geometry node for DTGeom is  " << &pDD;
  edm::LogVerbatim("DTGeometry") << " I have " << pDD.detTypes().size() << " detTypes";
  edm::LogVerbatim("DTGeometry") << " I have " << pDD.detUnits().size() << " detUnits";
  edm::LogVerbatim("DTGeometry") << " I have " << pDD.dets().size() << " dets";
  edm::LogVerbatim("DTGeometry") << " I have " << pDD.layers().size() << " layers";
  edm::LogVerbatim("DTGeometry") << " I have " << pDD.superLayers().size() << " superlayers";
  edm::LogVerbatim("DTGeometry") << " I have " << pDD.chambers().size() << " chambers";

  edm::LogVerbatim("DTGeometry") << myName() << ": Begin iteration over geometry...";
  edm::LogVerbatim("DTGeometry") << "iter " << dashedLine_;

  // check detUnits
  for (const auto& det : pDD.detUnits()) {
    DetId detId = det->geographicalId();
    int id = detId();  // or detId.rawId()
    const GeomDet* gdet_ = pDD.idToDet(detId);
    const GeomDetUnit* gdet = pDD.idToDetUnit(detId);
    const DTLayer* lay = dynamic_cast<const DTLayer*>(gdet);
    edm::LogVerbatim("DTGeometry") << "GeomDetUnit is of type " << detId.det() << " and raw id = " << id;
    assert(det == gdet);
    assert(gdet_ == gdet);
    assert(gdet_ == lay);
  }

  // check layers
  edm::LogVerbatim("DTGeometry") << "LAYERS " << dashedLine_;
  for (auto det : pDD.layers()) {
    const DTTopology& topo = det->specificTopology();
    const BoundPlane& surf = det->surface();
    edm::LogVerbatim("DTGeometry") << "Layer " << det->id() << " SL " << det->superLayer()->id() << " chamber "
                                   << det->chamber()->id() << " Topology W/H/L: " << topo.cellWidth() << "/"
                                   << topo.cellHeight() << "/" << topo.cellLenght() << " first/last/# wire "
                                   << topo.firstChannel() << "/" << topo.lastChannel() << "/" << topo.channels()
                                   << " Position " << surf.position() << " normVect "
                                   << roundVecIfNear0(surf.normalVector(), tolerance_)
                                   << " bounds W/H/L: " << surf.bounds().width() << "/" << surf.bounds().thickness()
                                   << "/" << surf.bounds().length();
  }

  // check superlayers
  edm::LogVerbatim("DTGeometry") << "SUPERLAYERS " << dashedLine_;
  for (auto det : pDD.superLayers()) {
    const BoundPlane& surf = det->surface();
    edm::LogVerbatim("DTGeometry") << "SuperLayer " << det->id() << " chamber " << det->chamber()->id() << " Position "
                                   << surf.position() << " normVect "
                                   << roundVecIfNear0(surf.normalVector(), tolerance_)
                                   << " bounds W/H/L: " << surf.bounds().width() << "/" << surf.bounds().thickness()
                                   << "/" << surf.bounds().length();
  }

  // check chamber
  edm::LogVerbatim("DTGeometry") << "CHAMBERS " << dashedLine_;
  for (auto det : pDD.chambers()) {
    //edm::LogVerbatim("DTGeometry") << "Chamber " << (*det)->geographicalId().det();
    const BoundPlane& surf = det->surface();
    //edm::LogVerbatim("DTGeometry") << "surf " << &surf;
    edm::LogVerbatim("DTGeometry") << "Chamber " << det->id() << " Position " << surf.position() << " normVect "
                                   << roundVecIfNear0(surf.normalVector(), tolerance_)
                                   << " bounds W/H/L: " << surf.bounds().width() << "/" << surf.bounds().thickness()
                                   << "/" << surf.bounds().length();
  }
  edm::LogVerbatim("DTGeometry") << "END " << dashedLine_;

  // Check chamber(), layer(), superlayer(), idToDet()
  for (int w = -2; w <= 2; ++w) {
    for (int st = 1; st <= 4; ++st) {
      for (int se = 1; se <= ((st == 4) ? 14 : 12); ++se) {
        DTChamberId id(w, st, se);
        const DTChamber* ch = pDD.chamber(id);
        if (!ch)
          edm::LogVerbatim("DTGeometry") << "ERROR ch not found " << id;
        else {
          if (id != ch->id())
            edm::LogVerbatim("DTGeometry")
                << "ERROR: got wrong chamber: Cerco camera " << id << " e trovo " << ch->id();
          // test idToDet for chamber
          const GeomDet* gdetc = pDD.idToDet(id);
          assert(gdetc == ch);

          for (int sl = 1; sl <= 3; ++sl) {
            if (sl == 2 && st == 4)
              continue;
            DTSuperLayerId slid(id, sl);
            const DTSuperLayer* dtsl = pDD.superLayer(slid);
            if (!dtsl)
              edm::LogVerbatim("DTGeometry") << "ERROR sl not found " << slid;
            else {
              if (slid != dtsl->id())
                edm::LogVerbatim("DTGeometry") << "ERROR: got wrong sl! Cerco sl " << slid << " e trovo " << dtsl->id();
              // test idToDet for superLayer
              const GeomDet* gdets = pDD.idToDet(slid);
              assert(gdets == dtsl);

              for (int l = 1; l <= 4; ++l) {
                DTLayerId lid(slid, l);
                const DTLayer* lay = pDD.layer(lid);
                if (!lay)
                  edm::LogVerbatim("DTGeometry") << "ERROR lay not found " << lid;
                if (lid != lay->id())
                  edm::LogVerbatim("DTGeometry")
                      << "ERROR: got wrong layer Cerco lay  " << lid << " e trovo " << lay->id();
                // test idToDet for layer
                const GeomDet* gdetl = pDD.idToDet(lid);
                assert(gdetl == lay);
              }
            }
          }
        }
      }
    }
  }
  edm::LogVerbatim("DTGeometry") << "END " << dashedLine_;
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(DTGeometryAnalyzer);
