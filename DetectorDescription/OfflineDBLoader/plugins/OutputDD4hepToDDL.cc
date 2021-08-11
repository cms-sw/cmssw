#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h"
#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "TGeoManager.h"

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

class OutputDD4hepToDDL : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit OutputDD4hepToDDL(const edm::ParameterSet &iConfig);
  ~OutputDD4hepToDDL() override;

  void beginJob() override {}
  void beginRun(edm::Run const &iEvent, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &iEvent, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> cpvTokendd4hep_;
  std::string m_fname;
  std::ostream *m_xos;
};

OutputDD4hepToDDL::OutputDD4hepToDDL(const edm::ParameterSet &iConfig)
    : cpvTokendd4hep_(esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", "make-payload"))), m_fname() {
  m_fname = iConfig.getUntrackedParameter<std::string>("fileName");
  if (m_fname.empty()) {
    m_xos = &std::cout;
  } else {
    m_xos = new std::ofstream(m_fname.c_str());
  }
  (*m_xos) << "<?xml version=\"1.0\"?>" << std::endl;
  (*m_xos) << "<DDDefinition>" << std::endl;
}

OutputDD4hepToDDL::~OutputDD4hepToDDL() {
  (*m_xos) << "</DDDefinition>" << std::endl;
  (*m_xos) << std::endl;
  m_xos->flush();
}

void OutputDD4hepToDDL::beginRun(const edm::Run &, edm::EventSetup const &es) {
  std::cout << "OutputDD4hepToDDL::beginRun" << std::endl;

  edm::ESTransientHandle<cms::DDCompactView> cpv = es.getTransientHandle(cpvTokendd4hep_);

  const cms::DDDetector *det = cpv->detector();
  const dd4hep::Detector &detector = *det->description();
  const dd4hep::SpecParRegistry &specStore = det->specpars();

  DDCoreToDDXMLOutput out;

  std::string rn = m_fname;
  size_t foundLastDot = rn.find_last_of('.');
  size_t foundLastSlash = rn.find_last_of('/');
  if (foundLastSlash > foundLastDot && foundLastSlash != std::string::npos) {
    std::cout << "What? last . before last / in path for filename... this should die..." << std::endl;
  }
  std::string ns_("none");
  if (foundLastDot != std::string::npos && foundLastSlash != std::string::npos) {
    ns_ = rn.substr(foundLastSlash, foundLastDot);
  } else if (foundLastDot != std::string::npos) {
    ns_ = rn.substr(0, foundLastDot);
  } else {
    std::cout << "What? no file name? Attempt at namespace =\"" << ns_ << "\" filename was " << m_fname << std::endl;
  }
  std::cout << "m_fname = " << m_fname << " namespace = " << ns_ << std::endl;

  (*m_xos) << std::fixed << std::setprecision(5);  // Precison to 1e-5 mm
  cms::DDParsingContext *const parsingContext = detector.extension<cms::DDParsingContext>();

  {
    // Add rotation for reference and to ease validation
    using namespace angle_units::operators;
    cms::DDNamespace nameSpace(*parsingContext);
    dd4hep::Rotation3D rotation(1., 0., 0., 0., 1., 0., 0., 0., -1.);
    rotation = rotation * dd4hep::RotationY(1._pi);
    nameSpace.addRotation("ebalgo:reflZRotY", rotation);
  }

  (*m_xos) << "<PosPartSection label=\"" << ns_ << "\">" << std::endl;
  const TGeoManager &mgr = detector.manager();
  for (const auto &&iter : *mgr.GetListOfVolumes()) {
    auto *vol = dynamic_cast<TGeoVolume *>(iter);
    int numDaughters = vol->GetNdaughters();
    if (numDaughters > 0) {
      auto nodeArray = vol->GetNodes();
      for (int index = 0; index < numDaughters; ++index) {
        auto *node = dynamic_cast<TGeoNode *>((*nodeArray)[index]);
        auto *childVol = node->GetVolume();
        out.position(*vol, *node, childVol->GetName(), *parsingContext, *m_xos);
      }
    }
  }
  (*m_xos) << "</PosPartSection>" << std::endl;

  (*m_xos) << "<MaterialSection label=\"" << ns_ << "\">" << std::endl;
  for (const auto &&iter : *mgr.GetListOfMaterials()) {
    out.element(dynamic_cast<const TGeoMaterial *>(iter), *m_xos);
  }
  // Output composite materials
  for (const auto &it : parsingContext->allCompMaterials) {
    out.material(it, *m_xos);
  }
  (*m_xos) << "</MaterialSection>" << std::endl;
  (*m_xos) << "<RotationSection label=\"" << ns_ << "\">" << std::endl;
  (*m_xos) << std::fixed << std::setprecision(10);  // Rounds angles to integer values w/o loss of accuracy
  // rotRevMap has the unique rotations
  for (const auto &rotPair : parsingContext->rotRevMap) {
    out.rotation(parsingContext->rotations[rotPair.second], *m_xos, *parsingContext, rotPair.second);
  }
  (*m_xos) << "</RotationSection>" << std::endl;

  (*m_xos) << std::fixed << std::setprecision(5);
  (*m_xos) << "<SolidSection label=\"" << ns_ << "\">" << std::endl;
  for (const auto &&iter : *mgr.GetListOfShapes()) {
    auto *shape = dynamic_cast<TGeoShape *>(iter);
    if (shape->IsValid()) {
      dd4hep::Solid solid(shape);
      if (strlen(shape->GetTitle()) > 1) {
        out.solid(solid, *parsingContext, *m_xos);
      } else {
        std::string name(DDCoreToDDXMLOutput::trimShapeName(shape->GetName()));
        if (name != "Box" && name != "Tubs") {  // Skip solids with degenerate names
          if (dd4hep::isA<dd4hep::Tube>(solid)) {
            shape->SetTitle("Tube");
            out.solid(solid, *parsingContext, *m_xos);
          } else if (dd4hep::isA<dd4hep::Box>(solid)) {
            shape->SetTitle("Box");
            out.solid(solid, *parsingContext, *m_xos);
          } else if (dd4hep::isA<dd4hep::Trd1>(solid)) {
            shape->SetTitle("Trd1");
            out.solid(solid, *parsingContext, *m_xos);
          } else
            std::cout << "Division solid not a box, trd1, or tube = " << solid.name() << std::endl;
        }
      }
    }
  }
  for (const auto &asmEntry : parsingContext->assemblySolids) {
    (*m_xos) << "<Assembly name=\"" << asmEntry << "\"/>" << std::endl;
  }
  (*m_xos) << "</SolidSection>" << std::endl;

  (*m_xos) << "<LogicalPartSection label=\"" << ns_ << "\">" << std::endl;
  for (const auto &&iter : *mgr.GetListOfVolumes()) {
    auto *vol = dynamic_cast<TGeoVolume *>(iter);
    // Skip unused logical parts and assemblies, which are listed separately
    if (vol->GetRefCount() > 0 && vol->IsAssembly() == false) {
      out.logicalPart(*vol, *m_xos);
    }
  }
  for (const auto &asEntry : parsingContext->assemblies) {
    out.logicalPart(asEntry.first, *m_xos);
  }
  (*m_xos) << "</LogicalPartSection>" << std::endl;

  (*m_xos) << std::fixed << std::setprecision(10);  // Some Tracker values specified to 1e-9
  (*m_xos) << "<SpecParSection label=\"" << ns_ << "\">" << std::endl;
  for (const auto &specPar : specStore.specpars) {
    out.specpar(specPar.first, specPar.second, *m_xos);
  }
  (*m_xos) << "</SpecParSection>" << std::endl;
}

DEFINE_FWK_MODULE(OutputDD4hepToDDL);
