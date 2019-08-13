#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

using namespace cms_units::operators;

static long algorithm(dd4hep::Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      dd4hep::SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  // Header section
  double deltaPhi = args.value<double>("DeltaPhi");
  double deltaZ = args.value<double>("DeltaZ");
  int numberPhi = args.value<int>("NumberPhi");
  std::string material = args.value<std::string>("Material");
  std::vector<double> areaSection = args.value<std::vector<double> >("AreaSection");
  std::vector<double> rStart = args.value<std::vector<double> >("RadiusStart");
  std::vector<double> rEnd = args.value<std::vector<double> >("RadiusEnd");
  std::vector<int> bundle = args.value<std::vector<int> >("Bundles");
  double tilt = args.value<double>("TiltAngle");
  std::string childPrefix = args.value<std::string>("Child");
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Parent " << args.parentName() << " with " << bundle.size()
                               << " children with prefix " << childPrefix << ", material " << material << " with "
                               << numberPhi << " bundles along phi; width of"
                               << " mother " << deltaZ << " along Z, " << convertRadToDeg(deltaPhi)
                               << " along phi and with " << rStart.size() << " different bundle types";
  for (unsigned int i = 0; i < areaSection.size(); ++i)
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Child[" << i << "] Area " << areaSection[i] << " R at Start "
                                 << rStart[i] << " R at End " << rEnd[i];
  edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: NameSpace " << ns.name() << " Tilt Angle "
                               << convertRadToDeg(tilt) << " Bundle type at different positions";
  for (unsigned int i = 0; i < bundle.size(); ++i) {
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Position[" << i << "] "
                                 << " with Type " << bundle[i];
  }
#endif

  dd4hep::Volume mother = ns.volume(args.parentName());
  dd4hep::Material matter = ns.material(material);

  // Create the rotation matrices
  double dPhi = deltaPhi / numberPhi;
  std::vector<dd4hep::Rotation3D> rotation;
  for (int i = 0; i < numberPhi; ++i) {
    double phi = -0.5 * deltaPhi + (i + 0.5) * dPhi;
    dd4hep::Rotation3D rot = cms::makeRotation3D(90._deg, phi, 90._deg, (90._deg + phi), 0, 0);
#ifdef EDM_ML_DEBUG
    double phideg = convertRadToDeg(phi);
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Creating a new rotation " << 90 << "," << phideg << "," << 90
                                 << "," << (phideg + 90) << ", 0, 0";
#endif
    rotation.emplace_back(rot);
  }

  // Create the solids and logical parts
  std::vector<dd4hep::Volume> logs;
  for (unsigned int i = 0; i < areaSection.size(); ++i) {
    double r0 = rEnd[i] / std::cos(tilt);
    double dStart = areaSection[i] / (2 * dPhi * rStart[i]);
    double dEnd = areaSection[i] / (2 * dPhi * r0);
    std::string name = childPrefix + std::to_string(i);
    dd4hep::Solid solid = dd4hep::ConeSegment(
        0.5 * deltaZ, rStart[i] - dStart, rStart[i] + dStart, r0 - dEnd, r0 + dEnd, -0.5 * dPhi, 0.5 * dPhi);
    ns.addSolidNS(name, solid);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Creating a new solid " << name << " a cons with dZ " << deltaZ
                                 << " rStart " << rStart[i] - dStart << ":" << rStart[i] + dStart << " rEnd "
                                 << r0 - dEnd << ":" << r0 + dEnd << " Phi " << convertRadToDeg(-0.5 * dPhi) << ":"
                                 << convertRadToDeg(0.5 * dPhi);
#endif
    dd4hep::Volume log(name, solid, matter);
    ns.addVolumeNS(log);
    logs.emplace_back(log);
  }

  // Now posiiton them
  int copy = 0;
  int nY = static_cast<int>(bundle.size()) / numberPhi;
  for (unsigned int i = 0; i < bundle.size(); i++) {
    dd4hep::Position tran(0, 0, 0);
    int ir = static_cast<int>(i) / nY;
    if (ir >= numberPhi)
      ir = numberPhi - 1;
    int ib = bundle[i];
    copy++;
    if (ib >= 0 && ib < (int)(logs.size())) {
      mother.placeVolume(logs[ib], copy, dd4hep::Transform3D(rotation[ir], tran));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: " << logs[ib].name() << " number " << copy
                                   << " positioned in " << mother.name() << " at (0, 0, 0)" << tran << " with "
                                   << rotation[ir];
#endif
    }
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalFibreBundle, algorithm);
