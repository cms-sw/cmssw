#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
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
                               << " mother " << cms::convert2mm(deltaZ) << " along Z, " << convertRadToDeg(deltaPhi)
                               << " along phi and with " << rStart.size() << " different bundle types";
  for (unsigned int i = 0; i < areaSection.size(); ++i)
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Child[" << i << "] Area "
                                 << cms::convert2mm(areaSection[i] / dd4hep::mm) << " R at Start "
                                 << cms::convert2mm(rStart[i]) << " R at End " << cms::convert2mm(rEnd[i]);
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
  dd4hep::Rotation3D rot;
  for (int i = 0; i < numberPhi; ++i) {
    double phi = -0.5 * deltaPhi + (i + 0.5) * dPhi;
    rot = dd4hep::RotationZ(phi);
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
        name, 0.5 * deltaZ, rStart[i] - dStart, rStart[i] + dStart, r0 - dEnd, r0 + dEnd, -0.5 * dPhi, 0.5 * dPhi);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Creating a new solid " << name << " a cons with dZ "
                                 << cms::convert2mm(deltaZ) << " rStart " << cms::convert2mm(rStart[i] - dStart) << ":"
                                 << cms::convert2mm(rStart[i] + dStart) << " rEnd " << cms::convert2mm(r0 - dEnd) << ":"
                                 << cms::convert2mm(r0 + dEnd) << " Phi " << convertRadToDeg(-0.5 * dPhi) << ":"
                                 << convertRadToDeg(0.5 * dPhi);
#endif
    dd4hep::Volume log(name, solid, matter);
    logs.emplace_back(log);
  }

  // Now posiiton them
  int copy = 0;
  int nY = static_cast<int>(bundle.size()) / numberPhi;
  for (unsigned int i = 0; i < bundle.size(); i++) {
    int ir = static_cast<int>(i) / nY;
    if (ir >= numberPhi)
      ir = numberPhi - 1;
    int ib = bundle[i];
    copy++;
    if (ib >= 0 && ib < (int)(logs.size())) {
      mother.placeVolume(logs[ib], copy, rotation[ir]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: " << logs[ib].name() << " number " << copy
                                   << " positioned in " << mother.name() << " at (0, 0, 0) with " << rotation[ir];
#endif
    }
  }
  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hcal_DDHCalFibreBundle, algorithm);
