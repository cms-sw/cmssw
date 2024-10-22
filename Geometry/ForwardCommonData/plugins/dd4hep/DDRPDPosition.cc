#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DD4hep/DetFactoryHelper.h"

#include <string>
#include <vector>
#include <sstream>

//#define EDM_ML_DEBUG

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  const auto& xpos = args.value<std::vector<double> >("positionX");
  const auto& ypos = args.value<double>("positionY");
  const auto& zpos = args.value<double>("positionZ");
  const auto& childName = args.value<std::string>("ChildName");

  dd4hep::Volume parent = ns.volume(args.parentName());
  dd4hep::Volume child = ns.volume(ns.prepend(childName));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ForwardGeom") << "DDRPDPosition: Parameters for positioning-- " << xpos.size() << " copies of "
                                  << child.name() << " to be positioned inside " << parent.name() << " at y = " << ypos
                                  << ", z = " << zpos << " and at x = (";
  std::ostringstream st1;
  for (const auto& x : xpos)
    st1 << x << " ";
  edm::LogVerbatim("ForwardGeom") << st1.str() << ")";
#endif

  for (unsigned int jj = 0; jj < xpos.size(); jj++) {
    dd4hep::Position tran(xpos[jj], ypos, zpos);
    parent.placeVolume(child, jj + 1, tran);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardGeom") << "DDRPDPosition: " << child.name() << " number " << jj + 1 << " positioned in "
                                    << parent.name() << " at " << tran << " with no rotation";
#endif
  }
  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_rpdalgo_DDRPDPosition, algorithm)
