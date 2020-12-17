#include "DataFormats/Math/interface/GeantUnits.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
#ifdef EDM_ML_DEBUG
  static constexpr double f2mm = (1.0 / dd4hep::mm);
#endif
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  std::string motherName = args.parentName();

  auto const& m_tiltAngle = args.value<double>("tiltAngle");       // Tilt  angle
  auto const& m_rMin = args.value<double>("rMin");                 // Inner radius
  auto const& m_rMax = args.value<double>("rMax");                 // Outer radius
  auto const& m_zoffset = args.value<double>("zoffset");           // Offset in z
  auto const& m_xyoffset = args.value<double>("xyoffset");         // Offset in x or y
  auto const& m_startCopyNo = args.value<int>("startCopyNo");      // Start copy Number
  auto const& m_incrCopyNo = args.value<int>("incrCopyNo");        // Increment copy Number
  auto const& m_childName = args.value<std::string>("ChildName");  // Children name
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Tilt Angle " << m_tiltAngle << " R " << (f2mm * m_rMin) << ":" << (f2mm * m_rMax)
                                << " Offset " << (f2mm * m_zoffset) << ":" << (f2mm * m_xyoffset) << " Copy "
                                << m_startCopyNo << ":" << m_incrCopyNo << " Child " << m_childName;

  edm::LogVerbatim("HGCalGeom") << "DDHGCalNoTaperEndcap: NameSpace " << ns.name() << "\tParent " << args.parentName();
#endif

  dd4hep::Volume parent = ns.volume(args.parentName());
  std::string name = ns.prepend(m_childName);

  const int ix[4] = {1, -1, -1, 1};
  const int iy[4] = {1, 1, -1, -1};
  int copyNo = m_startCopyNo;
  for (int i = 0; i < 4; ++i) {
    int xQuadrant = ix[i];
    int yQuadrant = iy[i];
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Create quarter " << xQuadrant << ":" << yQuadrant;
#endif
    double tiltAngle = m_tiltAngle;
    double xphi = xQuadrant * tiltAngle;
    double yphi = yQuadrant * tiltAngle;
    double theta = 90._deg;
    double phiX = 0.0;
    double phiY = theta;
    double phiZ = 3 * theta;
    double offsetZ = m_zoffset;
    double offsetXY = m_xyoffset;
    double offsetX = xQuadrant * 0.5 * offsetXY;
    double offsetY = yQuadrant * 0.5 * offsetXY;

#ifdef EDM_ML_DEBUG
    int rowmax(0), column(0);
#endif
    while (std::abs(offsetX) < m_rMax) {
#ifdef EDM_ML_DEBUG
      column++;
      int row(0);
#endif
      while (std::abs(offsetY) < m_rMax) {
#ifdef EDM_ML_DEBUG
        row++;
#endif
        double limit1 = sqrt((offsetX + 0.5 * xQuadrant * offsetXY) * (offsetX + 0.5 * xQuadrant * offsetXY) +
                             (offsetY + 0.5 * yQuadrant * offsetXY) * (offsetY + 0.5 * yQuadrant * offsetXY));
        double limit2 = sqrt((offsetX - 0.5 * xQuadrant * offsetXY) * (offsetX - 0.5 * xQuadrant * offsetXY) +
                             (offsetY - 0.5 * yQuadrant * offsetXY) * (offsetY - 0.5 * yQuadrant * offsetXY));
        // Make sure we do not add supermodules in rMin area
        if (limit2 > m_rMin && limit1 < m_rMax) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << m_childName << " copyNo = " << copyNo << " (" << column << "," << row
                                        << "): offsetX,Y = " << (f2mm * offsetX) << "," << (f2mm * offsetY)
                                        << " limit=" << (f2mm * limit1) << ":" << (f2mm * limit2)
                                        << " rMin, rMax = " << (f2mm * m_rMin) << "," << (f2mm * m_rMax);
#endif

          dd4hep::Rotation3D rotation = (cms::makeRotation3D(theta, phiX, theta + yphi, phiY, -yphi, phiZ) *
                                         cms::makeRotation3D(theta + xphi, phiX, 90._deg, 90._deg, xphi, 0.0));

          dd4hep::Position tran(offsetX, offsetY, offsetZ);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << "Module " << copyNo << ": location = (" << (f2mm * offsetX) << ", "
                                        << (f2mm * offsetY) << ", " << (f2mm * offsetZ) << ") Rotation " << rotation;
#endif
          parent.placeVolume(ns.volume(name), copyNo, dd4hep::Transform3D(rotation, tran));

          copyNo += m_incrCopyNo;
        } else {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("HGCalGeom") << " (" << column << "," << row << "): offsetX,Y = " << (f2mm * offsetX) << ","
                                        << (f2mm * offsetY) << " is out of limit=" << (f2mm * limit1) << ":"
                                        << (f2mm * limit2) << " rMin, rMax = " << (f2mm * m_rMin) << ","
                                        << (f2mm * m_rMax);
#endif
        }
        yphi += yQuadrant * 2. * tiltAngle;
        offsetY += yQuadrant * offsetXY;
      }
#ifdef EDM_ML_DEBUG
      if (row > rowmax)
        rowmax = row;
#endif
      xphi += xQuadrant * 2. * tiltAngle;
      yphi = yQuadrant * tiltAngle;
      offsetY = yQuadrant * 0.5 * offsetXY;
      offsetX += xQuadrant * offsetXY;
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << rowmax << " rows and " << column << " columns in quadrant " << xQuadrant << ":"
                                  << yQuadrant;
#endif
  }

  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_hgcal_DDHGCalNoTaperEndcap, algorithm)
