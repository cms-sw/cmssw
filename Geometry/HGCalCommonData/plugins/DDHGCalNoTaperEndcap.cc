#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "DataFormats/Math/interface/angle_units.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

//#define EDM_ML_DEBUG
using namespace angle_units::operators;

class DDHGCalNoTaperEndcap : public DDAlgorithm {
public:
  DDHGCalNoTaperEndcap(void);

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  int createQuarter(DDCompactView& cpv, int xQuadrant, int yQuadrant, int startCopyNo);

  double m_startAngle;        // Start angle
  double m_tiltAngle;         // Tilt  angle
  int m_invert;               // Inverted or forward
  double m_rMin;              // Inner radius
  double m_rMax;              // Outer radius
  double m_zoffset;           // Offset in z
  double m_xyoffset;          // Offset in x or y
  int m_n;                    // Mumber of copies
  int m_startCopyNo;          // Start copy Number
  int m_incrCopyNo;           // Increment copy Number
  std::string m_childName;    // Children name
  std::string m_idNameSpace;  // Namespace of this and ALL sub-parts
};

DDHGCalNoTaperEndcap::DDHGCalNoTaperEndcap() {
  edm::LogVerbatim("HGCalGeom") << "DDHGCalNoTaperEndcap test: Creating an instance";
}

void DDHGCalNoTaperEndcap::initialize(const DDNumericArguments& nArgs,
                                      const DDVectorArguments& vArgs,
                                      const DDMapArguments&,
                                      const DDStringArguments& sArgs,
                                      const DDStringVectorArguments&) {
  m_tiltAngle = nArgs["tiltAngle"];
  m_invert = int(nArgs["invert"]);
  m_rMin = int(nArgs["rMin"]);
  m_rMax = int(nArgs["rMax"]);
  m_zoffset = nArgs["zoffset"];
  m_xyoffset = nArgs["xyoffset"];
  m_n = int(nArgs["n"]);
  m_startCopyNo = int(nArgs["startCopyNo"]);
  m_incrCopyNo = int(nArgs["incrCopyNo"]);
  m_childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Tilt Angle " << m_tiltAngle << " Invert " << m_invert << " R " << m_rMin << ":"
                                << m_rMax << " Offset " << m_zoffset << ":" << m_xyoffset << " N " << m_n << " Copy "
                                << m_startCopyNo << ":" << m_incrCopyNo << " Child " << m_childName;
#endif

  m_idNameSpace = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalNoTaperEndcap: NameSpace " << m_idNameSpace << "\tParent "
                                << parent().name();
#endif
}

void DDHGCalNoTaperEndcap::execute(DDCompactView& cpv) {
  int lastCopyNo = m_startCopyNo;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create quarter 1:1";
#endif
  lastCopyNo = createQuarter(cpv, 1, 1, lastCopyNo);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create quarter -1:1";
#endif
  lastCopyNo = createQuarter(cpv, -1, 1, lastCopyNo);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create quarter -1:-1";
#endif
  lastCopyNo = createQuarter(cpv, -1, -1, lastCopyNo);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Create quarter 1:-1";
#endif
  createQuarter(cpv, 1, -1, lastCopyNo);
}

int DDHGCalNoTaperEndcap::createQuarter(DDCompactView& cpv, int xQuadrant, int yQuadrant, int startCopyNo) {
  int copyNo = startCopyNo;
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
                                      << "): offsetX,Y = " << offsetX << "," << offsetY << " limit=" << limit1 << ":"
                                      << limit2 << " rMin, rMax = " << m_rMin << "," << m_rMax;
#endif
        DDRotation rotation;
        std::string rotstr("NULL");

        // Check if we've already created the rotation matrix
        rotstr = "R";
        rotstr += std::to_string(copyNo);
        rotation = DDRotation(DDName(rotstr));
        if (!rotation) {
          rotation = DDrot(DDName(rotstr, m_idNameSpace),
                           std::make_unique<DDRotationMatrix>(
                               *DDcreateRotationMatrix(theta, phiX, theta + yphi, phiY, -yphi, phiZ) *
                               (*DDcreateRotationMatrix(theta + xphi, phiX, 90._deg, 90._deg, xphi, 0.0))));
        }

        DDTranslation tran(offsetX, offsetY, offsetZ);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Module " << copyNo << ": location = " << tran << " Rotation " << rotation;
#endif
        DDName parentName = parent().name();
        cpv.position(DDName(m_childName), parentName, copyNo, tran, rotation);

        copyNo += m_incrCopyNo;
      } else {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << " (" << column << "," << row << "): offsetX,Y = " << offsetX << "," << offsetY
                                      << " is out of limit=" << limit1 << ":" << limit2 << " rMin, rMax = " << m_rMin
                                      << "," << m_rMax;
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
  return copyNo;
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDHGCalNoTaperEndcap, "hgcal:DDHGCalNoTaperEndcap");
