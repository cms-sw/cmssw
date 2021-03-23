
//////////////////////////////////////////////////////////////////////////////
// File: DDEcalEndcapAlgo.cc
// Description: Geometry factory class for Ecal Barrel
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <CLHEP/Geometry/Transform3D.h>

// Header files for endcap supercrystal geometry
#include "Geometry/EcalCommonData/interface/DDEcalEndcapTrap.h"

#include <map>
#include <string>
#include <vector>
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "Geometry/CaloGeometry/interface/EcalTrapezoidParameters.h"
#include "CLHEP/Geometry/Transform3D.h"

//#define EDM_ML_DEBUG

class DDEcalEndcapAlgo : public DDAlgorithm {
public:
  typedef EcalTrapezoidParameters Trap;
  typedef HepGeom::Point3D<double> Pt3D;
  typedef HepGeom::Transform3D Tf3D;
  typedef HepGeom::ReflectZ3D RfZ3D;
  typedef HepGeom::Translate3D Tl3D;
  typedef HepGeom::Rotate3D Ro3D;
  typedef HepGeom::RotateZ3D RoZ3D;
  typedef HepGeom::RotateY3D RoY3D;
  typedef HepGeom::RotateX3D RoX3D;

  typedef CLHEP::Hep3Vector Vec3;
  typedef CLHEP::HepRotation Rota;

  //Constructor and Destructor
  DDEcalEndcapAlgo();
  ~DDEcalEndcapAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;
  void execute(DDCompactView& cpv) override;

  //  New methods for SC geometry
  void EEPositionCRs(const DDName& pName, const DDTranslation& offset, const int iSCType, DDCompactView& cpv);

  void EECreateSC(const unsigned int iSCType, DDCompactView& cpv);

  void EECreateCR();

  void EEPosSC(const int iCol, const int iRow, DDName EEDeeName);

  unsigned int EEGetSCType(const unsigned int iCol, const unsigned int iRow);

  DDName EEGetSCName(const int iCol, const int iRow);

  std::vector<double> EEGetSCCtrs(const int iCol, const int iRow);

  DDMaterial ddmat(const std::string& s) const;
  DDName ddname(const std::string& s) const;
  DDRotation myrot(const std::string& s, const DDRotationMatrix& r) const;

  const std::string& idNameSpace() const { return m_idNameSpace; }

  // endcap parent volume
  DDMaterial eeMat() const { return ddmat(m_EEMat); }
  double eezOff() const { return m_EEzOff; }

  DDName eeQuaName() const { return ddname(m_EEQuaName); }
  DDMaterial eeQuaMat() const { return ddmat(m_EEQuaMat); }

  DDMaterial eeCrysMat() const { return ddmat(m_EECrysMat); }
  DDMaterial eeWallMat() const { return ddmat(m_EEWallMat); }

  double eeCrysLength() const { return m_EECrysLength; }
  double eeCrysRear() const { return m_EECrysRear; }
  double eeCrysFront() const { return m_EECrysFront; }
  double eeSCELength() const { return m_EESCELength; }
  double eeSCERear() const { return m_EESCERear; }
  double eeSCEFront() const { return m_EESCEFront; }
  double eeSCALength() const { return m_EESCALength; }
  double eeSCARear() const { return m_EESCARear; }
  double eeSCAFront() const { return m_EESCAFront; }
  double eeSCAWall() const { return m_EESCAWall; }
  double eeSCHLength() const { return m_EESCHLength; }
  double eeSCHSide() const { return m_EESCHSide; }

  double eenSCTypes() const { return m_EEnSCTypes; }
  double eenColumns() const { return m_EEnColumns; }
  double eenSCCutaway() const { return m_EEnSCCutaway; }
  double eenSCquad() const { return m_EEnSCquad; }
  double eenCRSC() const { return m_EEnCRSC; }
  const std::vector<double>& eevecEESCProf() const { return m_vecEESCProf; }
  const std::vector<double>& eevecEEShape() const { return m_vecEEShape; }
  const std::vector<double>& eevecEESCCutaway() const { return m_vecEESCCutaway; }
  const std::vector<double>& eevecEESCCtrs() const { return m_vecEESCCtrs; }
  const std::vector<double>& eevecEECRCtrs() const { return m_vecEECRCtrs; }

  DDName cutBoxName() const { return ddname(m_cutBoxName); }
  double eePFHalf() const { return m_PFhalf; }
  double eePFFifth() const { return m_PFfifth; }
  double eePF45() const { return m_PF45; }

  DDName envName(unsigned int i) const { return ddname(m_envName + std::to_string(i)); }
  DDName alvName(unsigned int i) const { return ddname(m_alvName + std::to_string(i)); }
  DDName intName(unsigned int i) const { return ddname(m_intName + std::to_string(i)); }
  DDName cryName() const { return ddname(m_cryName); }

  DDName addTmp(DDName aName) const { return ddname(aName.name() + "Tmp"); }

  const DDTranslation& cryFCtr(unsigned int iRow, unsigned int iCol) const { return m_cryFCtr[iRow - 1][iCol - 1]; }

  const DDTranslation& cryRCtr(unsigned int iRow, unsigned int iCol) const { return m_cryRCtr[iRow - 1][iCol - 1]; }

  const DDTranslation& scrFCtr(unsigned int iRow, unsigned int iCol) const { return m_scrFCtr[iRow - 1][iCol - 1]; }

  const DDTranslation& scrRCtr(unsigned int iRow, unsigned int iCol) const { return m_scrRCtr[iRow - 1][iCol - 1]; }

  const std::vector<double>& vecEESCLims() const { return m_vecEESCLims; }

  double iLength() const { return m_iLength; }
  double iXYOff() const { return m_iXYOff; }

protected:
private:
  std::string m_idNameSpace;  //Namespace of this and ALL sub-parts

  // Barrel volume
  std::string m_EEMat;
  double m_EEzOff;

  std::string m_EEQuaName;
  std::string m_EEQuaMat;

  std::string m_EECrysMat;
  std::string m_EEWallMat;

  double m_EECrysLength;
  double m_EECrysRear;
  double m_EECrysFront;
  double m_EESCELength;
  double m_EESCERear;
  double m_EESCEFront;
  double m_EESCALength;
  double m_EESCARear;
  double m_EESCAFront;
  double m_EESCAWall;
  double m_EESCHLength;
  double m_EESCHSide;

  double m_EEnSCTypes;
  std::vector<double> m_vecEESCProf;
  double m_EEnColumns;
  std::vector<double> m_vecEEShape;
  double m_EEnSCCutaway;
  std::vector<double> m_vecEESCCutaway;
  double m_EEnSCquad;
  std::vector<double> m_vecEESCCtrs;
  double m_EEnCRSC;
  std::vector<double> m_vecEECRCtrs;

  const std::vector<double>* m_cutParms;
  std::string m_cutBoxName;

  std::string m_envName;
  std::string m_alvName;
  std::string m_intName;
  std::string m_cryName;

  DDTranslation m_cryFCtr[5][5];
  DDTranslation m_cryRCtr[5][5];

  DDTranslation m_scrFCtr[10][10];
  DDTranslation m_scrRCtr[10][10];

  double m_PFhalf;
  double m_PFfifth;
  double m_PF45;

  std::vector<double> m_vecEESCLims;

  double m_iLength;

  double m_iXYOff;

  double m_cryZOff;

  double m_zFront;
};

namespace std {}
using namespace std;

DDEcalEndcapAlgo::DDEcalEndcapAlgo()
    : m_idNameSpace(""),
      m_EEMat(""),
      m_EEzOff(0),
      m_EEQuaName(""),
      m_EEQuaMat(""),
      m_EECrysMat(""),
      m_EEWallMat(""),
      m_EECrysLength(0),
      m_EECrysRear(0),
      m_EECrysFront(0),
      m_EESCELength(0),
      m_EESCERear(0),
      m_EESCEFront(0),
      m_EESCALength(0),
      m_EESCARear(0),
      m_EESCAFront(0),
      m_EESCAWall(0),
      m_EESCHLength(0),
      m_EESCHSide(0),
      m_EEnSCTypes(0),
      m_vecEESCProf(),
      m_EEnColumns(0),
      m_vecEEShape(),
      m_EEnSCCutaway(0),
      m_vecEESCCutaway(),
      m_EEnSCquad(0),
      m_vecEESCCtrs(),
      m_EEnCRSC(0),
      m_vecEECRCtrs(),
      m_cutParms(nullptr),
      m_cutBoxName(""),
      m_envName(""),
      m_alvName(""),
      m_intName(""),
      m_cryName(""),
      m_PFhalf(0),
      m_PFfifth(0),
      m_PF45(0),
      m_vecEESCLims(),
      m_iLength(0),
      m_iXYOff(0),
      m_cryZOff(0),
      m_zFront(0) {
  edm::LogVerbatim("EcalGeomX") << "DDEcalEndcapAlgo info: Creating an instance";
}

DDEcalEndcapAlgo::~DDEcalEndcapAlgo() {}

void DDEcalEndcapAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments& /*mArgs*/,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments& /*vsArgs*/) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeomX") << "DDEcalEndcapAlgo info: Initialize";
#endif
  m_idNameSpace = DDCurrentNamespace::ns();
  // TRICK!
  m_idNameSpace = parent().name().ns();
  // barrel parent volume
  m_EEMat = sArgs["EEMat"];
  m_EEzOff = nArgs["EEzOff"];

  m_EEQuaName = sArgs["EEQuaName"];
  m_EEQuaMat = sArgs["EEQuaMat"];
  m_EECrysMat = sArgs["EECrysMat"];
  m_EEWallMat = sArgs["EEWallMat"];
  m_EECrysLength = nArgs["EECrysLength"];
  m_EECrysRear = nArgs["EECrysRear"];
  m_EECrysFront = nArgs["EECrysFront"];
  m_EESCELength = nArgs["EESCELength"];
  m_EESCERear = nArgs["EESCERear"];
  m_EESCEFront = nArgs["EESCEFront"];
  m_EESCALength = nArgs["EESCALength"];
  m_EESCARear = nArgs["EESCARear"];
  m_EESCAFront = nArgs["EESCAFront"];
  m_EESCAWall = nArgs["EESCAWall"];
  m_EESCHLength = nArgs["EESCHLength"];
  m_EESCHSide = nArgs["EESCHSide"];
  m_EEnSCTypes = nArgs["EEnSCTypes"];
  m_EEnColumns = nArgs["EEnColumns"];
  m_EEnSCCutaway = nArgs["EEnSCCutaway"];
  m_EEnSCquad = nArgs["EEnSCquad"];
  m_EEnCRSC = nArgs["EEnCRSC"];
  m_vecEESCProf = vArgs["EESCProf"];
  m_vecEEShape = vArgs["EEShape"];
  m_vecEESCCutaway = vArgs["EESCCutaway"];
  m_vecEESCCtrs = vArgs["EESCCtrs"];
  m_vecEECRCtrs = vArgs["EECRCtrs"];

  m_cutBoxName = sArgs["EECutBoxName"];

  m_envName = sArgs["EEEnvName"];
  m_alvName = sArgs["EEAlvName"];
  m_intName = sArgs["EEIntName"];
  m_cryName = sArgs["EECryName"];

  m_PFhalf = nArgs["EEPFHalf"];
  m_PFfifth = nArgs["EEPFFifth"];
  m_PF45 = nArgs["EEPF45"];

  m_vecEESCLims = vArgs["EESCLims"];

  m_iLength = nArgs["EEiLength"];

  m_iXYOff = nArgs["EEiXYOff"];

  m_cryZOff = nArgs["EECryZOff"];

  m_zFront = nArgs["EEzFront"];
}

////////////////////////////////////////////////////////////////////
// DDEcalEndcapAlgo methods...
////////////////////////////////////////////////////////////////////

DDRotation DDEcalEndcapAlgo::myrot(const std::string& s, const DDRotationMatrix& r) const {
  return DDrot(ddname(m_idNameSpace + ":" + s), std::make_unique<DDRotationMatrix>(r));
}

DDMaterial DDEcalEndcapAlgo::ddmat(const std::string& s) const { return DDMaterial(ddname(s)); }

DDName DDEcalEndcapAlgo::ddname(const std::string& s) const {
  const pair<std::string, std::string> temp(DDSplit(s));
  if (temp.second.empty()) {
    return DDName(temp.first, m_idNameSpace);
  } else {
    return DDName(temp.first, temp.second);
  }
}

//-------------------- Endcap SC geometry methods ---------------------

void DDEcalEndcapAlgo::execute(DDCompactView& cpv) {
  //  Position supercrystals in EE Quadrant
  //  Version:    1.00
  //  Created:    30 July 2007
  //  Last Mod:
  //---------------------------------------------------------------------

  //********************************* cutbox for trimming edge SCs
  const double cutWid(eeSCERear() / sqrt(2.));
  const DDSolid eeCutBox(DDSolidFactory::box(cutBoxName(), cutWid, cutWid, eeSCELength() / sqrt(2.)));
  m_cutParms = &eeCutBox.parameters();
  //**************************************************************

  const double zFix(m_zFront - 3172.0 * mm);  // fix for changing z offset

  //** fill supercrystal front and rear center positions from xml input
  for (unsigned int iC(0); iC != (unsigned int)eenSCquad(); ++iC) {
    const unsigned int iOff(8 * iC);
    const unsigned int ix((unsigned int)eevecEESCCtrs()[iOff + 0]);
    const unsigned int iy((unsigned int)eevecEESCCtrs()[iOff + 1]);

    assert(ix > 0 && ix < 11 && iy > 0 && iy < 11);

    m_scrFCtr[ix - 1][iy - 1] =
        DDTranslation(eevecEESCCtrs()[iOff + 2], eevecEESCCtrs()[iOff + 4], eevecEESCCtrs()[iOff + 6] + zFix);

    m_scrRCtr[ix - 1][iy - 1] =
        DDTranslation(eevecEESCCtrs()[iOff + 3], eevecEESCCtrs()[iOff + 5], eevecEESCCtrs()[iOff + 7] + zFix);
  }

  //** fill crystal front and rear center positions from xml input
  for (unsigned int iC(0); iC != 25; ++iC) {
    const unsigned int iOff(8 * iC);
    const unsigned int ix((unsigned int)eevecEECRCtrs()[iOff + 0]);
    const unsigned int iy((unsigned int)eevecEECRCtrs()[iOff + 1]);

    assert(ix > 0 && ix < 6 && iy > 0 && iy < 6);

    m_cryFCtr[ix - 1][iy - 1] =
        DDTranslation(eevecEECRCtrs()[iOff + 2], eevecEECRCtrs()[iOff + 4], eevecEECRCtrs()[iOff + 6]);

    m_cryRCtr[ix - 1][iy - 1] =
        DDTranslation(eevecEECRCtrs()[iOff + 3], eevecEECRCtrs()[iOff + 5], eevecEECRCtrs()[iOff + 7]);
  }

  EECreateCR();  // make a single crystal just once here

  for (unsigned int isc(0); isc < eenSCTypes(); ++isc) {
    EECreateSC(isc + 1, cpv);
  }

  const std::vector<double>& colLimits(eevecEEShape());
  //** Loop over endcap columns
  for (int icol = 1; icol <= int(eenColumns()); icol++) {
    //**  Loop over SCs in column, using limits from xml input
    for (int irow = int(colLimits[2 * icol - 2]); irow <= int(colLimits[2 * icol - 1]); ++irow) {
      if (vecEESCLims()[0] <= icol && vecEESCLims()[1] >= icol && vecEESCLims()[2] <= irow &&
          vecEESCLims()[3] >= irow) {
        // Find SC type (complete or partial) for this location
        const unsigned int isctype(EEGetSCType(icol, irow));

        // Create SC as a DDEcalEndcapTrap object and calculate rotation and
        // translation required to position it in the endcap.
        DDEcalEndcapTrap scrys(1, eeSCEFront(), eeSCERear(), eeSCELength());

        scrys.moveto(scrFCtr(icol, irow), scrRCtr(icol, irow));
        scrys.translate(DDTranslation(0., 0., -eezOff()));

        DDName rname(envName(isctype).name() + std::to_string(icol) + "R" + std::to_string(irow));

#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EcalGeoXm") << "Quadrant, SC col/row " << eeQuaName() << " " << icol << " / " << irow
                                      << std::endl
                                      << "   Limits " << int(colLimits[2 * icol - 2]) << "->"
                                      << int(colLimits[2 * icol - 1]) << std::endl
                                      << "   SC type = " << isctype << std::endl
                                      << "   Zoff = " << eezOff() << std::endl
                                      << "   Rotation " << rname << " " << scrys.rotation() << std::endl
                                      << "   Position " << scrys.centrePos();
#endif
        // Position SC in endcap
        cpv.position(envName(isctype),
                     eeQuaName(),
                     100 * isctype + 10 * (icol - 1) + (irow - 1),
                     scrys.centrePos(),
                     myrot(rname.fullname(), scrys.rotation()));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EEGeom") << envName(isctype) << " " << (100 * isctype + 10 * (icol - 1) + (irow - 1))
                                   << " in " << eeQuaName();
        edm::LogVerbatim("EcalGeom") << envName(isctype) << " " << (100 * isctype + 10 * (icol - 1) + (irow - 1))
                                     << " in " << eeQuaName() << " at " << scrys.centrePos();
#endif
      }
    }
  }
}

void DDEcalEndcapAlgo::EECreateSC(const unsigned int iSCType,
                                  DDCompactView& cpv) {  //  EECreateSCType   Create SC logical volume of the given type

  DDRotation noRot;
  DDLogicalPart eeSCELog;
  DDLogicalPart eeSCALog;
  DDLogicalPart eeSCILog;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeomX") << "EECreateSC: Creating SC envelope";
#endif
  const string anum(std::to_string(iSCType));

  const double eFront(0.5 * eeSCEFront());
  const double eRear(0.5 * eeSCERear());
  const double eAng(atan((eeSCERear() - eeSCEFront()) / (sqrt(2.) * eeSCELength())));
  const double ffived(45 * deg);
  const double zerod(0 * deg);
  DDSolid eeSCEnv(DDSolidFactory::trap((1 == iSCType ? envName(iSCType) : addTmp(envName(iSCType))),
                                       0.5 * eeSCELength(),
                                       eAng,
                                       ffived,
                                       eFront,
                                       eFront,
                                       eFront,
                                       zerod,
                                       eRear,
                                       eRear,
                                       eRear,
                                       zerod));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << eeSCEnv.name() << " Trap with parameters: " << 0.5 * eeSCELength() << ":" << eAng
                               << ffived << ":" << eFront << ":" << eFront << ":" << eFront << ":" << zerod << ":"
                               << eRear << ":" << eRear << ":" << eRear << ":" << zerod;
#endif

  const double aFront(0.5 * eeSCAFront());
  const double aRear(0.5 * eeSCARear());
  const double aAng(atan((eeSCARear() - eeSCAFront()) / (sqrt(2.) * eeSCALength())));
  const DDSolid eeSCAlv(DDSolidFactory::trap((1 == iSCType ? alvName(iSCType) : addTmp(alvName(iSCType))),
                                             0.5 * eeSCALength(),
                                             aAng,
                                             ffived,
                                             aFront,
                                             aFront,
                                             aFront,
                                             zerod,
                                             aRear,
                                             aRear,
                                             aRear,
                                             zerod));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << eeSCAlv.name() << " Trap with parameters: " << 0.5 * eeSCALength() << ":" << aAng
                               << ":" << ffived << ":" << aFront << ":" << aFront << ":" << aFront << ":" << zerod
                               << ":" << aRear << ":" << aRear << ":" << aRear << ":" << zerod;
#endif
  const double dwall(eeSCAWall());
  const double iFront(aFront - dwall);
  const double iRear(iFront);    //aRear  - dwall ) ;
  const double iLen(iLength());  //0.075*eeSCALength() ) ;
  const DDSolid eeSCInt(DDSolidFactory::trap((1 == iSCType ? intName(iSCType) : addTmp(intName(iSCType))),
                                             iLen / 2.,
                                             atan((eeSCARear() - eeSCAFront()) / (sqrt(2.) * eeSCALength())),
                                             ffived,
                                             iFront,
                                             iFront,
                                             iFront,
                                             zerod,
                                             iRear,
                                             iRear,
                                             iRear,
                                             zerod));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << eeSCInt.name() << " Trap with parameters: " << iLen / 2. << ":"
                               << (atan((eeSCARear() - eeSCAFront()) / (sqrt(2.) * eeSCALength()))) << ":" << ffived
                               << ":" << iFront << ":" << iFront << ":" << iFront << ":" << zerod << ":" << iRear << ":"
                               << iRear << ":" << iRear << ":" << zerod;
#endif
  const double dz(-0.5 * (eeSCELength() - eeSCALength()));
  const double dxy(0.5 * dz * (eeSCERear() - eeSCEFront()) / eeSCELength());
  const double zIOff(-(eeSCALength() - iLen) / 2.);
  const double xyIOff(iXYOff());

  if (1 == iSCType)  // standard SC in this block
  {
    eeSCELog = DDLogicalPart(envName(iSCType), eeMat(), eeSCEnv);
    eeSCALog = DDLogicalPart(alvName(iSCType), eeWallMat(), eeSCAlv);
    eeSCILog = DDLogicalPart(intName(iSCType), eeMat(), eeSCInt);
  } else  // partial SCs this block: create subtraction volumes as appropriate
  {
    const double half((*m_cutParms)[0] - eePFHalf() * eeCrysRear());
    const double fifth((*m_cutParms)[0] + eePFFifth() * eeCrysRear());
    const double fac(eePF45());

    const double zmm(0 * mm);

    DDTranslation cutTra(
        2 == iSCType ? DDTranslation(zmm, half, zmm)
                     : (3 == iSCType ? DDTranslation(half, zmm, zmm)
                                     : (4 == iSCType ? DDTranslation(zmm, -fifth, zmm)
                                                     : (5 == iSCType ? DDTranslation(-half * fac, -half * fac, zmm)
                                                                     : DDTranslation(-fifth, zmm, zmm)))));

    const CLHEP::HepRotationZ cutm(ffived);

    DDRotation cutRot(5 != iSCType ? noRot
                                   : myrot("EECry5Rot",
                                           DDRotationMatrix(cutm.xx(),
                                                            cutm.xy(),
                                                            cutm.xz(),
                                                            cutm.yx(),
                                                            cutm.yy(),
                                                            cutm.yz(),
                                                            cutm.zx(),
                                                            cutm.zy(),
                                                            cutm.zz())));

    DDSolid eeCutEnv(
        DDSolidFactory::subtraction(envName(iSCType), addTmp(envName(iSCType)), cutBoxName(), cutTra, cutRot));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << eeCutEnv.name() << " Subtracted by " << (eeSCERear() / sqrt(2.)) << ":"
                                 << (eeSCERear() / sqrt(2.)) << ":" << (eeSCELength() / sqrt(2.));
#endif
    const DDTranslation extra(dxy, dxy, dz);

    DDSolid eeCutAlv(
        DDSolidFactory::subtraction(alvName(iSCType), addTmp(alvName(iSCType)), cutBoxName(), cutTra - extra, cutRot));

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << eeCutAlv.name() << " Subtracted by " << (eeSCERear() / sqrt(2.)) << ":"
                                 << (eeSCERear() / sqrt(2.)) << ":" << (eeSCELength() / sqrt(2.));
#endif
    const double mySign(iSCType < 4 ? +1. : -1.);

    const DDTranslation extraI(xyIOff + mySign * 2 * mm, xyIOff + mySign * 2 * mm, zIOff);

    DDSolid eeCutInt(
        DDSolidFactory::subtraction(intName(iSCType), addTmp(intName(iSCType)), cutBoxName(), cutTra - extraI, cutRot));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << eeCutInt.name() << " Subtracted by " << (eeSCERear() / sqrt(2.)) << ":"
                                 << (eeSCERear() / sqrt(2.)) << ":" << (eeSCELength() / sqrt(2.));
#endif

    eeSCELog = DDLogicalPart(envName(iSCType), eeMat(), eeCutEnv);
    eeSCALog = DDLogicalPart(alvName(iSCType), eeWallMat(), eeCutAlv);
    eeSCILog = DDLogicalPart(intName(iSCType), eeMat(), eeCutInt);
  }

  cpv.position(eeSCALog, envName(iSCType), iSCType * 100 + 1, DDTranslation(dxy, dxy, dz), noRot);
  cpv.position(eeSCILog, alvName(iSCType), iSCType * 100 + 1, DDTranslation(xyIOff, xyIOff, zIOff), noRot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EEGeom") << eeSCALog.name() << " " << (iSCType * 100 + 1) << " in " << envName(iSCType);
  edm::LogVerbatim("EEGeom") << eeSCILog.name() << " " << (iSCType * 100 + 1) << " in " << alvName(iSCType);
  edm::LogVerbatim("EcalGeom") << eeSCALog.name() << " " << (iSCType * 100 + 1) << " in " << envName(iSCType) << " at ("
                               << dxy << ", " << dxy << ", " << dz << ")";
  edm::LogVerbatim("EcalGeom") << eeSCILog.name() << " " << (iSCType * 100 + 1) << " in " << alvName(iSCType) << " at ("
                               << xyIOff << ", " << xyIOff << ", " << zIOff << ")";
#endif
  DDTranslation croffset(0., 0., 0.);
  EEPositionCRs(alvName(iSCType), croffset, iSCType, cpv);
}

unsigned int DDEcalEndcapAlgo::EEGetSCType(const unsigned int iCol, const unsigned int iRow) {
  unsigned int iType = 1;
  for (unsigned int ii = 0; ii < (unsigned int)(eenSCCutaway()); ++ii) {
    if ((eevecEESCCutaway()[3 * ii] == iCol) && (eevecEESCCutaway()[3 * ii + 1] == iRow)) {
      iType = int(eevecEESCCutaway()[3 * ii + 2]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalGeomX") << "EEGetSCType: col, row, type = " << iCol << " " << iRow << " " << iType;
#endif
    }
  }
  return iType;
}

void DDEcalEndcapAlgo::EECreateCR() {
  //  EECreateCR   Create endcap crystal logical volume

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeomX") << "EECreateCR:  = ";
#endif
  DDSolid EECRSolid(DDSolidFactory::trap(cryName(),
                                         0.5 * eeCrysLength(),
                                         atan((eeCrysRear() - eeCrysFront()) / (sqrt(2.) * eeCrysLength())),
                                         45. * deg,
                                         0.5 * eeCrysFront(),
                                         0.5 * eeCrysFront(),
                                         0.5 * eeCrysFront(),
                                         0. * deg,
                                         0.5 * eeCrysRear(),
                                         0.5 * eeCrysRear(),
                                         0.5 * eeCrysRear(),
                                         0. * deg));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << EECRSolid.name() << " Trap with parameters: " << 0.5 * eeCrysLength() << ":"
                               << (atan((eeCrysRear() - eeCrysFront()) / (sqrt(2.) * eeCrysLength()))) << ":"
                               << 45. * deg << ":" << 0.5 * eeCrysFront() << ":" << 0.5 * eeCrysFront() << ":"
                               << 0.5 * eeCrysFront() << ":" << 0. * deg << ":" << 0.5 * eeCrysRear() << ":"
                               << 0.5 * eeCrysRear() << ":" << 0.5 * eeCrysRear() << ":" << 0. * deg;
#endif

  DDLogicalPart part(cryName(), eeCrysMat(), EECRSolid);
}

void DDEcalEndcapAlgo::EEPositionCRs(const DDName& pName,
                                     const DDTranslation& /*offset*/,
                                     const int iSCType,
                                     DDCompactView& cpv) {
  //  EEPositionCRs Position crystals within parent supercrystal interior volume

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeomX") << "EEPositionCRs called ";
#endif
  static const unsigned int ncol(5);

  if (iSCType > 0 && iSCType <= eenSCTypes()) {
    const unsigned int icoffset((iSCType - 1) * ncol - 1);

    // Loop over columns of SC
    for (unsigned int icol(1); icol <= ncol; ++icol) {
      // Get column limits for this SC type from xml input
      const int ncrcol((int)eevecEESCProf()[icoffset + icol]);

      const int imin(0 < ncrcol ? 1 : (0 > ncrcol ? ncol + ncrcol + 1 : 0));
      const int imax(0 < ncrcol ? ncrcol : (0 > ncrcol ? ncol : 0));

      if (imax > 0) {
        // Loop over crystals in this row
        for (int irow(imin); irow <= imax; ++irow) {
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EcalGeomX") << " type, col, row " << iSCType << " " << icol << " " << irow;
#endif
          // Create crystal as a DDEcalEndcapTrap object and calculate rotation and
          // translation required to position it in the SC.
          DDEcalEndcapTrap crystal(1, eeCrysFront(), eeCrysRear(), eeCrysLength());

          crystal.moveto(cryFCtr(icol, irow), cryRCtr(icol, irow));

          DDName rname("EECrRoC" + std::to_string(icol) + "R" + std::to_string(irow));

          cpv.position(cryName(),
                       pName,
                       100 * iSCType + 10 * (icol - 1) + (irow - 1),
                       crystal.centrePos() - DDTranslation(0, 0, m_cryZOff),
                       myrot(rname.fullname(), crystal.rotation()));
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("EEGeom") << cryName() << " " << (100 * iSCType + 10 * (icol - 1) + (irow - 1)) << " in "
                                     << pName;
          edm::LogVerbatim("EcalGeom") << cryName() << " " << (100 * iSCType + 10 * (icol - 1) + (irow - 1)) << " in "
                                       << pName << " at " << (crystal.centrePos() - DDTranslation(0, 0, m_cryZOff));
#endif
        }
      }
    }
  }
}

#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDEcalEndcapAlgo, "ecal:DDEcalEndcapAlgo");
