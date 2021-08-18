#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "DataFormats/Math/interface/angle_units.h"
// Header files for endcap supercrystal geometry
#include "Geometry/EcalCommonData/interface/DDEcalEndcapTrapX.h"
#include <CLHEP/Geometry/Transform3D.h>

#include <string>
#include <vector>

using namespace angle_units::operators;

//#define EDM_ML_DEBUG

namespace {
  struct Endcap {
    std::string mat;
    double zOff;

    std::string quaName;
    std::string quaMat;

    std::string crysMat;
    std::string wallMat;

    double crysLength;
    double crysRear;
    double crysFront;
    double sCELength;
    double sCERear;
    double sCEFront;
    double sCALength;
    double sCARear;
    double sCAFront;
    double sCAWall;
    double sCHLength;
    double sCHSide;

    double nSCTypes;
    std::vector<double> vecEESCProf;
    double nColumns;
    std::vector<double> vecEEShape;
    double nSCCutaway;
    std::vector<double> vecEESCCutaway;
    double nSCquad;
    std::vector<double> vecEESCCtrs;
    double nCRSC;
    std::vector<double> vecEECRCtrs;

    std::array<double, 3> cutParms;
    std::string cutBoxName;

    std::string envName;
    std::string alvName;
    std::string intName;
    std::string cryName;

    DDTranslation cryFCtr[5][5];
    DDTranslation cryRCtr[5][5];
    DDTranslation scrFCtr[10][10];
    DDTranslation scrRCtr[10][10];

    double pFHalf;
    double pFFifth;
    double pF45;

    std::vector<double> vecEESCLims;

    double iLength;
    double iXYOff;
    double cryZOff;
    double zFront;
  };

  const DDRotationMatrix& myrot(cms::DDNamespace& ns, const std::string& nam, const DDRotationMatrix& r) {
    ns.addRotation(nam, r);
    return ns.rotation(ns.prepend(nam));
  }

  std::string_view mynamespace(std::string_view input) {
    std::string_view v = input;
    auto trim_pos = v.find(':');
    if (trim_pos != v.npos)
      v.remove_suffix(v.size() - (trim_pos + 1));
    return v;
  }
}  // namespace

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  BenchmarkGrd counter("DDEcalEndcapAlgo");
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  // TRICK!
  std::string myns{mynamespace(args.parentName()).data(), mynamespace(args.parentName()).size()};

  Endcap ee;
  ee.mat = args.str("EEMat");
  ee.zOff = args.dble("EEzOff");

  ee.quaName = args.str("EEQuaName");
  ee.quaMat = args.str("EEQuaMat");
  ee.crysMat = args.str("EECrysMat");
  ee.wallMat = args.str("EEWallMat");
  ee.crysLength = args.dble("EECrysLength");
  ee.crysRear = args.dble("EECrysRear");
  ee.crysFront = args.dble("EECrysFront");
  ee.sCELength = args.dble("EESCELength");
  ee.sCERear = args.dble("EESCERear");
  ee.sCEFront = args.dble("EESCEFront");
  ee.sCALength = args.dble("EESCALength");
  ee.sCARear = args.dble("EESCARear");
  ee.sCAFront = args.dble("EESCAFront");
  ee.sCAWall = args.dble("EESCAWall");
  ee.sCHLength = args.dble("EESCHLength");
  ee.sCHSide = args.dble("EESCHSide");
  ee.nSCTypes = args.dble("EEnSCTypes");
  ee.nColumns = args.dble("EEnColumns");
  ee.nSCCutaway = args.dble("EEnSCCutaway");
  ee.nSCquad = args.dble("EEnSCquad");
  ee.nCRSC = args.dble("EEnCRSC");
  ee.vecEESCProf = args.vecDble("EESCProf");
  ee.vecEEShape = args.vecDble("EEShape");
  ee.vecEESCCutaway = args.vecDble("EESCCutaway");
  ee.vecEESCCtrs = args.vecDble("EESCCtrs");
  ee.vecEECRCtrs = args.vecDble("EECRCtrs");

  ee.cutBoxName = args.str("EECutBoxName");

  ee.envName = args.str("EEEnvName");
  ee.alvName = args.str("EEAlvName");
  ee.intName = args.str("EEIntName");
  ee.cryName = args.str("EECryName");

  ee.pFHalf = args.dble("EEPFHalf");
  ee.pFFifth = args.dble("EEPFFifth");
  ee.pF45 = args.dble("EEPF45");

  ee.vecEESCLims = args.vecDble("EESCLims");
  ee.iLength = args.dble("EEiLength");
  ee.iXYOff = args.dble("EEiXYOff");
  ee.cryZOff = args.dble("EECryZOff");
  ee.zFront = args.dble("EEzFront");

  //  Position supercrystals in EE Quadrant

  //********************************* cutbox for trimming edge SCs
  const double cutWid(ee.sCERear / sqrt(2.));
  ee.cutParms[0] = cutWid;
  ee.cutParms[1] = cutWid;
  ee.cutParms[2] = ee.sCELength / sqrt(2.);
  dd4hep::Solid eeCutBox = dd4hep::Box(ee.cutBoxName, ee.cutParms[0], ee.cutParms[1], ee.cutParms[2]);
  //**************************************************************

  const double zFix(ee.zFront - 3172 * dd4hep::mm);  // fix for changing z offset

  //** fill supercrystal front and rear center positions from xml input
  for (unsigned int iC(0); iC != (unsigned int)ee.nSCquad; ++iC) {
    const unsigned int iOff(8 * iC);
    const unsigned int ix((unsigned int)ee.vecEESCCtrs[iOff + 0]);
    const unsigned int iy((unsigned int)ee.vecEESCCtrs[iOff + 1]);

    assert(ix > 0 && ix < 11 && iy > 0 && iy < 11);

    ee.scrFCtr[ix - 1][iy - 1] =
        DDTranslation(ee.vecEESCCtrs[iOff + 2], ee.vecEESCCtrs[iOff + 4], ee.vecEESCCtrs[iOff + 6] + zFix);

    ee.scrRCtr[ix - 1][iy - 1] =
        DDTranslation(ee.vecEESCCtrs[iOff + 3], ee.vecEESCCtrs[iOff + 5], ee.vecEESCCtrs[iOff + 7] + zFix);
  }

  //** fill crystal front and rear center positions from xml input
  for (unsigned int iC(0); iC != 25; ++iC) {
    const unsigned int iOff(8 * iC);
    const unsigned int ix((unsigned int)ee.vecEECRCtrs[iOff + 0]);
    const unsigned int iy((unsigned int)ee.vecEECRCtrs[iOff + 1]);

    assert(ix > 0 && ix < 6 && iy > 0 && iy < 6);

    ee.cryFCtr[ix - 1][iy - 1] =
        DDTranslation(ee.vecEECRCtrs[iOff + 2], ee.vecEECRCtrs[iOff + 4], ee.vecEECRCtrs[iOff + 6]);

    ee.cryRCtr[ix - 1][iy - 1] =
        DDTranslation(ee.vecEECRCtrs[iOff + 3], ee.vecEECRCtrs[iOff + 5], ee.vecEECRCtrs[iOff + 7]);
  }

  dd4hep::Solid eeCRSolid = dd4hep::Trap(ee.cryName,
                                         0.5 * ee.crysLength,
                                         atan((ee.crysRear - ee.crysFront) / (sqrt(2.) * ee.crysLength)),
                                         45._deg,
                                         0.5 * ee.crysFront,
                                         0.5 * ee.crysFront,
                                         0.5 * ee.crysFront,
                                         0._deg,
                                         0.5 * ee.crysRear,
                                         0.5 * ee.crysRear,
                                         0.5 * ee.crysRear,
                                         0._deg);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalGeom") << eeCRSolid.name() << " Trap with parameters: " << cms::convert2mm(0.5 * ee.crysLength)
                               << ":" << (atan((ee.crysRear - ee.crysFront) / (sqrt(2.) * ee.crysLength))) << ":"
                               << 45._deg << ":" << cms::convert2mm(0.5 * ee.crysFront) << ":"
                               << cms::convert2mm(0.5 * ee.crysFront) << ":" << cms::convert2mm(0.5 * ee.crysFront)
                               << ":" << 0._deg << ":" << cms::convert2mm(0.5 * ee.crysRear) << ":"
                               << cms::convert2mm(0.5 * ee.crysRear) << ":" << cms::convert2mm(0.5 * ee.crysRear) << ":"
                               << 0._deg;
#endif
  dd4hep::Volume eeCRLog = dd4hep::Volume(myns + ee.cryName, eeCRSolid, ns.material(ee.crysMat));

  for (unsigned int isc(0); isc < ee.nSCTypes; ++isc) {
    unsigned int iSCType = isc + 1;
    const std::string anum(std::to_string(iSCType));
    const double eFront(0.5 * ee.sCEFront);
    const double eRear(0.5 * ee.sCERear);
    const double eAng(atan((ee.sCERear - ee.sCEFront) / (sqrt(2.) * ee.sCELength)));
    const double ffived(45_deg);
    const double zerod(0_deg);
    std::string eeSCEnvName(1 == iSCType ? ee.envName + std::to_string(iSCType)
                                         : (ee.envName + std::to_string(iSCType) + "Tmp"));
    dd4hep::Solid eeSCEnv = ns.addSolidNS(
        eeSCEnvName,
        dd4hep::Trap(
            eeSCEnvName, 0.5 * ee.sCELength, eAng, ffived, eFront, eFront, eFront, zerod, eRear, eRear, eRear, zerod));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << eeSCEnv.name() << " Trap with parameters: " << cms::convert2mm(0.5 * ee.sCELength)
                                 << ":" << eAng << ":" << ffived << ":" << cms::convert2mm(eFront) << ":"
                                 << cms::convert2mm(eFront) << ":" << cms::convert2mm(eFront) << ":" << zerod << ":"
                                 << cms::convert2mm(eRear) << ":" << cms::convert2mm(eRear) << ":"
                                 << cms::convert2mm(eRear) << ":" << zerod;
#endif

    const double aFront(0.5 * ee.sCAFront);
    const double aRear(0.5 * ee.sCARear);
    const double aAng(atan((ee.sCARear - ee.sCAFront) / (sqrt(2.) * ee.sCALength)));
    std::string eeSCAlvName(
        (1 == iSCType ? ee.alvName + std::to_string(iSCType) : (ee.alvName + std::to_string(iSCType) + "Tmp")));
    dd4hep::Solid eeSCAlv = ns.addSolidNS(
        eeSCAlvName,
        dd4hep::Trap(
            eeSCAlvName, 0.5 * ee.sCALength, aAng, ffived, aFront, aFront, aFront, zerod, aRear, aRear, aRear, zerod));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalGeom") << eeSCAlv.name() << " Trap with parameters: " << cms::convert2mm(0.5 * ee.sCALength)
                                 << ":" << aAng << ":" << ffived << ":" << cms::convert2mm(aFront) << ":"
                                 << cms::convert2mm(aFront) << ":" << cms::convert2mm(aFront) << ":" << zerod << ":"
                                 << cms::convert2mm(aRear) << ":" << cms::convert2mm(aRear) << ":"
                                 << cms::convert2mm(aRear) << ":" << zerod;
#endif

    const double dwall(ee.sCAWall);
    const double iFront(aFront - dwall);
    const double iRear(iFront);
    const double iLen(ee.iLength);
    std::string eeSCIntName(1 == iSCType ? ee.intName + std::to_string(iSCType)
                                         : (ee.intName + std::to_string(iSCType) + "Tmp"));
    dd4hep::Solid eeSCInt = ns.addSolidNS(eeSCIntName,
                                          dd4hep::Trap(eeSCIntName,
                                                       iLen / 2.,
                                                       atan((ee.sCARear - ee.sCAFront) / (sqrt(2.) * ee.sCALength)),
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
    edm::LogVerbatim("EcalGeom") << eeSCAlv.name() << " Trap with parameters: " << cms::convert2mm(iLen / 2.) << ":"
                                 << (atan((ee.sCARear - ee.sCAFront) / (sqrt(2.) * ee.sCALength))) << ":" << ffived
                                 << ":" << cms::convert2mm(iFront) << ":" << cms::convert2mm(iFront) << ":"
                                 << cms::convert2mm(iFront) << ":" << zerod << ":" << cms::convert2mm(iRear) << ":"
                                 << cms::convert2mm(iRear) << ":" << cms::convert2mm(iRear) << ":" << zerod;
#endif

    const double dz(-0.5 * (ee.sCELength - ee.sCALength));
    const double dxy(0.5 * dz * (ee.sCERear - ee.sCEFront) / ee.sCELength);
    const double zIOff(-(ee.sCALength - iLen) / 2.);
    const double xyIOff(ee.iXYOff);

    dd4hep::Volume eeSCELog;
    dd4hep::Volume eeSCALog;
    dd4hep::Volume eeSCILog;

    if (1 == iSCType) {  // standard SC in this block
      eeSCELog =
          ns.addVolumeNS(dd4hep::Volume(myns + ee.envName + std::to_string(iSCType), eeSCEnv, ns.material(ee.mat)));
      eeSCALog = dd4hep::Volume(myns + ee.alvName + std::to_string(iSCType), eeSCAlv, ns.material(ee.wallMat));
      eeSCILog = dd4hep::Volume(myns + ee.intName + std::to_string(iSCType), eeSCInt, ns.material(ee.mat));
    } else {  // partial SCs this block: create subtraction volumes as appropriate
      const double half(ee.cutParms[0] - ee.pFHalf * ee.crysRear);
      const double fifth(ee.cutParms[0] + ee.pFFifth * ee.crysRear);
      const double fac(ee.pF45);

      const double zmm(0 * dd4hep::mm);

      DDTranslation cutTra(
          2 == iSCType ? DDTranslation(zmm, half, zmm)
                       : (3 == iSCType ? DDTranslation(half, zmm, zmm)
                                       : (4 == iSCType ? DDTranslation(zmm, -fifth, zmm)
                                                       : (5 == iSCType ? DDTranslation(-half * fac, -half * fac, zmm)
                                                                       : DDTranslation(-fifth, zmm, zmm)))));

      const CLHEP::HepRotationZ cutm(ffived);

      DDRotationMatrix cutRot(5 != iSCType ? DDRotationMatrix()
                                           : myrot(ns,
                                                   "EECry5Rot",
                                                   DDRotationMatrix(cutm.xx(),
                                                                    cutm.xy(),
                                                                    cutm.xz(),
                                                                    cutm.yx(),
                                                                    cutm.yy(),
                                                                    cutm.yz(),
                                                                    cutm.zx(),
                                                                    cutm.zy(),
                                                                    cutm.zz())));

      dd4hep::Solid eeCutEnv = dd4hep::SubtractionSolid(ee.envName + std::to_string(iSCType),
                                                        ns.solid(ee.envName + std::to_string(iSCType) + "Tmp"),
                                                        eeCutBox,
                                                        dd4hep::Transform3D(cutRot, cutTra));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalGeom") << eeCutEnv.name() << " Subtracted by " << cms::convert2mm(ee.cutParms[0]) << ":"
                                   << cms::convert2mm(ee.cutParms[1]) << ":" << cms::convert2mm(ee.cutParms[2]);
#endif

      const DDTranslation extra(dxy, dxy, dz);

      dd4hep::Solid eeCutAlv = dd4hep::SubtractionSolid(ee.alvName + std::to_string(iSCType),
                                                        ns.solid(ee.alvName + std::to_string(iSCType) + "Tmp"),
                                                        eeCutBox,
                                                        dd4hep::Transform3D(cutRot, cutTra - extra));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalGeom") << eeCutAlv.name() << " Subtracted by " << cms::convert2mm(ee.cutParms[0]) << ":"
                                   << cms::convert2mm(ee.cutParms[1]) << ":" << cms::convert2mm(ee.cutParms[2]);
#endif

      const double mySign(iSCType < 4 ? +1. : -1.);

      const DDTranslation extraI(xyIOff + mySign * 2 * dd4hep::mm, xyIOff + mySign * 2 * dd4hep::mm, zIOff);

      dd4hep::Solid eeCutInt = dd4hep::SubtractionSolid(ee.intName + std::to_string(iSCType),
                                                        ns.solid(ee.intName + std::to_string(iSCType) + "Tmp"),
                                                        eeCutBox,
                                                        dd4hep::Transform3D(cutRot, cutTra - extraI));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalGeom") << eeCutInt.name() << " Subtracted by " << cms::convert2mm(ee.cutParms[0]) << ":"
                                   << cms::convert2mm(ee.cutParms[1]) << ":" << cms::convert2mm(ee.cutParms[2]);
#endif

      eeSCELog =
          ns.addVolumeNS(dd4hep::Volume(myns + ee.envName + std::to_string(iSCType), eeCutEnv, ns.material(ee.mat)));
      eeSCALog = dd4hep::Volume(myns + ee.alvName + std::to_string(iSCType), eeCutAlv, ns.material(ee.wallMat));
      eeSCILog = dd4hep::Volume(myns + ee.intName + std::to_string(iSCType), eeCutInt, ns.material(ee.mat));
    }
    eeSCELog.placeVolume(eeSCALog, iSCType * 100 + 1, dd4hep::Position(dxy, dxy, dz));
    eeSCALog.placeVolume(eeSCILog, iSCType * 100 + 1, dd4hep::Position(xyIOff, xyIOff, zIOff));
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EEGeom") << eeSCALog.name() << " " << (iSCType * 100 + 1) << " in " << eeSCELog.name();
    edm::LogVerbatim("EEGeom") << eeSCILog.name() << " " << (iSCType * 100 + 1) << " in " << eeSCALog.name();
    edm::LogVerbatim("EcalGeom") << eeSCALog.name() << " " << (iSCType * 100 + 1) << " in " << eeSCELog.name()
                                 << " at (" << cms::convert2mm(dxy) << ", " << cms::convert2mm(dxy) << ", "
                                 << cms::convert2mm(dz) << ")";
    edm::LogVerbatim("EcalGeom") << eeSCILog.name() << " " << (iSCType * 100 + 1) << " in " << eeSCALog.name()
                                 << " at (" << cms::convert2mm(xyIOff) << ", " << cms::convert2mm(xyIOff) << ", "
                                 << cms::convert2mm(zIOff) << ")";
#endif
    DDTranslation croffset(0., 0., 0.);

    // Position crystals within parent supercrystal interior volume
    static const unsigned int ncol(5);

    if (iSCType > 0 && iSCType <= ee.nSCTypes) {
      const unsigned int icoffset((iSCType - 1) * ncol - 1);

      // Loop over columns of SC
      for (unsigned int icol(1); icol <= ncol; ++icol) {
        // Get column limits for this SC type from xml input
        const int ncrcol((int)ee.vecEESCProf[icoffset + icol]);

        const int imin(0 < ncrcol ? 1 : (0 > ncrcol ? ncol + ncrcol + 1 : 0));
        const int imax(0 < ncrcol ? ncrcol : (0 > ncrcol ? ncol : 0));

        if (imax > 0) {
          // Loop over crystals in this row
          for (int irow(imin); irow <= imax; ++irow) {
            // Create crystal as a DDEcalEndcapTrapX object and calculate rotation and
            // translation required to position it in the SC.
            DDEcalEndcapTrapX crystal(1, ee.crysFront, ee.crysRear, ee.crysLength);

            crystal.moveto(ee.cryFCtr[icol - 1][irow - 1], ee.cryRCtr[icol - 1][irow - 1]);

            std::string rname("EECrRoC" + std::to_string(icol) + "R" + std::to_string(irow));

            eeSCALog.placeVolume(eeCRLog,
                                 100 * iSCType + 10 * (icol - 1) + (irow - 1),
                                 dd4hep::Transform3D(myrot(ns, rname, crystal.rotation()),
                                                     dd4hep::Position(crystal.centrePos().x(),
                                                                      crystal.centrePos().y(),
                                                                      crystal.centrePos().z() - ee.cryZOff)));
#ifdef EDM_ML_DEBUG
            edm::LogVerbatim("EEGeom") << eeCRLog.name() << " " << (100 * iSCType + 10 * (icol - 1) + (irow - 1))
                                       << " in " << eeSCALog.name();
            edm::LogVerbatim("EcalGeom") << eeCRLog.name() << " " << (100 * iSCType + 10 * (icol - 1) + (irow - 1))
                                         << " in " << eeSCALog.name() << " at ("
                                         << cms::convert2mm(crystal.centrePos().x()) << ", "
                                         << cms::convert2mm(crystal.centrePos().y()) << ", "
                                         << cms::convert2mm((crystal.centrePos().z() - ee.cryZOff)) << ")";
#endif
          }
        }
      }
    }
  }

  //** Loop over endcap columns
  for (int icol = 1; icol <= int(ee.nColumns); icol++) {
    //**  Loop over SCs in column, using limits from xml input
    for (int irow = int(ee.vecEEShape[2 * icol - 2]); irow <= int(ee.vecEEShape[2 * icol - 1]); ++irow) {
      if (ee.vecEESCLims[0] <= icol && ee.vecEESCLims[1] >= icol && ee.vecEESCLims[2] <= irow &&
          ee.vecEESCLims[3] >= irow) {
        // Find SC type (complete or partial) for this location
        unsigned int isctype = 1;

        for (unsigned int ii = 0; ii < (unsigned int)(ee.nSCCutaway); ++ii) {
          if ((ee.vecEESCCutaway[3 * ii] == icol) && (ee.vecEESCCutaway[3 * ii + 1] == irow)) {
            isctype = int(ee.vecEESCCutaway[3 * ii + 2]);
          }
        }

        // Create SC as a DDEcalEndcapTrapX object and calculate rotation and
        // translation required to position it in the endcap.
        DDEcalEndcapTrapX scrys(1, ee.sCEFront, ee.sCERear, ee.sCELength);
        scrys.moveto(ee.scrFCtr[icol - 1][irow - 1], ee.scrRCtr[icol - 1][irow - 1]);
        scrys.translate(DDTranslation(0., 0., -ee.zOff));

        std::string rname(ee.envName + std::to_string(isctype) + std::to_string(icol) + "R" + std::to_string(irow));
        // Position SC in endcap
        dd4hep::Volume quaLog = ns.volume(ee.quaName);
        dd4hep::Volume childEnvLog = ns.volume(myns + ee.envName + std::to_string(isctype));
        quaLog.placeVolume(childEnvLog,
                           100 * isctype + 10 * (icol - 1) + (irow - 1),
                           dd4hep::Transform3D(scrys.rotation(), scrys.centrePos()));
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("EEGeom") << childEnvLog.name() << " " << (100 * isctype + 10 * (icol - 1) + (irow - 1))
                                   << " in " << quaLog.name();
        edm::LogVerbatim("EcalGeom") << childEnvLog.name() << " " << (100 * isctype + 10 * (icol - 1) + (irow - 1))
                                     << " in " << quaLog.name() << " at (" << cms::convert2mm(scrys.centrePos().x())
                                     << ", " << cms::convert2mm(scrys.centrePos().y()) << ", "
                                     << cms::convert2mm(scrys.centrePos().z()) << ")";
#endif
      }
    }
  }

  return 1;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_ecal_DDEcalEndcapAlgo, algorithm)
