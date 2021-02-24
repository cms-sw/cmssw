#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "DetectorDescription/DDCMS/interface/DDutils.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DD4hep/Shapes.h"

#include <string>
#include <vector>

using namespace angle_units::operators;

//#define EDM_ML_DEBUG

namespace {

  struct EcalPreshower {
    std::vector<std::string> materials;  // materials of the presh-layers
    std::vector<std::string> layName;    // names of the presh-layers
    std::vector<std::string> ladPfx;     // name prefix for ladders
    std::vector<std::string> typesL5;
    std::vector<std::string> typesL4;
    std::vector<std::string> typeOfLaddRow0;
    std::vector<std::string> typeOfLaddRow1;
    std::vector<std::string> typeOfLaddRow2;
    std::vector<std::string> typeOfLaddRow3;

    std::vector<double> thickLayers;
    std::vector<double> abs1stx;
    std::vector<double> abs1sty;
    std::vector<double> abs2ndx;
    std::vector<double> abs2ndy;
    std::vector<double> asymLadd;
    std::vector<double> rminVec;
    std::vector<double> rmaxVec;
    std::vector<double> noLaddInCol;
    std::vector<double> startOfFirstLadd;
    std::vector<double> laddL5map;
    std::vector<double> laddL4map;
    std::string laddMaterial;  // ladd material - air
    double thickness;     // overall thickness of the preshower envelope

    double zlead1;
    double zlead2;
    double waf_intra_col_sep;
    double waf_inter_col_sep;
    double waf_active;
    double wedge_length;
    double wedge_offset;
    double zwedge_ceramic_diff;
    double ywedge_ceramic_diff;
    double wedge_angle;
    double box_thick;
    double dee_separation;
    double in_rad_Abs_Al;
    double in_rad_Abs_Pb;
    double ladder_thick;
    double ladder_width;
    double micromodule_length;
    double absAlX_X;
    double absAlX_Y;
    double absAlX_subtr1_Xshift;
    double absAlX_subtr1_Yshift;
    double rMax_Abs_Al;
    double absAlY_X;
    double absAlY_Y;
    double absAlY_subtr1_Xshift;
    double absAlY_subtr1_Yshift;
    double ldrBck_Length;
    double ldrFrnt_Length;
    double ldrFrnt_Offset;
    double ldrBck_Offset;
    double ceramic_length;
    double wedge_back_thick;
  };
}  // namespace

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  BenchmarkGrd counter("DDEcalPreshowerAlgo");
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  dd4hep::Volume parentVolume = ns.volume(args.parentName());
  dd4hep::Volume swedLog = ns.volume("esalgo:SWED");
  dd4hep::Volume sfLog = ns.volume("esalgo:SF");
  dd4hep::Volume sfbxLog = ns.volume("esalgo:SFBX");
  dd4hep::Volume sfbyLog = ns.volume("esalgo:SFBY");

  EcalPreshower es;
  es.asymLadd = args.vecDble("ASYMETRIC_LADDER");
  es.typesL5 = args.vecStr("TYPES_OF_LADD_L5");
  es.typesL4 = args.vecStr("TYPES_OF_LADD_L4");
  es.laddL5map = args.vecDble("LADD_L5_MAP");
  es.laddL4map = args.vecDble("LADD_L4_MAP");
  es.noLaddInCol = args.vecDble("NUMB_OF_LADD_IN_COL");
  es.startOfFirstLadd = args.vecDble("START_OF_1ST_LADD");
  es.typeOfLaddRow0 = args.vecStr("TYPE_OF_LADD_1");
  es.typeOfLaddRow1 = args.vecStr("TYPE_OF_LADD_2");
  es.typeOfLaddRow2 = args.vecStr("TYPE_OF_LADD_3");
  es.typeOfLaddRow3 = args.vecStr("TYPE_OF_LADD_4");
  es.thickLayers = args.vecDble("Layers");
  es.thickness = args.dble("PRESH_Z_TOTAL");
  es.materials = args.vecStr("LayMat");
  es.layName = args.vecStr("LayName");
  es.rmaxVec = args.vecDble("R_MAX");  // inner radii
  es.rminVec = args.vecDble("R_MIN");  // outer radii
  es.waf_intra_col_sep = args.dble("waf_intra_col_sep");
  es.waf_inter_col_sep = args.dble("waf_inter_col_sep");
  es.waf_active = args.dble("waf_active");
  es.wedge_length = args.dble("wedge_length");
  es.wedge_offset = args.dble("wedge_offset");
  es.zwedge_ceramic_diff = args.dble("zwedge_ceramic_diff");
  es.ywedge_ceramic_diff = args.dble("ywedge_ceramic_diff");
  es.ceramic_length = args.dble("ceramic_length");
  es.wedge_angle = args.dble("wedge_angle");
  es.wedge_back_thick = args.dble("wedge_back_thick");
  es.ladder_thick = args.dble("ladder_thick");
  es.ladder_width = args.dble("ladder_width");
  es.micromodule_length = args.dble("micromodule_length");
  es.box_thick = args.dble("box_thick");
  es.abs1stx = args.vecDble("1ST_ABSX");
  es.abs1sty = args.vecDble("1ST_ABSY");
  es.abs2ndx = args.vecDble("2ND_ABSX");
  es.abs2ndy = args.vecDble("2ND_ABSY");
  es.ladPfx = args.vecStr("LadPrefix");
  es.laddMaterial = args.str("LadderMaterial");
  es.ldrFrnt_Length = args.dble("LdrFrnt_Length");
  es.ldrFrnt_Offset = args.dble("LdrFrnt_Offset");
  es.ldrBck_Length = args.dble("LdrBck_Length");
  es.ldrBck_Offset = args.dble("LdrBck_Offset");
  es.dee_separation = args.dble("dee_sep");
  es.in_rad_Abs_Al = args.dble("R_MIN_Abs_Al");
  es.in_rad_Abs_Pb = args.dble("R_MIN_Abs_Pb");
  es.rMax_Abs_Al = args.dble("R_MAX_Abs_Al");
  es.absAlX_X = args.dble("AbsAlX_X");
  es.absAlX_Y = args.dble("AbsAlX_Y");
  es.absAlX_subtr1_Xshift = args.dble("AbsAlX_subtr1_Xshift");
  es.absAlX_subtr1_Yshift = args.dble("AbsAlX_subtr1_Yshift");
  es.absAlY_X = args.dble("AbsAlY_X");
  es.absAlY_Y = args.dble("AbsAlY_Y");
  es.absAlY_subtr1_Xshift = args.dble("AbsAlY_subtr1_Xshift");
  es.absAlY_subtr1_Yshift = args.dble("AbsAlY_subtr1_Yshift");

  // create all the tube-like layers of the preshower
  {
    double zpos = -es.thickness / 2., sdx(0), sdy(0), bdx(0), bdy(0);

    for (size_t i = 0; i < es.thickLayers.size(); ++i) {
      int I = int(i) + 1;  // FOTRAN I (offset +1)

      double rIn(0), rOut(0), zHalf(0);

      // create the name
      const std::string& ddname("esalgo:" + es.layName[i]);  // namespace:name

      // cone dimensions
      rIn = es.rminVec[i];
      rOut = es.rmaxVec[i];
      zHalf = es.thickLayers[i] / 2.;

      // create a logical part representing a single layer in the preshower
      dd4hep::Solid solid = ns.addSolid(ddname, dd4hep::Tube(ddname, rIn, rOut, zHalf, 0., 360._deg));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("SFGeomX") << ddname << " Tubs " << cms::convert2mm(zHalf) << ":" << cms::convert2mm(rIn) << ":" << cms::convert2mm(rOut) << ":0:360";
#endif
      dd4hep::Volume layer = ns.addVolume(dd4hep::Volume(ddname, solid, ns.material(es.materials[i])));

      // position the logical part w.r.t. the parent volume
      zpos += zHalf;

      // create a logical part representing a single layer in the preshower
      // skip layers with detectors, front and rear window
      if (I == 2 || I == 28 || I == 13 || I == 23) {
        zpos += zHalf;
        continue;
      }

      if (I == 12) {
        es.zlead1 = zpos + zHalf;
      }
      if (I == 22) {
        es.zlead2 = zpos + zHalf;
      }

      if (I == 10 || I == 20) {  // New lead shape

        int absz = 0;
        double outalbx, outalby, shiftR, outalbx2, outalby2, shiftR2;

        absz = int(es.abs1stx.size());
        if (I == 20)
          absz = int(es.abs2ndx.size());
        int cutabsx = -1;
        int cutabsy = -1;

        const std::string& dd_tmp_name_b("esalgo:" + es.layName[i] + "Lcut");
        const std::string& dd_tmp_name_c("esalgo:" + es.layName[i] + "tmpb");
        const std::string& dd_tmp_name_d("esalgo:" + es.layName[i] + "LinPb");
        const std::string& dd_tmp_name_e("esalgo:" + es.layName[i] + "LinAl");
        const std::string& dd_tmp_name_f("esalgo:" + es.layName[i] + "LOutAl");
	const std::string& dd_Alname_fin("esalgo:" + es.layName[i] + "LtmpAl" + std::to_string(absz - 1));

        const std::string& dd_Alname_f("esalgo:" + es.layName[i] + "LOutAl");
        const std::string& dd_Alname_g("esalgo:" + es.layName[i] + "LOutAl2");
        const std::string& dd_Alname_h("esalgo:" + es.layName[i] + "LOutAltmp");
        const std::string& dd_Alname_i("esalgo:" + es.layName[i] + "LOutAltmp2");
        const std::string& dd_Alname_j("esalgo:" + es.layName[i] + "LOutAltmp3");
        const std::string& dd_Alname_k("esalgo:" + es.layName[i] + "LOutAltmp4");
        const std::string& dd_Alname_l("esalgo:" + es.layName[i] + "LOutAltmp5");
        const std::string& dd_Alname_m("esalgo:" + es.layName[i] + "LOutAltmp6");

        dd4hep::Solid outAl = ns.addSolid(dd_Alname_f, dd4hep::Tube(dd_Alname_f, es.rMax_Abs_Al - 20 * dd4hep::cm, es.rMax_Abs_Al, zHalf - 0.1 * dd4hep::mm, 0., 90._deg));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeomX") << dd_Alname_f << " Tubs " << cms::convert2mm(zHalf - 0.1 * dd4hep::mm) << ":" << cms::convert2mm(es.rMax_Abs_Al - 20 * dd4hep::cm) << ":" << cms::convert2mm(es.rMax_Abs_Al) << ":0:90";
#endif

        outalbx = es.absAlX_X * 0.1;
        outalby = es.rMax_Abs_Al + 0.1 * dd4hep::mm - es.absAlX_subtr1_Yshift;
        shiftR = es.absAlX_subtr1_Yshift;
        if (I == 20) {
          outalbx = es.absAlY_X * 0.1;
          outalby = es.rMax_Abs_Al + 0.1 * dd4hep::mm - es.absAlY_subtr1_Yshift;
          shiftR = es.absAlY_subtr1_Xshift;
        }
        dd4hep::Solid outAltmp = ns.addSolid(dd_Alname_h,
                                     dd4hep::Box(dd_Alname_h,
                                         outalbx / 2. + 0.1 * dd4hep::mm,
                                         outalby / 2. + 0.1 * dd4hep::mm,
                                         zHalf));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeomX") << dd_Alname_h << " Box " << cms::convert2mm(outalbx / 2. + 0.1 * dd4hep::mm) << ":" << cms::convert2mm(outalby / 2. + 0.1 * dd4hep::mm) << ":" << cms::convert2mm(zHalf);
#endif
        dd4hep::Solid outAltmp3 = ns.addSolid(
            dd_Alname_j,
            dd4hep::SubtractionSolid(dd_Alname_j, outAl, outAltmp, dd4hep::Position(outalbx / 2., outalby / 2. + shiftR, 0)));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeomX") << dd_Alname_j << " Subtraction " << outAl.name() << ":" << outAltmp.name() << " at (" << cms::convert2mm(outalbx / 2.) << "," << cms::convert2mm(outalby / 2. + shiftR) << "," << "0) no rotation";
#endif

        outalby2 = es.absAlX_Y * 0.1;
        outalbx2 = es.rMax_Abs_Al + 0.1 * dd4hep::mm - es.absAlX_subtr1_Xshift;
        shiftR2 = es.absAlX_subtr1_Xshift;
        if (I == 20) {
          outalby2 = es.absAlY_Y * 0.1;
          outalbx2 = es.rMax_Abs_Al + 0.1 * dd4hep::mm - es.absAlY_subtr1_Xshift;
          shiftR2 = es.absAlY_subtr1_Xshift;
        }
        dd4hep::Solid outAltmp2 = ns.addSolid(dd_Alname_i,
                                      dd4hep::Box(dd_Alname_i,
                                          outalbx2 / 2. + 0.1 * dd4hep::mm,
                                          outalby2 / 2. + 0.1 * dd4hep::mm,
                                          zHalf));
        dd4hep::Solid outAltmp4 = ns.addSolid(
            dd_Alname_k,
            dd4hep::SubtractionSolid(dd_Alname_k, outAltmp3, outAltmp2, dd4hep::Position(outalbx2 / 2. + shiftR2, outalby2 / 2., 0)));
        dd4hep::Solid outAltmp5 =
            ns.addSolid(dd_Alname_l, dd4hep::UnionSolid(dd_Alname_l, outAltmp4, outAltmp4, ns.rotation("esalgo:RABS90")));
        dd4hep::Solid outAltmp6 =
            ns.addSolid(dd_Alname_m, dd4hep::UnionSolid(dd_Alname_m, outAltmp5, outAltmp4, ns.rotation("esalgo:RABS180B")));
        dd4hep::Solid outAl2 =
            ns.addSolid(dd_Alname_g, dd4hep::UnionSolid(dd_Alname_g, outAltmp6, outAltmp4, ns.rotation("esalgo:R180")));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeomX") << dd_Alname_i << " Box " << cms::convert2mm(outalbx2 / 2. + 0.1 * dd4hep::mm) << ":" << cms::convert2mm(outalby2 / 2. + 0.1 * dd4hep::mm) << ":" << cms::convert2mm(zHalf);
	edm::LogVerbatim("SFGeomX") << dd_Alname_k << " Subtraction " << outAltmp3.name() << ":" << outAltmp2.name() << " at (" << cms::convert2mm(outalbx2 / 2. + shiftR2) << "," << cms::convert2mm(outalby2 / 2) << ",0) no rotation";
	edm::LogVerbatim("SFGeomX") << dd_Alname_l << " Union " << outAltmp4.name() << ":" << outAltmp4.name() << " at (0,0,0) rotation esalgo:RABS90";
	edm::LogVerbatim("SFGeomX") << dd_Alname_m << " Union " << outAltmp5.name() << ":" << outAltmp4.name() << " at (0,0,0) rotation esalgo:RABS180B";
	edm::LogVerbatim("SFGeomX") << dd_Alname_g << " Union " << outAltmp6.name() << ":" << outAltmp4.name() << " at (0,0,0) rotation esalgo:R180";
#endif

        dd4hep::Solid outAlCut = dd4hep::Box(65 * dd4hep::cm, 60 * dd4hep::cm - 0.1 * dd4hep::mm, zHalf + 0.2 * dd4hep::mm);
        dd4hep::Solid outAlFin = ns.addSolid(dd_Alname_fin, dd4hep::SubtractionSolid(dd_Alname_fin, outAl2, outAlCut));
        dd4hep::Volume layerFinOutAl = dd4hep::Volume(dd_tmp_name_f, outAlFin, ns.material(es.materials[i - 1]));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeomX") << outAlCut.name() << " Box " << cms::convert2mm(65 * dd4hep::cm) << ":" << cms::convert2mm(60 * dd4hep::cm - 0.1 * dd4hep::mm) << ":" << cms::convert2mm(zHalf + 0.2 * dd4hep::mm);
	edm::LogVerbatim("SFGeomX") << outAlFin.name() << " Subtraction " << outAl2.name() << ":" << outAlCut.name() << " at (0,0,0) no rotation";
#endif
        for (int L = 0; L < absz; ++L) {
          int K = L;
          std::ostringstream tmp_name_b, tmp_name_b2, tmp_FAl_name_c, tmp_FAl_name_d1, tmp_FAl_name_d2, tmp_FAl_name_d3,
              tmp_FAl_name_d;
          tmp_name_b << es.layName[i] << "L" << K;
          tmp_name_b2 << es.layName[i] << "Lb2" << K;

          if (L == 0)
            tmp_FAl_name_c << es.layName[i] << "LOutAl2";
          if (L > 0)
            tmp_FAl_name_c << es.layName[i] << "LtmpAl" << K - 1;

          tmp_FAl_name_d1 << es.layName[i] << "LtmpAl" << K << "_1";
          tmp_FAl_name_d2 << es.layName[i] << "LtmpAl" << K << "_2";
          tmp_FAl_name_d3 << es.layName[i] << "LtmpAl" << K << "_3";
          tmp_FAl_name_d << es.layName[i] << "LtmpAl" << K;

          const std::string& dd_tmp_name_b("esalgo:" + tmp_name_b.str());
          const std::string& dd_tmp_name_b2("esalgo:" + tmp_name_b2.str());
          const std::string& dd_FAl_name_c("esalgo:" + tmp_FAl_name_c.str());
          const std::string& dd_FAl_name_d1("esalgo:" + tmp_FAl_name_d1.str());
          const std::string& dd_FAl_name_d2("esalgo:" + tmp_FAl_name_d2.str());
          const std::string& dd_FAl_name_d3("esalgo:" + tmp_FAl_name_d3.str());
          const std::string& dd_FAl_name_d("esalgo:" + tmp_FAl_name_d.str());

          if (L == 0)
            bdx = abs(es.abs1stx[K]) / 2.;
          if (L > 0)
            bdx = abs(es.abs1stx[K] - es.abs1stx[K - 1]) / 2.;
          bdy = es.abs1sty[K];
          if (es.abs1stx[K] < rIn + 30 * dd4hep::cm) {
            bdy = es.abs1sty[K] / 2. - 30 * dd4hep::cm;
            cutabsx = K;
          }

          if (I == 20) {
            if (L == 0)
              bdx = abs(es.abs2ndx[K]) / 2.;
            if (L > 0)
              bdx = abs(es.abs2ndx[K] - es.abs2ndx[K - 1]) / 2.;
            bdy = es.abs2ndy[K];
          }

          if ((es.abs2ndx[K] < rIn + 30 * dd4hep::cm) && I == 20) {
            bdy = es.abs2ndy[K] / 2. - 30 * dd4hep::cm;
            cutabsy = K;
          }

          dd4hep::Solid solid_b = dd4hep::Box(dd_tmp_name_b, bdx, bdy, zHalf);
          dd4hep::Solid solid_b2 = dd4hep::Box(dd_tmp_name_b2, bdx + 0.1 * dd4hep::mm, bdy + 0.1 * dd4hep::mm, zHalf);
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeomX") << dd_tmp_name_b << " Box " << cms::convert2mm(bdx) << ":" << cms::convert2mm(bdy) << ":" << cms::convert2mm(zHalf);
	edm::LogVerbatim("SFGeomX") << dd_tmp_name_b2 << " Box " << cms::convert2mm(bdx + 0.1 * dd4hep::mm) << ":" << cms::convert2mm(bdy + 0.1 * dd4hep::mm) << ":" << cms::convert2mm(zHalf);
#endif
          sdx = es.abs1stx[K] - bdx;
          sdy = 0;
          if (es.abs1stx[K] < rIn + 30 * dd4hep::cm)
            sdy = es.abs1sty[K] - bdy;

          if (I == 20) {
            sdx = es.abs2ndx[K] - bdx;
            sdy = 0;
          }
          if ((es.abs2ndx[K] < rIn + 30 * dd4hep::cm) && I == 20)
            sdy = es.abs2ndy[K] - bdy;

          dd4hep::Volume layer = dd4hep::Volume(dd_tmp_name_b, solid_b, ns.material(es.materials[i]));

          layerFinOutAl.placeVolume(layer, 1, dd4hep::Position(sdx, sdy, 0));
          layerFinOutAl.placeVolume(layer, 2, dd4hep::Position(-sdx, sdy, 0));
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeom") << layer.name() << " copy 1 in " << layerFinOutAl.name() << " at (" << cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << ",0) no rotation";
	  edm::LogVerbatim("SFGeom") << layer.name() << " copy 2 in " << layerFinOutAl.name() << " at (" << -cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << ",0) no rotation";
#endif

          dd4hep::Solid solid_c = ns.solid(dd_FAl_name_c);
          dd4hep::Solid solid_d1 = dd4hep::UnionSolid(dd_FAl_name_d1, solid_c, solid_b2, dd4hep::Position(sdx, sdy, 0));
	  ns.addSolid(dd_FAl_name_d, dd4hep::UnionSolid(dd_FAl_name_d, solid_d1, solid_b2, dd4hep::Position(-sdx, -sdy, 0)));
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeomX") << dd_FAl_name_d1 << " Union " << solid_c.name() << ":" << solid_b2.name() << " at (" << cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << ",0) no rotation";
	  edm::LogVerbatim("SFGeomX") << dd_FAl_name_d << " Union " << solid_d1.name() << ":" << solid_b2.name() << " at (" << -cms::convert2mm(sdx) << "," << -cms::convert2mm(sdy) << ",0) no rotation";
#endif
          if (((es.abs1stx[K] < rIn + 30 * dd4hep::cm) && I == 10) ||
              ((es.abs2ndx[K] < rIn + 30 * dd4hep::cm) && I == 20)) {
            layerFinOutAl.placeVolume(layer, 3, dd4hep::Position(sdx, -sdy, 0));
            layerFinOutAl.placeVolume(layer, 4, dd4hep::Position(-sdx, -sdy, 0));
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeom") << layer.name() << " copy 3 in " << layerFinOutAl.name() << " at (" << cms::convert2mm(sdx) << "," << -cms::convert2mm(sdy) << ",0) no rotation";
	    edm::LogVerbatim("SFGeom") << layer.name() << " copy 4 in " << layerFinOutAl.name() << " at (" << -cms::convert2mm(sdx) << "," << -cms::convert2mm(sdy) << ",0) no rotation";
#endif
            dd4hep::Solid solid_c = ns.solid(dd_FAl_name_c);
            dd4hep::Solid solid_d1 = dd4hep::UnionSolid(dd_FAl_name_d1, solid_c, solid_b2, dd4hep::Position(sdx, sdy, 0));
	    dd4hep::Solid solid_d2 = dd4hep::UnionSolid(dd_FAl_name_d2, solid_d1, solid_b2, dd4hep::Position(sdx, -sdy, 0));
            dd4hep::Solid solid_d3 = dd4hep::UnionSolid(dd_FAl_name_d3, solid_d2, solid_b2, dd4hep::Position(-sdx, sdy, 0));
            ns.addSolid(dd_FAl_name_d, dd4hep::UnionSolid(dd_FAl_name_d, solid_d3, solid_b2, dd4hep::Position(-sdx, -sdy, 0)));
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << dd_FAl_name_d1 << " Union " << solid_c.name() << ":" << solid_b2.name() << " at (" << cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << ",0) no rotation";
	    edm::LogVerbatim("SFGeomX") << dd_FAl_name_d2 << " Union " << solid_d1.name() << ":" << solid_b2.name() << " at (" << cms::convert2mm(sdx) << "," << -cms::convert2mm(sdy) << ",0) no rotation";
	    edm::LogVerbatim("SFGeomX") << dd_FAl_name_d3 << " Union " << solid_d2.name() << ":" << solid_b2.name() << " at (" << -cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << ",0) no rotation";
	    edm::LogVerbatim("SFGeomX") << dd_FAl_name_d << " Union " << solid_d3.name() << ":" << solid_b2.name() << " at (" << -cms::convert2mm(sdx) << "," << -cms::convert2mm(sdy) << ",0) no rotation";
#endif
          }
        }

        bdx = es.abs1stx[cutabsx];
        if (I == 20)
          bdx = es.abs2ndx[cutabsy];
        bdy = 2 * 30 * dd4hep::cm;

        dd4hep::Solid solidcut = dd4hep::Box(dd_tmp_name_b, bdx, bdy, zHalf);
        dd4hep::Solid iner = dd4hep::Tube(dd_tmp_name_c, 0, es.in_rad_Abs_Pb, zHalf + 0.1 * dd4hep::mm, 0., 360._deg);
        dd4hep::Solid final = dd4hep::SubtractionSolid(dd_tmp_name_d, solidcut, iner);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeomX") << dd_tmp_name_b << " Box " << cms::convert2mm(bdx) << ":" << cms::convert2mm(bdy) << ":" << cms::convert2mm(zHalf);
	edm::LogVerbatim("SFGeomX") << dd_tmp_name_c << " Tubs " << cms::convert2mm(zHalf + 0.1 * dd4hep::mm) << ":0:" << cms::convert2mm(es.in_rad_Abs_Pb) << ":0:360";
	edm::LogVerbatim("SFGeomX") << dd_tmp_name_d << " Subtraction " << solidcut.name() << ":" << iner.name() << " at (0,0,0) no rotation";
#endif

        dd4hep::Volume blayer = dd4hep::Volume(dd_tmp_name_d, final, ns.material(es.materials[i]));
        parentVolume.placeVolume(blayer, 1, dd4hep::Position(0, 0, zpos));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeom") << blayer.name() << " copy 1 in " << parentVolume.name() << " at (0,0," << cms::convert2mm(zpos) << ") no rotation";
#endif
        dd4hep::Solid iner_Al =
            dd4hep::Tube(dd_tmp_name_e, es.in_rad_Abs_Al, es.in_rad_Abs_Pb - 0.01 * dd4hep::mm, zHalf, 0., 360._deg);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeomX") << dd_tmp_name_e << " Tubs " << cms::convert2mm(zHalf) << ":" << cms::convert2mm(es.in_rad_Abs_Al) << ":" << cms::convert2mm(es.in_rad_Abs_Pb - 0.01 * dd4hep::mm) << ":0:360";
#endif
        dd4hep::Volume layerAl = dd4hep::Volume(dd_tmp_name_e, iner_Al, ns.material(es.materials[i - 1]));
        parentVolume.placeVolume(layerAl, 1, dd4hep::Position(0, 0, zpos));
        parentVolume.placeVolume(layerFinOutAl, 1, dd4hep::Position(0, 0, zpos));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeom") << layerAl.name() << " copy 1 in " << parentVolume.name() << " at (0,0," << cms::convert2mm(zpos) << ") no rotation";
	edm::LogVerbatim("SFGeom") << layerFinOutAl.name() << " copy 1 in " << parentVolume.name() << " at (0,0," << cms::convert2mm(zpos) << ") no rotation";
#endif
      } else {
        parentVolume.placeVolume(layer, 1, dd4hep::Position(0., 0., zpos));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeom") << layer.name() << " copy 1 in " << parentVolume.name() << " at (0,0," << cms::convert2mm(zpos) << ") no rotation";
#endif
      }
      zpos += zHalf;
    }
  }
  // create and place the ladders
  {
    double xpos(0.), ypos(0.), zpos(0.), sdx(0.), sdy(0.), sdz(0.);
    double prev_length(0.), ladder_new_length(0.);
    double ladd_shift(0.);
    double ladder_length(0.);
    int enb(0), swed_scopy_glob(0);
    double sdxe[50] = {0}, sdye[50] = {0}, sdze[50] = {0};
    double sdxe2[50] = {0}, sdye2[50] = {0}, sdze2[50] = {0}, sdxe3[50] = {0}, sdye3[50] = {0}, sdze3[50] = {0};

    for (int M = 0; M < int(es.typesL5.size() + es.typesL4.size()); M++) {
      int scopy(0);
      double boxax(0.), boxay(0.), boxaz(0.);
      int ladd_not_plain(0), ladd_subtr_no(0), ladd_upper(0), ladd_side(0);
      dd4hep::Solid solid_lfront = dd4hep::Trap("esalgo:LDRFRNT",
                                es.ldrFrnt_Length / 2.,                                                 // pDz
                                -es.wedge_angle,                                                        // pTheta
                                0,                                                                      // pPhi
                                es.ladder_width / 2.,                                                   // pDy1
                                (es.ladder_thick) / 2.,                                                 // pDx1
                                (es.ladder_thick) / 2.,                                                 // pDx2
                                0,                                                                      // pAlp1
                                es.ladder_width / 2.,                                                   // pDy2
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.,  // pDx3
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.,  // pDx4
                                0.);

      dd4hep::Solid solid_lbck = dd4hep::Trap("esalgo:LDRBCK",
                              es.ldrBck_Length / 2.,                                              // pDz
                              -es.wedge_angle,                                                    // pTheta
                              0,                                                                  // pPhi
                              es.ladder_width / 2.,                                               // pDy1
                              (es.box_thick / cos(es.wedge_angle * 2) + 0.02 * dd4hep::mm) / 2.,  // pDx1
                              (es.box_thick / cos(es.wedge_angle * 2) + 0.02 * dd4hep::mm) / 2.,  // pDx2
                              0,                                                                  // pAlp1
                              es.ladder_width / 2.,                                               // pDy2
                              (es.ladder_thick - es.wedge_back_thick) / 2.,                       // pDx3
                              (es.ladder_thick - es.wedge_back_thick) / 2.,                       // pDx4
                              0.);

      dd4hep::Solid solid_lfhalf = dd4hep::Trap("esalgo:LDRFHALF",
                                es.ldrFrnt_Length / 2.,                                                 // pDz
                                -es.wedge_angle,                                                        // pTheta
                                0,                                                                      // pPhi
                                (es.ladder_width / 2.) / 2.,                                            // pDy1
                                (es.ladder_thick) / 2.,                                                 // pDx1
                                (es.ladder_thick) / 2.,                                                 // pDx2
                                0,                                                                      // pAlp1
                                (es.ladder_width / 2.) / 2.,                                            // pDy2
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.,  // pDx3
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.,  // pDx4
                                0.);

      dd4hep::Solid solid_lbhalf = dd4hep::Trap("esalgo:LDRBHALF",
                                es.ldrBck_Length / 2.,                                               // pDz
                                -es.wedge_angle,                                                     // pTheta
                                0,                                                                   // pPhi
                                (es.ladder_width / 2.) / 2.,                                         // pDy1
                                (es.box_thick / cos(es.wedge_angle * 2.) + 0.02 * dd4hep::mm) / 2.,  // pDx1
                                (es.box_thick / cos(es.wedge_angle * 2.) + 0.02 * dd4hep::mm) / 2.,  // pDx2
                                0,                                                                   // pAlp1
                                (es.ladder_width / 2.) / 2.,                                         // pDy2
                                (es.ladder_thick - es.wedge_back_thick) / 2.,                        // pDx3
                                (es.ladder_thick - es.wedge_back_thick) / 2.,                        // pDx4
                                0.);

      dd4hep::Solid solid_lfhtrunc =
          dd4hep::Trap("esalgo:LDRFHTR",
               (es.ldrFrnt_Length - es.waf_active) / 2.,                                                // pDz
               -es.wedge_angle,                                                                         // pTheta
               0,                                                                                       // pPhi
               (es.ladder_width / 2.) / 2.,                                                             // pDy1
               (es.ladder_thick) / 2.,                                                                  // pDx1
               (es.ladder_thick) / 2.,                                                                  // pDx2
               0,                                                                                       // pAlp1
               (es.ladder_width / 2.) / 2.,                                                             // pDy2
               (es.ladder_thick - (es.ceramic_length - es.waf_active) * sin(es.wedge_angle * 2)) / 2.,  // pDx3
               (es.ladder_thick - (es.ceramic_length - es.waf_active) * sin(es.wedge_angle * 2)) / 2.,  // pDx4
               0.);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("SFGeomX") << "esalgo:LDRFRNT Trap " << cms::convert2mm(es.ldrFrnt_Length / 2.) << ":" << -convertRadToDeg(es.wedge_angle) << ":0:" << cms::convert2mm(es.ladder_width / 2.) << ":" << cms::convert2mm((es.ladder_thick) / 2.) << ":" << cms::convert2mm((es.ladder_thick) / 2.) << ":0:" << cms::convert2mm(es.ladder_width / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.) << ":0";
      edm::LogVerbatim("SFGeomX") << "esalgo:LDRBCK Trap " << cms::convert2mm(es.ldrBck_Length / 2.) << ":" << -convertRadToDeg(es.wedge_angle) << ":0:" << cms::convert2mm(es.ladder_width / 2.) << ":" << cms::convert2mm((es.box_thick / cos(es.wedge_angle * 2) + 0.02 * dd4hep::mm) / 2.) << ":" << cms::convert2mm((es.box_thick / cos(es.wedge_angle * 2) + 0.02 * dd4hep::mm) / 2.) << ":0:" << cms::convert2mm(es.ladder_width / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.wedge_back_thick) / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.wedge_back_thick) / 2.) << ":0";
      edm::LogVerbatim("SFGeomX") << "esalgo:LDRFHALF Trap " << cms::convert2mm(es.ldrFrnt_Length / 2.) << ":" << -convertRadToDeg(es.wedge_angle) << ":0:" << cms::convert2mm((es.ladder_width / 2.) / 2.) << ":" << cms::convert2mm((es.ladder_thick) / 2.) << ":" << cms::convert2mm((es.ladder_thick) / 2.) << ":0:" << cms::convert2mm((es.ladder_width / 2.) / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2.)) / 2.) << ":0";
      edm::LogVerbatim("SFGeomX") << "esalgo:LDRBHALF Trap " << cms::convert2mm(es.ldrBck_Length / 2.) << ":" << -convertRadToDeg(es.wedge_angle) << ":0:" << cms::convert2mm((es.ladder_width / 2.) / 2.) << ":" << cms::convert2mm((es.box_thick / cos(es.wedge_angle * 2.) + 0.02 * dd4hep::mm) / 2.) << ":" << cms::convert2mm((es.box_thick / cos(es.wedge_angle * 2.) + 0.02 * dd4hep::mm) / 2.) << ":0:" << cms::convert2mm((es.ladder_width / 2.) / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.wedge_back_thick) / 2.) << ":" << cms::convert2mm((es.ladder_thick - es.wedge_back_thick) / 2.) << ":0";
      edm::LogVerbatim("SFGeomX") << "esalgo:LDRFHTR Trap " << cms::convert2mm((es.ldrFrnt_Length - es.waf_active) / 2.) << ":" << -convertRadToDeg(es.wedge_angle) << ":0:" << cms::convert2mm((es.ladder_width / 2.) / 2.) << ":" << cms::convert2mm((es.ladder_thick) / 2.) << ":" << cms::convert2mm((es.ladder_thick) / 2.) << ":0:" << cms::convert2mm((es.ladder_width / 2.) / 2.) << ":" << cms::convert2mm((es.ladder_thick - (es.ceramic_length - es.waf_active) * sin(es.wedge_angle * 2)) / 2.) << ":" << cms::convert2mm((es.ladder_thick - (es.ceramic_length - es.waf_active) * sin(es.wedge_angle * 2)) / 2.) << ":0";
#endif

      // Creation of ladders with 5 micromodules length

      if (M < int(es.typesL5.size())) {
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            if (es.laddL5map[(i + j * 2 + M * 10)] != 1) {
              ladd_not_plain = 1;
              ladd_subtr_no++;
              if (j > 1)
                ladd_upper = 1;
	      ladd_side = i;
            }
          }
        }

        const std::string& ddname("esalgo:" + es.ladPfx[0] + es.typesL5[M]);
        ladder_length = es.micromodule_length + 4 * es.waf_active + 0.1 * dd4hep::mm;

        if (ladd_not_plain) {
          if (!ladd_upper) {
            enb++;
            const std::string& dd_tmp_name_5a("esalgo:" + es.ladPfx[2]);
            const std::string& dd_tmp_name_5b("esalgo:" + es.ladPfx[3] + std::to_string(enb));
            const std::string& dd_tmp_name_5c("esalgo:" + es.ladPfx[4] + std::to_string(enb));

            boxay = ladder_length - es.ldrFrnt_Length - es.ldrBck_Length;
            boxax = es.ladder_width;
            boxaz = es.ladder_thick;

            dd4hep::Solid solid_5a = dd4hep::Box(dd_tmp_name_5a, boxax / 2., boxay / 2., boxaz / 2.);
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << dd_tmp_name_5a << " Box " << cms::convert2mm(boxax / 2.) << ":" << cms::convert2mm(boxay / 2.) << ":" << cms::convert2mm(boxaz / 2.);
#endif
            if (ladd_side == 0)
              sdxe[enb] = es.ladder_width / 4.;
            sdye[enb] = -boxay / 2. - es.ldrFrnt_Length / 2.;
            sdze[enb] = -es.ladder_thick / 2. + es.ldrFrnt_Offset;
            if (ladd_side == 1)
              sdxe[enb] = -es.ladder_width / 4.;

            dd4hep::Solid solid_5b =
                dd4hep::UnionSolid(dd_tmp_name_5b,
                           solid_5a,
                           solid_lfhalf,
                           dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe[enb], sdye[enb], sdze[enb])));
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << dd_tmp_name_5b << " Union " << solid_5a.name() << ":" << solid_lfhalf.name() << " at (" << cms::convert2mm(sdxe[enb]) << "," << cms::convert2mm(sdye[enb]) << "," << cms::convert2mm(sdze[enb]) << ") rotation esalgo:RM1299";
#endif

            if (ladd_side == 0)
              sdxe2[enb] = -es.ladder_width / 4.;
            sdye2[enb] = -boxay / 2. - es.ldrFrnt_Length / 2. + es.waf_active / 2.;
            sdze2[enb] = -es.ladder_thick / 2. + es.ldrFrnt_Offset + (es.waf_active * sin(es.wedge_angle * 2)) / 4.;
            if (ladd_side == 1)
              sdxe2[enb] = es.ladder_width / 4.;

            dd4hep::Solid solid_5c =
                dd4hep::UnionSolid(dd_tmp_name_5c,
                           solid_5b,
                           solid_lfhtrunc,
                           dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe2[enb], sdye2[enb], sdze2[enb])));
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << dd_tmp_name_5c << " Union " << solid_5b.name() << ":" << solid_lfhtrunc.name() << " at (" << cms::convert2mm(sdxe2[enb]) << "," << cms::convert2mm(sdye2[enb]) << "," << cms::convert2mm(sdze2[enb]) << ") rotation esalgo:RM1299";
#endif

            sdxe3[enb] = 0;
            sdye3[enb] = boxay / 2. + es.ldrBck_Length / 2.;
            sdze3[enb] = -es.ladder_thick / 2. + es.ldrBck_Offset;
            dd4hep::Solid solid =
                dd4hep::UnionSolid(ddname,
                           solid_5c,
                           solid_lbck,
                           dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe3[enb], sdye3[enb], sdze3[enb])));
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << ddname << " Union " << solid_5c.name() << ":" << solid_lbck.name() << " at (" << cms::convert2mm(sdxe3[enb]) << "," << cms::convert2mm(sdye3[enb]) << "," << cms::convert2mm(sdze3[enb]) << ") rotation esalgo:RM1299";
#endif

            ns.addVolumeNS(dd4hep::Volume(ddname, solid, ns.material(es.laddMaterial)));
            ns.addVolumeNS(dd4hep::Volume("esalgo:" + es.ladPfx[1] + es.typesL5[M], solid, ns.material(es.laddMaterial)));
          }
        }  // end of not plain ladder shape
        else {
          const std::string& dd_tmp_name_5pa("esalgo:" + es.ladPfx[2] + "5p");
          const std::string& dd_tmp_name_5pb("esalgo:" + es.ladPfx[3] + "5p");

          boxay = ladder_length - es.ldrFrnt_Length - es.ldrBck_Length;
          boxax = es.ladder_width;
          boxaz = es.ladder_thick;

          dd4hep::Solid solid_5pa = dd4hep::Box(dd_tmp_name_5pa, boxax / 2., boxay / 2., boxaz / 2.);
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << dd_tmp_name_5pa << " Box " << cms::convert2mm(boxax / 2) << ":" << cms::convert2mm(boxay / 2) << ":" << cms::convert2mm(boxaz / 2);
#endif
          sdx = 0;
          sdy = -boxay / 2. - es.ldrFrnt_Length / 2.;
          sdz = -es.ladder_thick / 2. + es.ldrFrnt_Offset;

          dd4hep::Solid solid_5pb = dd4hep::UnionSolid(dd_tmp_name_5pb,
                                       solid_5pa,
                                       solid_lfront,
                                       dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdx, sdy, sdz)));
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeomX") << dd_tmp_name_5pb << " Union " << solid_5pa.name() << ":" << solid_lfront.name() << " at (" << cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << "," << cms::convert2mm(sdz) << ") rotation esalgo:RM1299";
#endif

          sdx = 0;
          sdy = boxay / 2. + es.ldrBck_Length / 2.;
          sdz = -es.ladder_thick / 2. + es.ldrBck_Offset;

          dd4hep::Solid solid = dd4hep::UnionSolid(
              ddname, solid_5pb, solid_lbck, dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdx, sdy, sdz)));
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeomX") << ddname << " Union " << solid_5pb.name() << ":" << solid_lbck.name() << " at (" << cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << "," << cms::convert2mm(sdz) << ") rotation esalgo:RM1299";
#endif
          ns.addVolumeNS(dd4hep::Volume(ddname, solid, ns.material(es.laddMaterial)));
          ns.addVolumeNS(dd4hep::Volume("esalgo:" + es.ladPfx[1] + es.typesL5[M], solid, ns.material(es.laddMaterial)));
        }
      }

      // Creation of ladders with 4 micromodules length

      if (M >= int(es.typesL5.size())) {
        int d = M - es.typesL5.size();

        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            if (es.laddL4map[(i + j * 2 + (M - es.typesL5.size()) * 8)] != 1) {
              ladd_not_plain = 1;
              ladd_subtr_no++;
              if (j > 1)
                ladd_upper = 1;
	      ladd_side = i;
            }
          }
        }

        const std::string& ddname("esalgo:" + es.ladPfx[0] + es.typesL4[d]);
        ladder_length = es.micromodule_length + 3 * es.waf_active + 0.1 * dd4hep::mm;

        if (ladd_not_plain) {
          if (ladd_upper) {
            enb++;

            const std::string& dd_tmp_name_a("esalgo:" + es.ladPfx[7]);
            const std::string& dd_tmp_name_b("esalgo:" + es.ladPfx[8] + std::to_string(enb));

            boxay = ladder_length - es.ldrFrnt_Length - es.ldrBck_Length;
            boxax = es.ladder_width;
            boxaz = es.ladder_thick;
            dd4hep::Solid solid_a = dd4hep::Box(dd_tmp_name_a, boxax / 2., boxay / 2., boxaz / 2.);
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << dd_tmp_name_a << " Box " << cms::convert2mm(boxax / 2) << ":" << cms::convert2mm(boxay / 2) << ":" << cms::convert2mm(boxaz / 2);
#endif

            sdxe[enb] = 0;
            sdye[enb] = -boxay / 2. - es.ldrFrnt_Length / 2.;
            sdze[enb] = -es.ladder_thick / 2. + es.ldrFrnt_Offset;
            dd4hep::Solid solid_b =
                dd4hep::UnionSolid(dd_tmp_name_b,
                           solid_a,
                           solid_lfront,
                           dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe[enb], sdye[enb], sdze[enb])));
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << dd_tmp_name_b << " Union " << solid_a.name() << ":" << solid_lfront.name() << " at (" << cms::convert2mm(sdxe[enb]) << "," << cms::convert2mm(sdye[enb]) << "," << cms::convert2mm(sdze[enb]) << ") rotation esalgo:RM1299";
#endif

            if (ladd_side == 0)
              sdxe2[enb] = es.ladder_width / 4.;
            sdye2[enb] = boxay / 2. + es.ldrBck_Length / 2.;
            sdze2[enb] = -es.ladder_thick / 2. + es.ldrBck_Offset;
            if (ladd_side == 1)
              sdxe2[enb] = -es.ladder_width / 4.;
            dd4hep::Solid solid =
                dd4hep::UnionSolid(ddname,
                           solid_b,
                           solid_lbhalf,
                           dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe2[enb], sdye2[enb], sdze2[enb])));
#ifdef EDM_ML_DEBUG
	    edm::LogVerbatim("SFGeomX") << ddname << " Union " << solid_b.name() << ":" << solid_lbhalf.name() << " at (" << cms::convert2mm(sdxe2[enb]) << "," << cms::convert2mm(sdye2[enb]) << "," << cms::convert2mm(sdze2[enb]) << ") rotation esalgo:RM1299";
#endif

            ns.addVolumeNS(dd4hep::Volume(ddname, solid, ns.material(es.laddMaterial)));
            ns.addVolumeNS(dd4hep::Volume("esalgo:" + es.ladPfx[1] + es.typesL4[d], solid, ns.material(es.laddMaterial)));

          }  // upper
          else {
            if (ladd_subtr_no > 1) {
              enb++;

              const std::string& dd_tmp_name_a("esalgo:" + es.ladPfx[7]);
              const std::string& dd_tmp_name_b("esalgo:" + es.ladPfx[8] + std::to_string(enb));

              boxay = ladder_length - es.ldrFrnt_Length - es.ldrBck_Length;
              boxax = es.ladder_width;
              boxaz = es.ladder_thick;

              dd4hep::Solid solid_a = dd4hep::Box(dd_tmp_name_a, boxax / 2., boxay / 2., boxaz / 2.);
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeomX") << dd_tmp_name_a << " Box " << cms::convert2mm(boxax / 2) << ":" << cms::convert2mm(boxay / 2) << ":" << cms::convert2mm(boxaz / 2);
#endif
              if (ladd_side == 0)
                sdxe[enb] = es.ladder_width / 4.;
              sdye[enb] = -boxay / 2. - es.ldrFrnt_Length / 2.;
              sdze[enb] = -es.ladder_thick / 2. + es.ldrFrnt_Offset;
              if (ladd_side == 1)
                sdxe[enb] = -es.ladder_width / 4.;

              dd4hep::Solid solid_b =
                  dd4hep::UnionSolid(dd_tmp_name_b,
                             solid_a,
                             solid_lfhalf,
                             dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe[enb], sdye[enb], sdze[enb])));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeomX") << dd_tmp_name_b << " Union " << solid_a.name() << ":" << solid_lfhalf.name() << " at (" << cms::convert2mm(sdxe[enb]) << "," << cms::convert2mm(sdye[enb]) << "," << cms::convert2mm(sdze[enb]) << ") rotation esalgo:RM1299";
#endif

              sdxe2[enb] = 0;
              sdye2[enb] = boxay / 2. + es.ldrBck_Length / 2.;
              sdze2[enb] = -es.ladder_thick / 2. + es.ldrBck_Offset;

              dd4hep::Solid solid =
                  dd4hep::UnionSolid(ddname,
                             solid_b,
                             solid_lbck,
                             dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe2[enb], sdye2[enb], sdze2[enb])));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeomX") << ddname << " Union " << solid_b.name() << ":" << solid_lbck.name() << " at (" << cms::convert2mm(sdxe2[enb]) << "," << cms::convert2mm(sdye2[enb]) << "," << cms::convert2mm(sdze2[enb]) << ") rotation esalgo:RM1299";
#endif

              ns.addVolumeNS(dd4hep::Volume(ddname, solid, ns.material(es.laddMaterial)));
              ns.addVolumeNS(dd4hep::Volume("esalgo:" + es.ladPfx[1] + es.typesL4[d], solid, ns.material(es.laddMaterial)));
            } else {
              enb++;
              const std::string& dd_tmp_name_a("esalgo:" + es.ladPfx[7]);
              const std::string& dd_tmp_name_b("esalgo:" + es.ladPfx[8] + std::to_string(enb));
              const std::string& dd_tmp_name_c("esalgo:" + es.ladPfx[9] + std::to_string(enb));

              boxay = ladder_length - es.ldrFrnt_Length - es.ldrBck_Length;
              boxax = es.ladder_width;
              boxaz = es.ladder_thick;
              dd4hep::Solid solid_a = dd4hep::Box(dd_tmp_name_a, boxax / 2., boxay / 2., boxaz / 2.);
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeomX") << dd_tmp_name_a << " Box " << cms::convert2mm(boxax / 2) << ":" << cms::convert2mm(boxay / 2) << ":" << cms::convert2mm(boxaz / 2);
#endif
              if (ladd_side == 0)
                sdxe[enb] = es.ladder_width / 4.;
              sdye[enb] = -boxay / 2. - es.ldrFrnt_Length / 2.;
              sdze[enb] = -es.ladder_thick / 2. + es.ldrFrnt_Offset;
              if (ladd_side == 1)
                sdxe[enb] = -es.ladder_width / 4.;

              dd4hep::Solid solid_b =
                  dd4hep::UnionSolid(dd_tmp_name_b,
                             solid_a,
                             solid_lfhalf,
                             dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe[enb], sdye[enb], sdze[enb])));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeomX") << dd_tmp_name_b << " Union " << solid_a.name() << ":" << solid_lfhalf.name() << " at (" << cms::convert2mm(sdxe[enb]) << "," << cms::convert2mm(sdye[enb]) << "," << cms::convert2mm(sdze[enb]) << ") rotation esalgo:RM1299";
#endif

              if (ladd_side == 0)
                sdxe2[enb] = -es.ladder_width / 4.;
              sdye2[enb] = -boxay / 2. - es.ldrFrnt_Length / 2. + es.waf_active / 2.;
              sdze2[enb] = -es.ladder_thick / 2. + es.ldrFrnt_Offset + (es.waf_active * sin(es.wedge_angle * 2)) / 4.;
              if (ladd_side == 1)
                sdxe2[enb] = es.ladder_width / 4.;

              dd4hep::Solid solid_c =
                  dd4hep::UnionSolid(dd_tmp_name_c,
                             solid_b,
                             solid_lfhtrunc,
                             dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe2[enb], sdye2[enb], sdze2[enb])));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeomX") << dd_tmp_name_c << " Union " << solid_b.name() << ":" << solid_lfhtrunc.name() << " at (" << cms::convert2mm(sdxe2[enb]) << "," << cms::convert2mm(sdye2[enb]) << "," << cms::convert2mm(sdze2[enb]) << ") rotation esalgo:RM1299";
#endif

              sdxe3[enb] = 0;
              sdye3[enb] = boxay / 2. + es.ldrBck_Length / 2.;
              sdze3[enb] = -es.ladder_thick / 2. + es.ldrBck_Offset;
              dd4hep::Solid solid =
                  dd4hep::UnionSolid(ddname,
                             solid_c,
                             solid_lbck,
                             dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdxe3[enb], sdye3[enb], sdze3[enb])));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeomX") << ddname << " Union " << solid_c.name() << ":" << solid_lbck.name() << " at (" << cms::convert2mm(sdxe3[enb]) << "," << cms::convert2mm(sdye3[enb]) << "," << cms::convert2mm(sdze3[enb]) << ") rotation esalgo:RM1299";
#endif

              ns.addVolumeNS(dd4hep::Volume(ddname, solid, ns.material(es.laddMaterial)));
              ns.addVolumeNS(dd4hep::Volume("esalgo:" + es.ladPfx[1] + es.typesL4[d], solid, ns.material(es.laddMaterial)));
            }
          }
        }  // end of not plain ladder shape
        else {
          const std::string& dd_tmp_name_pa("esalgo:" + es.ladPfx[2] + "p");
          const std::string& dd_tmp_name_pb("esalgo:" + es.ladPfx[3] + "p");

          boxay = ladder_length - es.ldrFrnt_Length - es.ldrBck_Length;
          boxax = es.ladder_width;
          boxaz = es.ladder_thick;

          dd4hep::Solid solid_pa = dd4hep::Box(dd_tmp_name_pa, boxax / 2., boxay / 2., boxaz / 2.);
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeomX") << dd_tmp_name_pa << " Box " << cms::convert2mm(boxax / 2) << ":" << cms::convert2mm(boxay / 2) << ":" << cms::convert2mm(boxaz / 2);
#endif
          sdx = 0;
          sdy = -boxay / 2. - es.ldrFrnt_Length / 2.;
          sdz = -es.ladder_thick / 2. + es.ldrFrnt_Offset;

          dd4hep::Solid solid_pb = dd4hep::UnionSolid(dd_tmp_name_pb,
                                      solid_pa,
                                      solid_lfront,
                                      dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdx, sdy, sdz)));
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeomX") << dd_tmp_name_pb << " Union " << solid_pa.name() << ":" << solid_lfront.name() << " at (" << cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << "," << cms::convert2mm(sdz) << ") rotation esalgo:RM1299";
#endif

          sdx = 0;
          sdy = boxay / 2. + es.ldrBck_Length / 2.;
          sdz = -es.ladder_thick / 2. + es.ldrBck_Offset;
          dd4hep::Solid solid = dd4hep::UnionSolid(
              ddname, solid_pb, solid_lbck, dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(sdx, sdy, sdz)));
#ifdef EDM_ML_DEBUG
	  edm::LogVerbatim("SFGeomX") << ddname << " Union " << solid_pb.name() << ":" << solid_lbck.name() << " at (" << cms::convert2mm(sdx) << "," << cms::convert2mm(sdy) << "," << cms::convert2mm(sdz) << ") rotation esalgo:RM1299";
#endif
          ns.addVolumeNS(dd4hep::Volume(ddname, solid, ns.material(es.laddMaterial)));
          ns.addVolumeNS(dd4hep::Volume("esalgo:" + es.ladPfx[1] + es.typesL4[d], solid, ns.material(es.laddMaterial)));
        }
      }

      // insert SWED, SFBX and SFBY into ladders
      swed_scopy_glob++;
      if (M < int(es.typesL5.size())) {
        const std::string& ddname("esalgo:" + es.ladPfx[0] + es.typesL5[M]);
        const std::string& ddname2("esalgo:" + es.ladPfx[1] + es.typesL5[M]);
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 4; j++) {
            xpos = (i * 2 - 1) * es.waf_intra_col_sep / 2.;
            ypos = -ladder_length / 2. + 0.05 * dd4hep::mm - (es.ldrFrnt_Length - es.ldrBck_Length) / 2. +
                   es.wedge_length / 2. + j * es.waf_active;
            zpos = -es.ladder_thick / 2. + 0.005 * dd4hep::mm + es.wedge_offset;
            if (es.laddL5map[(i + j * 2 + M * 10)] == 1) {
              scopy++;
              ns.volume(ddname).placeVolume(swedLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(xpos, ypos, zpos)));
              ns.volume(ddname2).placeVolume(swedLog,
                                             scopy + 1000 * swed_scopy_glob + 100,
                                             dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(xpos, ypos, zpos)));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeom") << swedLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob) << " in " << ddname << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1299";
	      edm::LogVerbatim("SFGeom") << swedLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob + 100) << " in " << ddname2 << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1299";
#endif

              ypos = ypos + es.ywedge_ceramic_diff;
              zpos = -es.ladder_thick / 2. + 0.005 * dd4hep::mm + es.zwedge_ceramic_diff;
              ns.volume(ddname).placeVolume(sfbxLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            dd4hep::Transform3D(ns.rotation("esalgo:RM1298"), dd4hep::Position(xpos, ypos, zpos)));
              ns.volume(ddname2).placeVolume(sfbyLog,
                                             scopy + 1000 * swed_scopy_glob,
                                             dd4hep::Transform3D(ns.rotation("esalgo:RM1300A"), dd4hep::Position(xpos, ypos, zpos)));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeom") << sfbxLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob) << " in " << ddname << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1298";
	      edm::LogVerbatim("SFGeom") << sfbyLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob) << " in " << ddname2 << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1300A";
#endif
            }
          }
        }
      } else {
        int d = M - es.typesL5.size();
        const std::string& ddname("esalgo:" + es.ladPfx[0] + es.typesL4[d]);
        const std::string& ddname2("esalgo:" + es.ladPfx[1] + es.typesL4[d]);
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            xpos = (i * 2 - 1) * es.waf_intra_col_sep / 2.;
            ypos = -ladder_length / 2. + 0.05 * dd4hep::mm - (es.ldrFrnt_Length - es.ldrBck_Length) / 2. +
                   es.wedge_length / 2. + j * es.waf_active;
            zpos = -es.ladder_thick / 2. + 0.005 * dd4hep::mm + es.wedge_offset;
            if (es.laddL4map[(i + j * 2 + (M - es.typesL5.size()) * 8)] == 1) {
              scopy++;
              ns.volume(ddname).placeVolume(swedLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(xpos, ypos, zpos)));
              ns.volume(ddname2).placeVolume(swedLog,
                                             scopy + 1000 * swed_scopy_glob + 100,
                                             dd4hep::Transform3D(ns.rotation("esalgo:RM1299"), dd4hep::Position(xpos, ypos, zpos)));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeom") << swedLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob) << " in " << ddname << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1299";
	      edm::LogVerbatim("SFGeom") << swedLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob + 100) << " in " << ddname2 << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1299";
#endif

              ypos = ypos + es.ywedge_ceramic_diff;
              zpos = -es.ladder_thick / 2. + 0.005 * dd4hep::mm + es.zwedge_ceramic_diff;
              ns.volume(ddname).placeVolume(sfbxLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            dd4hep::Transform3D(ns.rotation("esalgo:RM1298"), dd4hep::Position(xpos, ypos, zpos)));
              ns.volume(ddname2).placeVolume(sfbyLog,
                                             scopy + 1000 * swed_scopy_glob,
                                             dd4hep::Transform3D(ns.rotation("esalgo:RM1300A"), dd4hep::Position(xpos, ypos, zpos)));
#ifdef EDM_ML_DEBUG
	      edm::LogVerbatim("SFGeom") << sfbxLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob) << " in " << ddname << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1298";
	      edm::LogVerbatim("SFGeom") << sfbyLog.name() << " copy " << (scopy + 1000 * swed_scopy_glob) << " in " << ddname2 << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:RM1300A";
#endif
            }
          }
        }
      }
    }

    // Positioning of ladders
    int icopy[100] = {0};
    constexpr int sz = 20;

    for (int I = -9; I <= 9; ++I) {
      prev_length = 0;
      int J = abs(I);
      for (int K = 0; K < es.noLaddInCol[J]; K++) {
        std::string type;

        ladder_new_length = es.micromodule_length + 3. * es.waf_active;
        ladd_shift = 4. * es.waf_active;

        if (K == 0)
          type = es.typeOfLaddRow0[J];
        if (K == 1)
          type = es.typeOfLaddRow1[J];
        if (K == 2)
          type = es.typeOfLaddRow2[J];
        if (K == 3)
          type = es.typeOfLaddRow3[J];

        for (const auto& i : es.typesL5)
          if (type == i) {
            ladder_new_length = es.micromodule_length + 4. * es.waf_active;
            ladd_shift = 5. * es.waf_active;
          }

        int j = 0;

        for (int t = 0; t < int(es.typesL5.size()); t++)
          if (type == es.typesL5[t]) {
            j = t;
            if (I < 0 && es.asymLadd[t] == 1) {
              j = j + 1;
              type = es.typesL5[j];
            }
          }
        for (int t = 0; t < int(es.typesL4.size()); t++)
          if (type == es.typesL4[t]) {
            j = t + es.typesL5.size();
            if (I < 0 && es.asymLadd[(t + es.typesL5.size())] == 1) {
              j = j + 1;
              type = es.typesL4[j - es.typesL5.size()];
            }
          }

        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);
        if (I > 0)
          xpos = xpos + es.dee_separation;
        if (I < 0)
          xpos = xpos - es.dee_separation;

        ypos = (sz - int(es.startOfFirstLadd[J])) * es.waf_active - ladder_new_length / 2. +
               (es.ldrFrnt_Length - es.ldrBck_Length) / 2. + es.micromodule_length + 0.05 * dd4hep::cm - prev_length;

        prev_length += ladd_shift;

        zpos = es.zlead1 + es.ladder_thick / 2. + 0.01 * dd4hep::mm;
        icopy[j] += 1;

        sfLog.placeVolume(ns.volume("esalgo:" + es.ladPfx[0] + type), icopy[j], dd4hep::Position(xpos, ypos, zpos));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeom") << ("esalgo:" + es.ladPfx[0] + type) << " copy " << icopy[j] << " in " << sfLog.name() << " at (" << cms::convert2mm(xpos) << "," << cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") no rotation";
#endif

        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);
        sfLog.placeVolume(ns.volume("esalgo:" + es.ladPfx[1] + type),
                          icopy[j],
                          dd4hep::Transform3D(ns.rotation("esalgo:R270"), dd4hep::Position(ypos, -xpos, zpos - es.zlead1 + es.zlead2)));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeom") << ("esalgo:" + es.ladPfx[1] + type) << " copy " << icopy[j] << " in " << sfLog.name() << " at (" << cms::convert2mm(ypos) << "," << -cms::convert2mm(xpos) << "," << cms::convert2mm(zpos - es.zlead1 + es.zlead2) << ") rotation esalgo:R270";
#endif

        int changed = 0;
        for (int t = 0; t < int(es.typesL5.size()); t++)
          if (type == es.typesL5[t]) {
            j = t;
            if (es.asymLadd[t] == 2 && !changed) {
              j = j - 1;
              changed = 1;
            }
            if (es.asymLadd[t] == 1 && !changed) {
              j = j + 1;
              changed = 1;
            }
            type = es.typesL5[j];
          }
        for (int t = 0; t < int(es.typesL4.size()); t++)
          if (type == es.typesL4[t]) {
            j = t + es.typesL5.size();
            if (es.asymLadd[(t + es.typesL5.size())] == 2 && !changed) {
              j = j - 1;
              changed = 1;
            }
            if (es.asymLadd[(t + es.typesL5.size())] == 1 && !changed) {
              j = j + 1;
              changed = 1;
            }
            type = es.typesL4[j - es.typesL5.size()];
          }

        icopy[j] += 1;

        if (I > 0)
          xpos = xpos + es.dee_separation;
        if (I < 0)
          xpos = xpos - es.dee_separation;

        sfLog.placeVolume(ns.volume("esalgo:" + es.ladPfx[0] + type),
                          icopy[j],
                          dd4hep::Transform3D(ns.rotation("esalgo:R180"), dd4hep::Position(xpos, -ypos, zpos)));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeom") << ("esalgo:" + es.ladPfx[0] + type) << " copy " << icopy[j] << " in " << sfLog.name() << " at (" << cms::convert2mm(xpos) << "," << -cms::convert2mm(ypos) << "," << cms::convert2mm(zpos) << ") rotation esalgo:R180";
#endif

        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);

        sfLog.placeVolume(
            ns.volume("esalgo:" + es.ladPfx[1] + type),
            icopy[j],
            dd4hep::Transform3D(ns.rotation("esalgo:R090"), dd4hep::Position(-ypos, -xpos, zpos - es.zlead1 + es.zlead2)));
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("SFGeom") << ("esalgo:" + es.ladPfx[1] + type) << " copy " << icopy[j] << " in " << sfLog.name() << " at (" << -cms::convert2mm(ypos) << "," << -cms::convert2mm(xpos) << "," << cms::convert2mm(zpos - es.zlead1 + es.zlead2) << ") rotation esalgo:R090";
#endif
      }
    }
  }
  // place the slicon strips in active silicon wafers
  {
    double xpos(0), ypos(0);
    dd4hep::Volume sfwxLog = ns.volume("esalgo:SFWX");
    dd4hep::Volume sfwyLog = ns.volume("esalgo:SFWY");
    dd4hep::Volume sfsxLog = ns.volume("esalgo:SFSX");
    dd4hep::Volume sfsyLog = ns.volume("esalgo:SFSY");

    for (size_t i = 0; i < 32; ++i) {
      xpos = -es.waf_active / 2. + i * es.waf_active / 32. + es.waf_active / 64.;
      sfwxLog.placeVolume(sfsxLog, i + 1, dd4hep::Position(xpos, 0., 0.));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("SFGeom") << sfsxLog.name() << " copy " << (i + 1) << " in " << sfwxLog.name() << " at (" << cms::convert2mm(xpos) << ",0,0) no rotation";
#endif

      ypos = -es.waf_active / 2. + i * es.waf_active / 32. + es.waf_active / 64.;
      sfwyLog.placeVolume(sfsyLog, i + 1, dd4hep::Position(0., ypos, 0.));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("SFGeom") << sfsyLog.name() << " copy " << (i + 1) << " in " << sfwyLog.name() << " at (0," << cms::convert2mm(ypos) << ",0) no rotation";
#endif
    }
  }
  return 1;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_ecal_DDEcalPreshowerAlgo, algorithm)
