#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DetectorDescription/DDCMS/interface/BenchmarkGrd.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DD4hep/Shapes.h"

#include <string>
#include <vector>

using namespace std;
using namespace cms;
using namespace dd4hep;
using namespace cms_units::operators;

static constexpr float const& k_half = 0.5;
static constexpr float const& k_one32nd = 0.03125;
static constexpr float const& k_one64th = 0.015625;

namespace {

  struct EcalPreshower {
    vector<string> materials;  // materials of the presh-layers
    vector<string> layName;    // names of the presh-layers
    vector<string> ladPfx;     // name prefix for ladders
    vector<string> typesL5;
    vector<string> typesL4;
    vector<string> typeOfLaddRow0;
    vector<string> typeOfLaddRow1;
    vector<string> typeOfLaddRow2;
    vector<string> typeOfLaddRow3;

    vector<float> thickLayers;
    vector<float> abs1stx;
    vector<float> abs1sty;
    vector<float> abs2ndx;
    vector<float> abs2ndy;
    vector<float> asymLadd;
    vector<float> rminVec;
    vector<float> rmaxVec;
    vector<float> noLaddInCol;
    vector<float> startOfFirstLadd;
    vector<float> laddL5map;
    vector<float> laddL4map;
    string laddMaterial;  // ladd material - air
    float thickness;      // overall thickness of the preshower envelope

    float zlead1;
    float zlead2;
    float waf_intra_col_sep;
    float waf_inter_col_sep;
    float waf_active;
    float wedge_length;
    float wedge_offset;
    float zwedge_ceramic_diff;
    float ywedge_ceramic_diff;
    float wedge_angle;
    float box_thick;
    float dee_separation;
    float in_rad_Abs_Al;
    float in_rad_Abs_Pb;
    float ladder_thick;
    float ladder_width;
    float micromodule_length;
    float absAlX_X;
    float absAlX_Y;
    float absAlX_subtr1_Xshift;
    float absAlX_subtr1_Yshift;
    float rMax_Abs_Al;
    float absAlY_X;
    float absAlY_Y;
    float absAlY_subtr1_Xshift;
    float absAlY_subtr1_Yshift;
    float ldrBck_Length;
    float ldrFrnt_Length;
    float ldrFrnt_Offset;
    float ldrBck_Offset;
    float ceramic_length;
    float wedge_back_thick;
  };
}  // namespace

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  BenchmarkGrd counter("DDEcalPreshowerAlgo");
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  Volume parentVolume = ns.volume(args.parentName());
  Volume swedLog = ns.volume("esalgo:SWED");
  Volume sfLog = ns.volume("esalgo:SF");
  Volume sfbxLog = ns.volume("esalgo:SFBX");
  Volume sfbyLog = ns.volume("esalgo:SFBY");

  EcalPreshower es;
  es.asymLadd = args.vecFloat("ASYMETRIC_LADDER");
  es.typesL5 = args.vecStr("TYPES_OF_LADD_L5");
  es.typesL4 = args.vecStr("TYPES_OF_LADD_L4");
  es.laddL5map = args.vecFloat("LADD_L5_MAP");
  es.laddL4map = args.vecFloat("LADD_L4_MAP");
  es.noLaddInCol = args.vecFloat("NUMB_OF_LADD_IN_COL");
  es.startOfFirstLadd = args.vecFloat("START_OF_1ST_LADD");
  es.typeOfLaddRow0 = args.vecStr("TYPE_OF_LADD_1");
  es.typeOfLaddRow1 = args.vecStr("TYPE_OF_LADD_2");
  es.typeOfLaddRow2 = args.vecStr("TYPE_OF_LADD_3");
  es.typeOfLaddRow3 = args.vecStr("TYPE_OF_LADD_4");
  es.thickLayers = args.vecFloat("Layers");
  es.thickness = args.dble("PRESH_Z_TOTAL");
  es.materials = args.vecStr("LayMat");
  es.layName = args.vecStr("LayName");
  es.rmaxVec = args.vecFloat("R_MAX");  // inner radii
  es.rminVec = args.vecFloat("R_MIN");  // outer radii
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
  es.abs1stx = args.vecFloat("1ST_ABSX");
  es.abs1sty = args.vecFloat("1ST_ABSY");
  es.abs2ndx = args.vecFloat("2ND_ABSX");
  es.abs2ndy = args.vecFloat("2ND_ABSY");
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
    double zpos = -es.thickness * k_half, sdx(0), sdy(0), bdx(0), bdy(0);

    for (size_t i = 0; i < es.thickLayers.size(); ++i) {
      int I = int(i) + 1;  // FOTRAN I (offset +1)

      float rIn(0), rOut(0), zHalf(0);

      // create the name
      const string& ddname("esalgo:" + es.layName[i]);  // namespace:name

      // cone dimensions
      rIn = es.rminVec[i];
      rOut = es.rmaxVec[i];
      zHalf = es.thickLayers[i] * k_half;

      // create a logical part representing a single layer in the preshower
      Solid solid = ns.addSolid(ddname, Tube(ddname, rIn, rOut, zHalf, 0., 360._deg));
      Volume layer = ns.addVolume(Volume(ddname, solid, ns.material(es.materials[i])));

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

        const string& dd_tmp_name_b("esalgo:" + es.layName[i] + "Lcut");
        const string& dd_tmp_name_c("esalgo:" + es.layName[i] + "tmpb");
        const string& dd_tmp_name_d("esalgo:" + es.layName[i] + "LinPb");
        const string& dd_tmp_name_e("esalgo:" + es.layName[i] + "LinAl");
        const string& dd_tmp_name_f("esalgo:" + es.layName[i] + "LOutAl");

        const string& dd_Alname_f("esalgo:" + es.layName[i] + "LOutAl");
        const string& dd_Alname_g("esalgo:" + es.layName[i] + "LOutAl2");
        const string& dd_Alname_h("esalgo:" + es.layName[i] + "LOutAltmp");
        const string& dd_Alname_i("esalgo:" + es.layName[i] + "LOutAltmp2");
        const string& dd_Alname_j("esalgo:" + es.layName[i] + "LOutAltmp3");
        const string& dd_Alname_k("esalgo:" + es.layName[i] + "LOutAltmp4");
        const string& dd_Alname_l("esalgo:" + es.layName[i] + "LOutAltmp5");
        const string& dd_Alname_m("esalgo:" + es.layName[i] + "LOutAltmp6");

        Solid outAl =
            ns.addSolid(dd_Alname_f, Tube(dd_Alname_f, es.rMax_Abs_Al - 70_cm, es.rMax_Abs_Al, zHalf, 0., 90._deg));

        outalbx = es.absAlX_X * 0.1;
        outalby = es.rMax_Abs_Al + 0.1_mm - es.absAlX_subtr1_Yshift;
        shiftR = es.absAlX_subtr1_Yshift;
        if (I == 20) {
          outalbx = es.absAlY_X * 0.1;
          outalby = es.rMax_Abs_Al + 0.1_mm - es.absAlY_subtr1_Yshift;
          shiftR = es.absAlY_subtr1_Xshift;
        }
        Solid outAltmp = ns.addSolid(
            dd_Alname_h, Box(dd_Alname_h, outalbx * k_half + 0.1_mm, outalby * k_half + 0.1_mm, zHalf + 0.1_mm));
        Solid outAltmp3 = ns.addSolid(
            dd_Alname_j,
            SubtractionSolid(dd_Alname_j, outAl, outAltmp, Position(outalbx * k_half, outalby * k_half + shiftR, 0)));

        outalby2 = es.absAlX_Y * 0.1;
        outalbx2 = es.rMax_Abs_Al + 0.1_mm - es.absAlX_subtr1_Xshift;
        shiftR2 = es.absAlX_subtr1_Xshift;
        if (I == 20) {
          outalby2 = es.absAlY_Y * 0.1;
          outalbx2 = es.rMax_Abs_Al + 0.1_mm - es.absAlY_subtr1_Xshift;
          shiftR2 = es.absAlY_subtr1_Xshift;
        }
        Solid outAltmp2 = ns.addSolid(
            dd_Alname_i, Box(dd_Alname_i, outalbx2 * k_half + 0.1_mm, outalby2 * k_half + 0.1_mm, zHalf + 0.1_mm));
        Solid outAltmp4 = ns.addSolid(
            dd_Alname_k,
            SubtractionSolid(
                dd_Alname_k, outAltmp3, outAltmp2, Position(outalbx2 * k_half + shiftR2, outalby2 * k_half, 0)));
        Solid outAltmp5 =
            ns.addSolid(dd_Alname_l, UnionSolid(dd_Alname_l, outAltmp4, outAltmp4, ns.rotation("esalgo:RABS90")));
        Solid outAltmp6 =
            ns.addSolid(dd_Alname_m, UnionSolid(dd_Alname_m, outAltmp5, outAltmp4, ns.rotation("esalgo:RABS180B")));
        Solid outAl2 =
            ns.addSolid(dd_Alname_g, UnionSolid(dd_Alname_g, outAltmp6, outAltmp4, ns.rotation("esalgo:R180")));

        Solid outAlCut = Box(65_cm, 60_cm - 0.1_mm, zHalf + 0.2_mm);
        Solid outAlFin = SubtractionSolid(outAl2, outAlCut);

        Volume layerFinOutAl = Volume(dd_tmp_name_f, outAlFin, ns.material(es.materials[i - 1]));

        for (int L = 0; L < absz; ++L) {
          int K = L;
          ostringstream tmp_name_b, tmp_name_b2, tmp_FAl_name_c, tmp_FAl_name_d1, tmp_FAl_name_d2, tmp_FAl_name_d3,
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

          const string& dd_tmp_name_b("esalgo:" + tmp_name_b.str());
          const string& dd_tmp_name_b2("esalgo:" + tmp_name_b2.str());
          const string& dd_FAl_name_c("esalgo:" + tmp_FAl_name_c.str());
          const string& dd_FAl_name_d1("esalgo:" + tmp_FAl_name_d1.str());
          const string& dd_FAl_name_d2("esalgo:" + tmp_FAl_name_d2.str());
          const string& dd_FAl_name_d3("esalgo:" + tmp_FAl_name_d3.str());
          const string& dd_FAl_name_d("esalgo:" + tmp_FAl_name_d.str());

          if (L == 0)
            bdx = abs(es.abs1stx[K]) * k_half;
          if (L > 0)
            bdx = abs(es.abs1stx[K] - es.abs1stx[K - 1]) * k_half;
          bdy = es.abs1sty[K];
          if (es.abs1stx[K] < rIn + 30_cm) {
            bdy = es.abs1sty[K] * k_half - 30_cm;
            cutabsx = K;
          }

          if (I == 20) {
            if (L == 0)
              bdx = abs(es.abs2ndx[K]) * k_half;
            if (L > 0)
              bdx = abs(es.abs2ndx[K] - es.abs2ndx[K - 1]) * k_half;
            bdy = es.abs2ndy[K];
          }

          if ((es.abs2ndx[K] < rIn + 30_cm) && I == 20) {
            bdy = es.abs2ndy[K] * k_half - 30_cm;
            cutabsy = K;
          }

          Solid solid_b = Box(dd_tmp_name_b, bdx, bdy, zHalf);
          Solid solid_b2 = Box(dd_tmp_name_b2, bdx + 0.1_mm, bdy + 0.1_mm, zHalf);

          sdx = es.abs1stx[K] - bdx;
          sdy = 0;
          if (es.abs1stx[K] < rIn + 30_cm)
            sdy = es.abs1sty[K] - bdy;

          if (I == 20) {
            sdx = es.abs2ndx[K] - bdx;
            sdy = 0;
          }
          if ((es.abs2ndx[K] < rIn + 30_cm) && I == 20)
            sdy = es.abs2ndy[K] - bdy;

          Volume layer = Volume(dd_tmp_name_b, solid_b, ns.material(es.materials[i]));

          layerFinOutAl.placeVolume(layer, 1, Position(sdx, sdy, 0));
          layerFinOutAl.placeVolume(layer, 2, Position(-sdx, sdy, 0));

          Solid solid_c = ns.solid(dd_FAl_name_c);
          Solid solid_d1 =
              ns.addSolid(dd_FAl_name_d1, UnionSolid(dd_FAl_name_d1, solid_c, solid_b2, Position(sdx, sdy, 0)));
          Solid solid_d2 =
              ns.addSolid(dd_FAl_name_d, UnionSolid(dd_FAl_name_d, solid_d1, solid_b2, Position(-sdx, -sdy, 0)));

          if (((es.abs1stx[K] < rIn + 30_cm) && I == 10) || ((es.abs2ndx[K] < rIn + 30_cm) && I == 20)) {
            layerFinOutAl.placeVolume(layer, 3, Position(sdx, -sdy, 0));
            layerFinOutAl.placeVolume(layer, 4, Position(-sdx, -sdy, 0));

            Solid solid_c = ns.solid(dd_FAl_name_c);
            Solid solid_d1 = UnionSolid(dd_FAl_name_d1, solid_c, solid_b2, Position(sdx, sdy, 0));
            ns.addSolid(dd_FAl_name_d2, UnionSolid(dd_FAl_name_d2, solid_d1, solid_b2, Position(sdx, -sdy, 0)));
            Solid solid_d3 = UnionSolid(dd_FAl_name_d3, solid_d2, solid_b2, Position(-sdx, sdy, 0));
            ns.addSolid(dd_FAl_name_d, UnionSolid(dd_FAl_name_d, solid_d3, solid_b2, Position(-sdx, -sdy, 0)));
          }
        }

        bdx = es.abs1stx[cutabsx];
        if (I == 20)
          bdx = es.abs2ndx[cutabsy];
        bdy = 2 * 30_cm;

        Solid solidcut = Box(dd_tmp_name_b, bdx, bdy, zHalf);
        Solid iner = Tube(dd_tmp_name_c, 0, es.in_rad_Abs_Pb, zHalf + 0.1_mm, 0., 360._deg);
        Solid final = SubtractionSolid(dd_tmp_name_d, solidcut, iner);

        Volume blayer = Volume(dd_tmp_name_d, final, ns.material(es.materials[i]));
        parentVolume.placeVolume(blayer, 1, Position(0, 0, zpos));

        Solid iner_Al = Tube(dd_tmp_name_e, es.in_rad_Abs_Al, es.in_rad_Abs_Pb - 0.01_mm, zHalf, 0., 360._deg);
        Volume layerAl = Volume(dd_tmp_name_e, iner_Al, ns.material(es.materials[i - 1]));
        parentVolume.placeVolume(layerAl, 1, Position(0, 0, zpos));
        parentVolume.placeVolume(layerFinOutAl, 1, Position(0, 0, zpos));
      } else {
        parentVolume.placeVolume(layer, 1, Position(0., 0., zpos));
      }
      zpos += zHalf;
    }
  }
  // create and place the ladders
  {
    double xpos(0.), ypos(0.), zpos(0.);  //, sdx(0.), sdy(0.), sdz(0.);
    float prev_length(0.), ladder_new_length(0.);
    float ladd_shift(0.);
    float ladder_length(0.);
    int swed_scopy_glob(0);

    for (int M = 0; M < int(es.typesL5.size() + es.typesL4.size()); M++) {
      int scopy(0);
      int ladd_not_plain(0), ladd_subtr_no(0), ladd_upper(0);

      // Creation of ladders with 5 micromodules length

      if (M < int(es.typesL5.size())) {
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            if (es.laddL5map[(i + j * 2 + M * 10)] != 1) {
              ladd_not_plain = 1;
              ladd_subtr_no++;
              if (j > 1)
                ladd_upper = 1;
            }
          }
        }

        const string& ddname("esalgo:" + es.ladPfx[0] + es.typesL5[M]);
        ladder_length = es.micromodule_length + 4 * es.waf_active + 0.1_mm;

        if (ladd_not_plain) {
          if (!ladd_upper) {
            ns.addAssembly(ddname);
            ns.addAssembly("esalgo:" + es.ladPfx[1] + es.typesL5[M]);
          }
        }  // end of not plain ladder shape
        else {
          ns.addAssembly(ddname);
          ns.addAssembly("esalgo:" + es.ladPfx[1] + es.typesL5[M]);
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
            }
          }
        }

        const string& ddname("esalgo:" + es.ladPfx[0] + es.typesL4[d]);
        ladder_length = es.micromodule_length + 3 * es.waf_active + 0.1_mm;

        if (ladd_not_plain) {
          if (ladd_upper) {
            ns.addAssembly(ddname);
            ns.addAssembly("esalgo:" + es.ladPfx[1] + es.typesL4[d]);

          }  // upper
          else {
            if (ladd_subtr_no > 1) {
              ns.addAssembly(ddname);
              ns.addAssembly("esalgo:" + es.ladPfx[1] + es.typesL4[d]);
            } else {
              ns.addAssembly(ddname);
              ns.addAssembly("esalgo:" + es.ladPfx[1] + es.typesL4[d]);
            }
          }
        }  // end of not plain ladder shape
        else {
          ns.addAssembly(ddname);
          ns.addAssembly("esalgo:" + es.ladPfx[1] + es.typesL4[d]);
        }
      }

      // insert SWED, SFBX and SFBY into ladders
      swed_scopy_glob++;
      if (M < int(es.typesL5.size())) {
        const string& ddname("esalgo:" + es.ladPfx[0] + es.typesL5[M]);
        const string& ddname2("esalgo:" + es.ladPfx[1] + es.typesL5[M]);
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 4; j++) {
            xpos = (i * 2 - 1) * es.waf_intra_col_sep * k_half;
            ypos = -ladder_length * k_half + 0.05_mm - (es.ldrFrnt_Length - es.ldrBck_Length) * k_half +
                   es.wedge_length * k_half + j * es.waf_active;
            zpos = -es.ladder_thick * k_half + 0.005_mm + es.wedge_offset;
            if (es.laddL5map[(i + j * 2 + M * 10)] == 1) {
              scopy++;
              ns.assembly(ddname).placeVolume(swedLog,
                                              scopy + 1000 * swed_scopy_glob,
                                              Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));
              ns.assembly(ddname2).placeVolume(swedLog,
                                               scopy + 1000 * swed_scopy_glob + 100,
                                               Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));

              ypos = ypos + es.ywedge_ceramic_diff;
              zpos = -es.ladder_thick * k_half + 0.005_mm + es.zwedge_ceramic_diff;
              ns.assembly(ddname).placeVolume(sfbxLog,
                                              scopy + 1000 * swed_scopy_glob,
                                              Transform3D(ns.rotation("esalgo:RM1298"), Position(xpos, ypos, zpos)));
              ns.assembly(ddname2).placeVolume(sfbyLog,
                                               scopy + 1000 * swed_scopy_glob,
                                               Transform3D(ns.rotation("esalgo:RM1300A"), Position(xpos, ypos, zpos)));
            }
          }
        }
      } else {
        int d = M - es.typesL5.size();
        const string& ddname("esalgo:" + es.ladPfx[0] + es.typesL4[d]);
        const string& ddname2("esalgo:" + es.ladPfx[1] + es.typesL4[d]);
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            xpos = (i * 2 - 1) * es.waf_intra_col_sep * k_half;
            ypos = -ladder_length * k_half + 0.05_mm - (es.ldrFrnt_Length - es.ldrBck_Length) * k_half +
                   es.wedge_length * k_half + j * es.waf_active;
            zpos = -es.ladder_thick * k_half + 0.005_mm + es.wedge_offset;
            if (es.laddL4map[(i + j * 2 + (M - es.typesL5.size()) * 8)] == 1) {
              scopy++;
              ns.assembly(ddname).placeVolume(swedLog,
                                              scopy + 1000 * swed_scopy_glob,
                                              Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));
              ns.assembly(ddname2).placeVolume(swedLog,
                                               scopy + 1000 * swed_scopy_glob + 100,
                                               Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));

              ypos = ypos + es.ywedge_ceramic_diff;
              zpos = -es.ladder_thick * k_half + 0.005_mm + es.zwedge_ceramic_diff;
              ns.assembly(ddname).placeVolume(sfbxLog,
                                              scopy + 1000 * swed_scopy_glob,
                                              Transform3D(ns.rotation("esalgo:RM1298"), Position(xpos, ypos, zpos)));
              ns.assembly(ddname2).placeVolume(sfbyLog,
                                               scopy + 1000 * swed_scopy_glob,
                                               Transform3D(ns.rotation("esalgo:RM1300A"), Position(xpos, ypos, zpos)));
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
        string type;

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

        ypos = (sz - int(es.startOfFirstLadd[J])) * es.waf_active - ladder_new_length * k_half +
               (es.ldrFrnt_Length - es.ldrBck_Length) * k_half + es.micromodule_length + 0.05_cm - prev_length;

        prev_length += ladd_shift;

        zpos = es.zlead1 + es.ladder_thick * k_half + 0.01_mm;
        icopy[j] += 1;

        sfLog.placeVolume(ns.assembly("esalgo:" + es.ladPfx[0] + type), icopy[j], Position(xpos, ypos, zpos));

        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);

        sfLog.placeVolume(ns.assembly("esalgo:" + es.ladPfx[1] + type),
                          icopy[j],
                          Transform3D(ns.rotation("esalgo:R270"), Position(ypos, -xpos, zpos - es.zlead1 + es.zlead2)));

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

        sfLog.placeVolume(ns.assembly("esalgo:" + es.ladPfx[0] + type),
                          icopy[j],
                          Transform3D(ns.rotation("esalgo:R180"), Position(xpos, -ypos, zpos)));

        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);

        sfLog.placeVolume(
            ns.assembly("esalgo:" + es.ladPfx[1] + type),
            icopy[j],
            Transform3D(ns.rotation("esalgo:R090"), Position(-ypos, -xpos, zpos - es.zlead1 + es.zlead2)));
      }
    }
  }
  // place the slicon strips in active silicon wafers
  {
    float xpos(0), ypos(0);
    Volume sfwxLog = ns.volume("esalgo:SFWX");
    Volume sfwyLog = ns.volume("esalgo:SFWY");
    Volume sfsxLog = ns.volume("esalgo:SFSX");
    Volume sfsyLog = ns.volume("esalgo:SFSY");

    for (size_t i = 0; i < 32; ++i) {
      xpos = -es.waf_active * k_half + i * es.waf_active * k_one32nd + es.waf_active * k_one64th;
      sfwxLog.placeVolume(sfsxLog, i + 1, Position(xpos, 0., 0.));

      ypos = -es.waf_active * k_half + i * es.waf_active * k_one32nd + es.waf_active * k_one64th;
      sfwyLog.placeVolume(sfsyLog, i + 1, Position(0., ypos, 0.));
    }
  }
  return 1;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_ecal_DDEcalPreshowerAlgo, algorithm)
