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

namespace {
  struct EcalPreshower {
    double thickness_;          // overall thickness of the preshower envelope
    vector<string> materials_;  // materials of the presh-layers
    vector<string> layName_;    // names of the presh-layers
    vector<string> ladPfx_;     // name prefix for ladders
    string LaddMaterial_;       // ladd material - air
    vector<double> thickLayers_;
    vector<double> abs1stx;
    vector<double> abs1sty;
    vector<double> abs2ndx;
    vector<double> abs2ndy;
    vector<double> asym_ladd_;
    vector<double> rminVec;
    vector<double> rmaxVec;
    vector<double> noLaddInCol_;
    vector<double> startOfFirstLadd_;
    vector<string> types_l5_;
    vector<string> types_l4_;
    vector<double> ladd_l5_map_;
    vector<double> ladd_l4_map_;
    vector<string> typeOfLaddRow0;
    vector<string> typeOfLaddRow1;
    vector<string> typeOfLaddRow2;
    vector<string> typeOfLaddRow3;

    double zlead1_, zlead2_, zfoam1_, zfoam2_;
    double waf_intra_col_sep, waf_inter_col_sep, waf_active, wedge_length, wedge_offset, zwedge_ceramic_diff,
        ywedge_ceramic_diff, wedge_angle, box_thick, dee_separation, In_rad_Abs_Al, In_rad_Abs_Pb;
    double ladder_thick, yladder_1stwedge_diff, ladder_width, ladder_length, micromodule_length;
    double absAlX_X_, absAlX_Y_, absAlX_subtr1_Xshift_, absAlX_subtr1_Yshift_, rMax_Abs_Al_;
    double absAlY_X_, absAlY_Y_, absAlY_subtr1_Xshift_, absAlY_subtr1_Yshift_;
    double LdrBck_Length, LdrFrnt_Length, LdrFrnt_Offset, LdrBck_Offset, ceramic_length, wedge_back_thick;
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
  es.asym_ladd_ = args.vecDble("ASYMETRIC_LADDER");
  es.types_l5_ = args.vecStr("TYPES_OF_LADD_L5");
  es.types_l4_ = args.vecStr("TYPES_OF_LADD_L4");
  es.ladd_l5_map_ = args.vecDble("LADD_L5_MAP");
  es.ladd_l4_map_ = args.vecDble("LADD_L4_MAP");
  es.noLaddInCol_ = args.vecDble("NUMB_OF_LADD_IN_COL");
  es.startOfFirstLadd_ = args.vecDble("START_OF_1ST_LADD");
  es.typeOfLaddRow0 = args.vecStr("TYPE_OF_LADD_1");
  es.typeOfLaddRow1 = args.vecStr("TYPE_OF_LADD_2");
  es.typeOfLaddRow2 = args.vecStr("TYPE_OF_LADD_3");
  es.typeOfLaddRow3 = args.vecStr("TYPE_OF_LADD_4");
  es.thickLayers_ = args.vecDble("Layers");
  es.thickness_ = args.dble("PRESH_Z_TOTAL");
  es.materials_ = args.vecStr("LayMat");
  es.layName_ = args.vecStr("LayName");
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
  es.ladPfx_ = args.vecStr("LadPrefix");
  es.LaddMaterial_ = args.str("LadderMaterial");
  es.LdrFrnt_Length = args.dble("LdrFrnt_Length");
  es.LdrFrnt_Offset = args.dble("LdrFrnt_Offset");
  es.LdrBck_Length = args.dble("LdrBck_Length");
  es.LdrBck_Offset = args.dble("LdrBck_Offset");
  es.dee_separation = args.dble("dee_sep");
  es.In_rad_Abs_Al = args.dble("R_MIN_Abs_Al");
  es.In_rad_Abs_Pb = args.dble("R_MIN_Abs_Pb");
  es.rMax_Abs_Al_ = args.dble("R_MAX_Abs_Al");
  es.absAlX_X_ = args.dble("AbsAlX_X");
  es.absAlX_Y_ = args.dble("AbsAlX_Y");
  es.absAlX_subtr1_Xshift_ = args.dble("AbsAlX_subtr1_Xshift");
  es.absAlX_subtr1_Yshift_ = args.dble("AbsAlX_subtr1_Yshift");
  es.absAlY_X_ = args.dble("AbsAlY_X");
  es.absAlY_Y_ = args.dble("AbsAlY_Y");
  es.absAlY_subtr1_Xshift_ = args.dble("AbsAlY_subtr1_Xshift");
  es.absAlY_subtr1_Yshift_ = args.dble("AbsAlY_subtr1_Yshift");

  // create all the tube-like layers of the preshower
  {
    double zpos = -es.thickness_ / 2., sdx(0), sdy(0), bdx(0), bdy(0);

    for (size_t i = 0; i < es.thickLayers_.size(); ++i) {
      int I = int(i) + 1;  // FOTRAN I (offset +1)

      double rIn(0), rOut(0), zHalf(0);

      // create the name
      string ddname(es.layName_[i]);  // namespace:name

      // cone dimensions
      rIn = es.rminVec[i];
      rOut = es.rmaxVec[i];
      zHalf = es.thickLayers_[i] / 2.;

      // create a logical part representing a single layer in the preshower
      Solid solid = ns.addSolid(ddname, Tube(ddname, rIn, rOut, zHalf, 0., 360._deg));
      Volume layer = ns.addVolume(Volume(ddname, solid, ns.material(es.materials_[i])));

      // position the logical part w.r.t. the parent volume
      zpos += zHalf;

      // create a logical part representing a single layer in the preshower
      // skip layers with detectors, front and rear window
      if (I == 2 || I == 28 || I == 13 || I == 23) {
        zpos += zHalf;
        continue;
      }

      if (I == 12) {
        es.zlead1_ = zpos + zHalf;
      }
      if (I == 22) {
        es.zlead2_ = zpos + zHalf;
      }

      if (I == 10 || I == 20) {  // New lead shape

        int absz = 0;
        double outalbx, outalby, shiftR, outalbx2, outalby2, shiftR2;

        absz = int(es.abs1stx.size());
        if (I == 20)
          absz = int(es.abs2ndx.size());
        int cutabsx = -1, cutabsy = -1;

        string dd_tmp_name_b("esalgo:" + es.layName_[i] + "Lcut");
        string dd_tmp_name_c("esalgo:" + es.layName_[i] + "tmpb");
        string dd_tmp_name_d("esalgo:" + es.layName_[i] + "LinPb");

        string dd_tmp_name_e("esalgo:" + es.layName_[i] + "LinAl");
        string dd_tmp_name_f("esalgo:" + es.layName_[i] + "LOutAl");

        ostringstream tmp_Alname_fin;
        tmp_Alname_fin << es.layName_[i] << "LtmpAl" << absz - 1;
        string dd_Alname_fin("esalgo:" + tmp_Alname_fin.str());

        string dd_Alname_f(es.layName_[i] + "LOutAl");
        string dd_Alname_g(es.layName_[i] + "LOutAl2");
        string dd_Alname_h(es.layName_[i] + "LOutAltmp");
        string dd_Alname_i(es.layName_[i] + "LOutAltmp2");
        string dd_Alname_j(es.layName_[i] + "LOutAltmp3");
        string dd_Alname_k(es.layName_[i] + "LOutAltmp4");
        string dd_Alname_l(es.layName_[i] + "LOutAltmp5");
        string dd_Alname_m(es.layName_[i] + "LOutAltmp6");

        Solid Out_Al = ns.addSolid(
            dd_Alname_f, Tube(dd_Alname_f, es.rMax_Abs_Al_ - 20_cm, es.rMax_Abs_Al_, zHalf - 0.1_mm, 0., 90._deg));

        outalbx = es.absAlX_X_ * 0.1;
        outalby = es.rMax_Abs_Al_ + 0.1_mm - es.absAlX_subtr1_Yshift_;
        shiftR = es.absAlX_subtr1_Yshift_;
        if (I == 20) {
          outalbx = es.absAlY_X_ * 0.1;
          outalby = es.rMax_Abs_Al_ + 0.1_mm - es.absAlY_subtr1_Yshift_;
          shiftR = es.absAlY_subtr1_Xshift_;
        }
        Solid OutAltmp = ns.addSolid(dd_Alname_h, Box(dd_Alname_h, outalbx / 2 + 0.1_mm, outalby / 2 + 0.1_mm, zHalf));
        Solid Out_Altmp3 = ns.addSolid(
            dd_Alname_j,
            SubtractionSolid(dd_Alname_j, Out_Al, OutAltmp, Position(outalbx / 2, outalby / 2 + shiftR, 0)));
        outalby2 = es.absAlX_Y_ * 0.1;
        outalbx2 = es.rMax_Abs_Al_ + 0.1_mm - es.absAlX_subtr1_Xshift_;
        shiftR2 = es.absAlX_subtr1_Xshift_;
        if (I == 20) {
          outalby2 = es.absAlY_Y_ * 0.1;
          outalbx2 = es.rMax_Abs_Al_ + 0.1_mm - es.absAlY_subtr1_Xshift_;
          shiftR2 = es.absAlY_subtr1_Xshift_;
        }
        Solid OutAltmp2 =
            ns.addSolid(dd_Alname_i, Box(dd_Alname_i, outalbx2 / 2 + 0.1_mm, outalby2 / 2 + 0.1_mm, zHalf));
        Solid Out_Altmp4 = ns.addSolid(
            dd_Alname_k,
            SubtractionSolid(dd_Alname_k, Out_Altmp3, OutAltmp2, Position(outalbx2 / 2 + shiftR2, outalby2 / 2, 0)));
        Solid Out_Altmp5 =
            ns.addSolid(dd_Alname_l, UnionSolid(dd_Alname_l, Out_Altmp4, Out_Altmp4, ns.rotation("esalgo:RABS90")));
        Solid Out_Altmp6 =
            ns.addSolid(dd_Alname_m, UnionSolid(dd_Alname_m, Out_Altmp5, Out_Altmp4, ns.rotation("esalgo:RABS180B")));
        Solid Out_Al2 =  //FIXME
            ns.addSolid(dd_Alname_g, UnionSolid(dd_Alname_g, Out_Altmp6, Out_Altmp4, ns.rotation("esalgo:R180")));

        // FIXME Solid Outer_Al = ns.solid(dd_Alname_fin);
        Volume layerFinOutAl = Volume(dd_tmp_name_f, /* FIXME: Outer_Al */ Out_Al2, ns.material(es.materials_[i - 1]));

        for (int L = 0; L < absz; ++L) {
          int K = L;
          ostringstream tmp_name_b, tmp_name_b2, tmp_FAl_name_c, tmp_FAl_name_d1, tmp_FAl_name_d2, tmp_FAl_name_d3,
              tmp_FAl_name_d;
          tmp_name_b << es.layName_[i] << "L" << K;
          tmp_name_b2 << es.layName_[i] << "Lb2" << K;

          if (L == 0)
            tmp_FAl_name_c << es.layName_[i] << "LOutAl2";
          if (L > 0)
            tmp_FAl_name_c << es.layName_[i] << "LtmpAl" << K - 1;

          tmp_FAl_name_d1 << es.layName_[i] << "LtmpAl" << K << "_1";
          tmp_FAl_name_d2 << es.layName_[i] << "LtmpAl" << K << "_2";
          tmp_FAl_name_d3 << es.layName_[i] << "LtmpAl" << K << "_3";
          tmp_FAl_name_d << es.layName_[i] << "LtmpAl" << K;

          string dd_tmp_name_b(tmp_name_b.str());
          string dd_tmp_name_b2(tmp_name_b2.str());
          string dd_FAl_name_c(tmp_FAl_name_c.str());
          string dd_FAl_name_d1(tmp_FAl_name_d1.str());
          string dd_FAl_name_d2(tmp_FAl_name_d2.str());
          string dd_FAl_name_d3(tmp_FAl_name_d3.str());
          string dd_FAl_name_d(tmp_FAl_name_d.str());

          if (L == 0)
            bdx = abs(es.abs1stx[K]) / 2;
          if (L > 0)
            bdx = abs(es.abs1stx[K] - es.abs1stx[K - 1]) / 2;
          bdy = es.abs1sty[K];
          if (es.abs1stx[K] < rIn + 30_cm) {
            bdy = es.abs1sty[K] / 2 - 30_cm;
            cutabsx = K;
          }

          if (I == 20) {
            if (L == 0)
              bdx = abs(es.abs2ndx[K]) / 2;
            if (L > 0)
              bdx = abs(es.abs2ndx[K] - es.abs2ndx[K - 1]) / 2;
            bdy = es.abs2ndy[K];
          }

          if ((es.abs2ndx[K] < rIn + 30_cm) && I == 20) {
            bdy = es.abs2ndy[K] / 2 - 30_cm;
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

          Volume layer = Volume(dd_tmp_name_b, solid_b, ns.material(es.materials_[i]));

          layerFinOutAl.placeVolume(layer, 1, Position(sdx, sdy, 0));
          layerFinOutAl.placeVolume(layer, 2, Position(-sdx, sdy, 0));

          Solid solid_c = ns.solid(dd_FAl_name_c);
          Solid solid_d1 = UnionSolid(dd_FAl_name_d1, solid_c, solid_b2, Position(sdx, sdy, 0));
          Solid solid_d2 =
              ns.addSolid(dd_FAl_name_d, UnionSolid(dd_FAl_name_d, solid_d1, solid_b2, Position(-sdx, -sdy, 0)));

          if (((es.abs1stx[K] < rIn + 30_cm) && I == 10) || ((es.abs2ndx[K] < rIn + 30_cm) && I == 20)) {
            layerFinOutAl.placeVolume(layer, 3, Position(sdx, -sdy, 0));
            layerFinOutAl.placeVolume(layer, 4, Position(-sdx, -sdy, 0));

            Solid solid_c = ns.solid(dd_FAl_name_c);
            Solid solid_d1 = UnionSolid(dd_FAl_name_d1, solid_c, solid_b2, Position(sdx, sdy, 0));
            /*Solid solid_d2 = */
            ns.addSolid(dd_FAl_name_d2, UnionSolid(dd_FAl_name_d2, solid_d1, solid_b2, Position(sdx, -sdy, 0)));
            Solid solid_d3 = UnionSolid(dd_FAl_name_d3, solid_d2, solid_b2, Position(-sdx, sdy, 0));
            /*Solid solid_d4 = */
            ns.addSolid(dd_FAl_name_d, UnionSolid(dd_FAl_name_d, solid_d3, solid_b2, Position(-sdx, -sdy, 0)));
          }
        }

        bdx = es.abs1stx[cutabsx];
        if (I == 20)
          bdx = es.abs2ndx[cutabsy];
        bdy = 2 * 30_cm;

        Solid solidcut = Box(dd_tmp_name_b, bdx, bdy, zHalf);
        Solid iner = Tube(dd_tmp_name_c, 0, es.In_rad_Abs_Pb, zHalf + 0.1_mm, 0., 360._deg);
        Solid final = SubtractionSolid(dd_tmp_name_d, solidcut, iner);

        Volume layer = Volume(dd_tmp_name_d, final, ns.material(es.materials_[i]));
        parentVolume.placeVolume(layer, 1, Position(0, 0, zpos));

        Solid iner_Al = Tube(dd_tmp_name_e, es.In_rad_Abs_Al, es.In_rad_Abs_Pb - 0.01_mm, zHalf, 0., 360._deg);
        Volume layerAl = Volume(dd_tmp_name_e, iner_Al, ns.material(es.materials_[i - 1]));
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
    double xpos(0), ypos(0), zpos(0), sdx(0), sdy(0), sdz(0);
    double prev_length_(0), ladder_new_length_(0), ladd_shift_(0), ladder_length(0);
    int enb(0), swed_scopy_glob(0);
    double sdxe[50] = {0}, sdye[50] = {0}, sdze[50] = {0};
    double sdxe2[50] = {0}, sdye2[50] = {0}, sdze2[50] = {0}, sdxe3[50] = {0}, sdye3[50] = {0}, sdze3[50] = {0};

    for (int M = 0; M < int(es.types_l5_.size() + es.types_l4_.size()); M++) {
      int scopy(0);
      double boxax(0), boxay(0), boxaz(0);
      int ladd_not_plain(0), ladd_subtr_no(0), ladd_upper(0), ladd_side(0);

      Solid solid_lfront = Trap("esalgo:LDRFRNT",
                                es.LdrFrnt_Length / 2,                                                // pDz
                                -es.wedge_angle,                                                      // pTheta
                                0,                                                                    // pPhi
                                es.ladder_width / 2,                                                  // pDy1
                                (es.ladder_thick) / 2,                                                // pDx1
                                (es.ladder_thick) / 2,                                                //     pDx2
                                0,                                                                    //pAlp1
                                es.ladder_width / 2,                                                  //pDy2
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2)) / 2,  // pDx3
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2)) / 2,  // pDx4
                                0);

      Solid solid_lbck = Trap("esalgo:LDRBCK",
                              es.LdrBck_Length / 2,                                    // pDz
                              -es.wedge_angle,                                         // pTheta
                              0,                                                       // pPhi
                              es.ladder_width / 2,                                     // pDy1
                              (es.box_thick / cos(es.wedge_angle * 2) + 0.02_mm) / 2,  // pDx1
                              (es.box_thick / cos(es.wedge_angle * 2) + 0.02_mm) / 2,  //     pDx2
                              0,                                                       //pAlp1
                              es.ladder_width / 2,                                     //pDy2
                              (es.ladder_thick - es.wedge_back_thick) / 2,             // pDx3
                              (es.ladder_thick - es.wedge_back_thick) / 2,             // pDx4
                              0);

      Solid solid_lfhalf = Trap("esalgo:LDRFHALF",
                                es.LdrFrnt_Length / 2,                                                // pDz
                                -es.wedge_angle,                                                      // pTheta
                                0,                                                                    // pPhi
                                (es.ladder_width / 2) / 2,                                            // pDy1
                                (es.ladder_thick) / 2,                                                // pDx1
                                (es.ladder_thick) / 2,                                                //     pDx2
                                0,                                                                    //pAlp1
                                (es.ladder_width / 2) / 2,                                            //pDy2
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2)) / 2,  // pDx3
                                (es.ladder_thick - es.ceramic_length * sin(es.wedge_angle * 2)) / 2,  // pDx4
                                0);

      Solid solid_lbhalf = Trap("esalgo:LDRBHALF",
                                es.LdrBck_Length / 2,                                    // pDz
                                -es.wedge_angle,                                         // pTheta
                                0,                                                       // pPhi
                                (es.ladder_width / 2) / 2,                               // pDy1
                                (es.box_thick / cos(es.wedge_angle * 2) + 0.02_mm) / 2,  // pDx1
                                (es.box_thick / cos(es.wedge_angle * 2) + 0.02_mm) / 2,  //     pDx2
                                0,                                                       //pAlp1
                                (es.ladder_width / 2) / 2,                               //pDy2
                                (es.ladder_thick - es.wedge_back_thick) / 2,             // pDx3
                                (es.ladder_thick - es.wedge_back_thick) / 2,             // pDx4
                                0);

      Solid solid_lfhtrunc =
          Trap("esalgo:LDRFHTR",
               (es.LdrFrnt_Length - es.waf_active) / 2,                                                // pDz
               -es.wedge_angle,                                                                        // pTheta
               0,                                                                                      // pPhi
               (es.ladder_width / 2) / 2,                                                              // pDy1
               (es.ladder_thick) / 2,                                                                  // pDx1
               (es.ladder_thick) / 2,                                                                  //     pDx2
               0,                                                                                      //pAlp1
               (es.ladder_width / 2) / 2,                                                              //pDy2
               (es.ladder_thick - (es.ceramic_length - es.waf_active) * sin(es.wedge_angle * 2)) / 2,  // pDx3
               (es.ladder_thick - (es.ceramic_length - es.waf_active) * sin(es.wedge_angle * 2)) / 2,  // pDx4
               0);

      // Creation of ladders with 5 micromodules length

      if (M < int(es.types_l5_.size())) {
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            if (es.ladd_l5_map_[(i + j * 2 + M * 10)] != 1) {
              ladd_not_plain = 1;
              ladd_subtr_no++;
              if (j > 1)
                ladd_upper = 1;
              ladd_side = i;
            }
          }
        }

        string ddname("esalgo:" + es.ladPfx_[0] + es.types_l5_[M]);
        ladder_length = es.micromodule_length + 4 * es.waf_active + 0.1_mm;

        if (ladd_not_plain) {
          //        enb++;
          ostringstream tmp_name_5b, tmp_name_5c, tmp_name_5d;
          if (ladd_upper) {
          }  //upper
          else {
            enb++;
            ostringstream tmp_name_5b, tmp_name_5c, tmp_name_5d;
            string dd_tmp_name_5a("esalgo:" + es.ladPfx_[2]);
            tmp_name_5b << es.ladPfx_[3] << enb;
            string dd_tmp_name_5b("esalgo:" + tmp_name_5b.str());
            tmp_name_5c << es.ladPfx_[4] << enb;
            string dd_tmp_name_5c("esalgo:" + tmp_name_5c.str());
            tmp_name_5d << es.ladPfx_[5] << enb;
            string dd_tmp_name_5d("esalgo:" + tmp_name_5d.str());

            string dd_tmp_name_5e("esalgo:" + es.ladPfx_[6]);

            boxay = ladder_length - es.LdrFrnt_Length - es.LdrBck_Length;
            boxax = es.ladder_width;
            boxaz = es.ladder_thick;

            Solid solid_5a = Box(dd_tmp_name_5a, boxax / 2, boxay / 2, boxaz / 2.);
            if (ladd_side == 0)
              sdxe[enb] = es.ladder_width / 4;
            sdye[enb] = -boxay / 2 - es.LdrFrnt_Length / 2;
            sdze[enb] = -es.ladder_thick / 2. + es.LdrFrnt_Offset;
            if (ladd_side == 1)
              sdxe[enb] = -es.ladder_width / 4;

            Solid solid_5b =
                UnionSolid(dd_tmp_name_5b,
                           solid_5a,
                           solid_lfhalf,
                           Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe[enb], sdye[enb], sdze[enb])));

            if (ladd_side == 0)
              sdxe2[enb] = -es.ladder_width / 4;
            sdye2[enb] = -boxay / 2 - es.LdrFrnt_Length / 2 + es.waf_active / 2;
            sdze2[enb] = -es.ladder_thick / 2. + es.LdrFrnt_Offset + (es.waf_active * sin(es.wedge_angle * 2)) / 4;
            if (ladd_side == 1)
              sdxe2[enb] = es.ladder_width / 4;

            Solid solid_5c =
                UnionSolid(dd_tmp_name_5c,
                           solid_5b,
                           solid_lfhtrunc,
                           Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe2[enb], sdye2[enb], sdze2[enb])));

            sdxe3[enb] = 0;
            sdye3[enb] = boxay / 2 + es.LdrBck_Length / 2;
            sdze3[enb] = -es.ladder_thick / 2. + es.LdrBck_Offset;
            Solid solid =
                UnionSolid(ddname,
                           solid_5c,
                           solid_lbck,
                           Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe3[enb], sdye3[enb], sdze3[enb])));

            /*Volume ladder = */
            ns.addVolumeNS(Volume(ddname, solid, ns.material(es.LaddMaterial_)));
            string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l5_[M]);
            /* Volume ladder2 = */
            ns.addVolumeNS(Volume(ddname2, solid, ns.material(es.LaddMaterial_)));
          }
        }  //end of not plain ladder shape
        else {
          string dd_tmp_name_5pa("esalgo:" + es.ladPfx_[2] + "5p");
          string dd_tmp_name_5pb("esalgo:" + es.ladPfx_[3] + "5p");

          boxay = ladder_length - es.LdrFrnt_Length - es.LdrBck_Length;
          boxax = es.ladder_width;
          boxaz = es.ladder_thick;

          Solid solid_5pa = Box(dd_tmp_name_5pa, boxax / 2, boxay / 2, boxaz / 2.);
          sdx = 0;
          sdy = -boxay / 2 - es.LdrFrnt_Length / 2;
          sdz = -es.ladder_thick / 2. + es.LdrFrnt_Offset;

          Solid solid_5pb = UnionSolid(dd_tmp_name_5pb,
                                       solid_5pa,
                                       solid_lfront,
                                       Transform3D(ns.rotation("esalgo:RM1299"), Position(sdx, sdy, sdz)));

          sdx = 0;
          sdy = boxay / 2 + es.LdrBck_Length / 2;
          sdz = -es.ladder_thick / 2. + es.LdrBck_Offset;

          Solid solid = UnionSolid(
              ddname, solid_5pb, solid_lbck, Transform3D(ns.rotation("esalgo:RM1299"), Position(sdx, sdy, sdz)));
          /* Volume ladder = */
          ns.addVolumeNS(Volume(ddname, solid, ns.material(es.LaddMaterial_)));
          string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l5_[M]);
          /* Volume ladder2 = */
          ns.addVolumeNS(Volume(ddname2, solid, ns.material(es.LaddMaterial_)));
        }
      }

      // Creation of ladders with 4 micromodules length

      if (M >= int(es.types_l5_.size())) {
        int d = M - es.types_l5_.size();

        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            if (es.ladd_l4_map_[(i + j * 2 + (M - es.types_l5_.size()) * 8)] != 1) {
              ladd_not_plain = 1;
              ladd_subtr_no++;
              if (j > 1)
                ladd_upper = 1;
              ladd_side = i;
            }
          }
        }

        string ddname("esalgo:" + es.ladPfx_[0] + es.types_l4_[d]);
        ladder_length = es.micromodule_length + 3 * es.waf_active + 0.1_mm;

        if (ladd_not_plain) {
          ostringstream tmp_name_b, tmp_name_c, tmp_name_d;
          if (ladd_upper) {
            enb++;

            string dd_tmp_name_a("esalgo:" + es.ladPfx_[7]);
            tmp_name_b << es.ladPfx_[8] << enb;
            string dd_tmp_name_b("esalgo:" + tmp_name_b.str());
            tmp_name_c << es.ladPfx_[9] << enb;
            string dd_tmp_name_c("esalgo:" + tmp_name_c.str());
            tmp_name_d << es.ladPfx_[10] << enb;
            string dd_tmp_name_d("esalgo:" + tmp_name_d.str());
            string dd_tmp_name_e("esalgo:" + es.ladPfx_[11]);

            boxay = ladder_length - es.LdrFrnt_Length - es.LdrBck_Length;
            boxax = es.ladder_width;
            boxaz = es.ladder_thick;
            Solid solid_a = Box(dd_tmp_name_a, boxax / 2, boxay / 2, boxaz / 2.);

            sdxe[enb] = 0;
            sdye[enb] = -boxay / 2 - es.LdrFrnt_Length / 2;
            sdze[enb] = -es.ladder_thick / 2. + es.LdrFrnt_Offset;
            Solid solid_b =
                UnionSolid(dd_tmp_name_b,
                           solid_a,
                           solid_lfront,
                           Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe[enb], sdye[enb], sdze[enb])));

            if (ladd_side == 0)
              sdxe2[enb] = es.ladder_width / 4;
            sdye2[enb] = boxay / 2 + es.LdrBck_Length / 2;
            sdze2[enb] = -es.ladder_thick / 2. + es.LdrBck_Offset;
            if (ladd_side == 1)
              sdxe2[enb] = -es.ladder_width / 4;
            Solid solid =
                UnionSolid(ddname,
                           solid_b,
                           solid_lbhalf,
                           Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe2[enb], sdye2[enb], sdze2[enb])));

            /* Volume ladder = */
            ns.addVolumeNS(Volume(ddname, solid, ns.material(es.LaddMaterial_)));
            string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l4_[d]);
            /* Volume ladder2 = */
            ns.addVolumeNS(Volume(ddname2, solid, ns.material(es.LaddMaterial_)));

          }  //upper
          else {
            if (ladd_subtr_no > 1) {
              enb++;

              string dd_tmp_name_a("esalgo:" + es.ladPfx_[7]);
              tmp_name_b << es.ladPfx_[8] << enb;
              string dd_tmp_name_b("esalgo:" + tmp_name_b.str());
              tmp_name_c << es.ladPfx_[9] << enb;
              string dd_tmp_name_c("esalgo:" + tmp_name_c.str());
              tmp_name_d << es.ladPfx_[10] << enb;
              string dd_tmp_name_d("esalgo:" + tmp_name_d.str());
              string dd_tmp_name_e("esalgo:" + es.ladPfx_[11]);

              boxay = ladder_length - es.LdrFrnt_Length - es.LdrBck_Length;
              boxax = es.ladder_width;
              boxaz = es.ladder_thick;

              Solid solid_a = Box(dd_tmp_name_a, boxax / 2, boxay / 2, boxaz / 2.);
              if (ladd_side == 0)
                sdxe[enb] = es.ladder_width / 4;
              sdye[enb] = -boxay / 2 - es.LdrFrnt_Length / 2;
              sdze[enb] = -es.ladder_thick / 2. + es.LdrFrnt_Offset;
              if (ladd_side == 1)
                sdxe[enb] = -es.ladder_width / 4;

              Solid solid_b =
                  UnionSolid(dd_tmp_name_b,
                             solid_a,
                             solid_lfhalf,
                             Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe[enb], sdye[enb], sdze[enb])));

              sdxe2[enb] = 0;
              sdye2[enb] = boxay / 2 + es.LdrBck_Length / 2;
              sdze2[enb] = -es.ladder_thick / 2. + es.LdrBck_Offset;

              Solid solid =
                  UnionSolid(ddname,
                             solid_b,
                             solid_lbck,
                             Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe2[enb], sdye2[enb], sdze2[enb])));

              /* Volume ladder = */
              ns.addVolumeNS(Volume(ddname, solid, ns.material(es.LaddMaterial_)));
              string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l4_[d]);
              /* Volume ladder2 = */
              ns.addVolumeNS(Volume(ddname2, solid, ns.material(es.LaddMaterial_)));
            } else {
              enb++;
              string dd_tmp_name_a("esalgo:" + es.ladPfx_[7]);
              tmp_name_b << es.ladPfx_[8] << enb;
              string dd_tmp_name_b("esalgo:" + tmp_name_b.str());
              tmp_name_c << es.ladPfx_[9] << enb;
              string dd_tmp_name_c("esalgo:" + tmp_name_c.str());
              tmp_name_d << es.ladPfx_[10] << enb;
              string dd_tmp_name_d("esalgo:" + tmp_name_d.str());
              string dd_tmp_name_e("esalgo:" + es.ladPfx_[11]);

              boxay = ladder_length - es.LdrFrnt_Length - es.LdrBck_Length;
              boxax = es.ladder_width;
              boxaz = es.ladder_thick;
              Solid solid_a = Box(dd_tmp_name_a, boxax / 2, boxay / 2, boxaz / 2.);
              if (ladd_side == 0)
                sdxe[enb] = es.ladder_width / 4;
              sdye[enb] = -boxay / 2 - es.LdrFrnt_Length / 2;
              sdze[enb] = -es.ladder_thick / 2. + es.LdrFrnt_Offset;
              if (ladd_side == 1)
                sdxe[enb] = -es.ladder_width / 4;

              Solid solid_b =
                  UnionSolid(dd_tmp_name_b,
                             solid_a,
                             solid_lfhalf,
                             Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe[enb], sdye[enb], sdze[enb])));

              if (ladd_side == 0)
                sdxe2[enb] = -es.ladder_width / 4;
              sdye2[enb] = -boxay / 2 - es.LdrFrnt_Length / 2 + es.waf_active / 2;
              sdze2[enb] = -es.ladder_thick / 2. + es.LdrFrnt_Offset + (es.waf_active * sin(es.wedge_angle * 2)) / 4;
              if (ladd_side == 1)
                sdxe2[enb] = es.ladder_width / 4;

              Solid solid_c =
                  UnionSolid(dd_tmp_name_c,
                             solid_b,
                             solid_lfhtrunc,
                             Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe2[enb], sdye2[enb], sdze2[enb])));

              sdxe3[enb] = 0;
              sdye3[enb] = boxay / 2 + es.LdrBck_Length / 2;
              sdze3[enb] = -es.ladder_thick / 2. + es.LdrBck_Offset;
              Solid solid =
                  UnionSolid(ddname,
                             solid_c,
                             solid_lbck,
                             Transform3D(ns.rotation("esalgo:RM1299"), Position(sdxe3[enb], sdye3[enb], sdze3[enb])));

              /* Volume ladder = */
              ns.addVolumeNS(Volume(ddname, solid, ns.material(es.LaddMaterial_)));
              string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l4_[d]);
              /* Volume ladder2 = */
              ns.addVolumeNS(Volume(ddname2, solid, ns.material(es.LaddMaterial_)));
            }
          }
        }  //end of not plain ladder shape
        else {
          string dd_tmp_name_pa("esalgo:" + es.ladPfx_[2] + "p");
          string dd_tmp_name_pb("esalgo:" + es.ladPfx_[3] + "p");

          boxay = ladder_length - es.LdrFrnt_Length - es.LdrBck_Length;
          boxax = es.ladder_width;
          boxaz = es.ladder_thick;

          Solid solid_pa = Box(dd_tmp_name_pa, boxax / 2, boxay / 2, boxaz / 2.);
          sdx = 0;
          sdy = -boxay / 2 - es.LdrFrnt_Length / 2;
          sdz = -es.ladder_thick / 2. + es.LdrFrnt_Offset;

          Solid solid_pb = UnionSolid(dd_tmp_name_pb,
                                      solid_pa,
                                      solid_lfront,
                                      Transform3D(ns.rotation("esalgo:RM1299"), Position(sdx, sdy, sdz)));

          sdx = 0;
          sdy = boxay / 2 + es.LdrBck_Length / 2;
          sdz = -es.ladder_thick / 2. + es.LdrBck_Offset;
          Solid solid = UnionSolid(
              ddname, solid_pb, solid_lbck, Transform3D(ns.rotation("esalgo:RM1299"), Position(sdx, sdy, sdz)));
          /* Volume ladder = */
          ns.addVolumeNS(Volume(ddname, solid, ns.material(es.LaddMaterial_)));
          string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l4_[d]);
          /* Volume ladder2 = */
          ns.addVolumeNS(Volume(ddname2, solid, ns.material(es.LaddMaterial_)));
        }
      }

      // insert SWED, SFBX and SFBY into ladders
      swed_scopy_glob++;
      if (M < int(es.types_l5_.size())) {
        string ddname("esalgo:" + es.ladPfx_[0] + es.types_l5_[M]);
        string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l5_[M]);
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 4; j++) {
            xpos = (i * 2 - 1) * es.waf_intra_col_sep / 2.;
            ypos = -ladder_length / 2. + 0.05_mm - (es.LdrFrnt_Length - es.LdrBck_Length) / 2 + es.wedge_length / 2. +
                   j * es.waf_active;
            zpos = -es.ladder_thick / 2. + 0.005_mm + es.wedge_offset;
            if (es.ladd_l5_map_[(i + j * 2 + M * 10)] == 1) {
              scopy++;
              ns.volume(ddname).placeVolume(swedLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));
              ns.volume(ddname2).placeVolume(swedLog,
                                             scopy + 1000 * swed_scopy_glob + 100,
                                             Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));

              ypos = ypos + es.ywedge_ceramic_diff;
              zpos = -es.ladder_thick / 2. + 0.005_mm + es.zwedge_ceramic_diff;
              ns.volume(ddname).placeVolume(sfbxLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            Transform3D(ns.rotation("esalgo:RM1298"), Position(xpos, ypos, zpos)));

              ns.volume(ddname2).placeVolume(sfbyLog,
                                             scopy + 1000 * swed_scopy_glob,
                                             Transform3D(ns.rotation("esalgo:RM1300A"), Position(xpos, ypos, zpos)));
            }
          }
        }
      } else {
        int d = M - es.types_l5_.size();
        string ddname("esalgo:" + es.ladPfx_[0] + es.types_l4_[d]);
        string ddname2("esalgo:" + es.ladPfx_[1] + es.types_l4_[d]);
        for (int i = 0; i <= 1; i++) {
          for (int j = 0; j <= 3; j++) {
            xpos = (i * 2 - 1) * es.waf_intra_col_sep / 2.;
            ypos = -ladder_length / 2. + 0.05_mm - (es.LdrFrnt_Length - es.LdrBck_Length) / 2 + es.wedge_length / 2. +
                   j * es.waf_active;
            zpos = -es.ladder_thick / 2. + 0.005_mm + es.wedge_offset;
            if (es.ladd_l4_map_[(i + j * 2 + (M - es.types_l5_.size()) * 8)] == 1) {
              scopy++;
              ns.volume(ddname).placeVolume(swedLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));
              ns.volume(ddname2).placeVolume(swedLog,
                                             scopy + 1000 * swed_scopy_glob + 100,
                                             Transform3D(ns.rotation("esalgo:RM1299"), Position(xpos, ypos, zpos)));

              ypos = ypos + es.ywedge_ceramic_diff;
              zpos = -es.ladder_thick / 2. + 0.005_mm + es.zwedge_ceramic_diff;
              ns.volume(ddname).placeVolume(sfbxLog,
                                            scopy + 1000 * swed_scopy_glob,
                                            Transform3D(ns.rotation("esalgo:RM1298"), Position(xpos, ypos, zpos)));

              ns.volume(ddname2).placeVolume(sfbyLog,
                                             scopy + 1000 * swed_scopy_glob,
                                             Transform3D(ns.rotation("esalgo:RM1300A"), Position(xpos, ypos, zpos)));
            }
          }
        }
      }
    }

    // Positioning of ladders

    string type;
    int icopy[100] = {0};

    for (int I = -9; I <= 9; ++I) {
      prev_length_ = 0;
      int J = abs(I);
      for (int K = 0; K < es.noLaddInCol_[J]; K++) {
        string type;

        ladder_new_length_ = es.micromodule_length + 3 * es.waf_active;
        ladd_shift_ = 4 * es.waf_active;

        if (K == 0)
          type = es.typeOfLaddRow0[J];
        if (K == 1)
          type = es.typeOfLaddRow1[J];
        if (K == 2)
          type = es.typeOfLaddRow2[J];
        if (K == 3)
          type = es.typeOfLaddRow3[J];

        for (const auto& i : es.types_l5_)
          if (type == i) {
            ladder_new_length_ = es.micromodule_length + 4 * es.waf_active;
            ladd_shift_ = 5 * es.waf_active;
          }

        int j = 0;

        for (int t = 0; t < int(es.types_l5_.size()); t++)
          if (type == es.types_l5_[t]) {
            j = t;
            if (I < 0 && es.asym_ladd_[t] == 1) {
              j = j + 1;
              type = es.types_l5_[j];
            }
          }
        for (int t = 0; t < int(es.types_l4_.size()); t++)
          if (type == es.types_l4_[t]) {
            j = t + es.types_l5_.size();
            if (I < 0 && es.asym_ladd_[(t + es.types_l5_.size())] == 1) {
              j = j + 1;
              type = es.types_l4_[j - es.types_l5_.size()];
            }
          }

        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);
        if (I > 0)
          xpos = xpos + es.dee_separation;
        if (I < 0)
          xpos = xpos - es.dee_separation;

        int sz = 20;
        ypos = (sz - int(es.startOfFirstLadd_[J])) * es.waf_active - ladder_new_length_ / 2. +
               (es.LdrFrnt_Length - es.LdrBck_Length) / 2 + es.micromodule_length + 0.05_cm - prev_length_;

        prev_length_ += ladd_shift_;

        zpos = es.zlead1_ + es.ladder_thick / 2. + 0.01_mm;
        icopy[j] += 1;

        string ddname("esalgo:" + es.ladPfx_[0] + type);
        sfLog.placeVolume(ns.volume(ddname), icopy[j], Position(xpos, ypos, zpos));

        string ddname2("esalgo:" + es.ladPfx_[1] + type);
        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);
        sfLog.placeVolume(
            ns.volume(ddname2),
            icopy[j],
            Transform3D(ns.rotation("esalgo:R270"), Position(ypos, -xpos, zpos - es.zlead1_ + es.zlead2_)));

        int changed = 0;
        for (int t = 0; t < int(es.types_l5_.size()); t++)
          if (type == es.types_l5_[t]) {
            j = t;
            if (es.asym_ladd_[t] == 2 && !changed) {
              j = j - 1;
              changed = 1;
            }
            if (es.asym_ladd_[t] == 1 && !changed) {
              j = j + 1;
              changed = 1;
            }
            type = es.types_l5_[j];
          }
        for (int t = 0; t < int(es.types_l4_.size()); t++)
          if (type == es.types_l4_[t]) {
            j = t + es.types_l5_.size();
            if (es.asym_ladd_[(t + es.types_l5_.size())] == 2 && !changed) {
              j = j - 1;
              changed = 1;
            }
            if (es.asym_ladd_[(t + es.types_l5_.size())] == 1 && !changed) {
              j = j + 1;
              changed = 1;
            }
            type = es.types_l4_[j - es.types_l5_.size()];
          }

        icopy[j] += 1;

        if (I > 0)
          xpos = xpos + es.dee_separation;
        if (I < 0)
          xpos = xpos - es.dee_separation;

        string ddname3("esalgo:" + es.ladPfx_[0] + type);
        sfLog.placeVolume(
            ns.volume(ddname3), icopy[j], Transform3D(ns.rotation("esalgo:R180"), Position(xpos, -ypos, zpos)));

        string ddname4("esalgo:" + es.ladPfx_[1] + type);

        xpos = I * (2 * es.waf_intra_col_sep + es.waf_inter_col_sep);

        sfLog.placeVolume(
            ns.volume(ddname3),
            icopy[j],
            Transform3D(ns.rotation("esalgo:R090"), Position(-ypos, -xpos, zpos - es.zlead1_ + es.zlead2_)));
      }
    }
  }
  // place the slicon strips in active silicon wafers
  {
    double xpos(0), ypos(0);
    Volume sfwxLog = ns.volume("esalgo:SFWX");
    Volume sfwyLog = ns.volume("esalgo:SFWY");
    Volume sfsxLog = ns.volume("esalgo:SFSX");
    Volume sfsyLog = ns.volume("esalgo:SFSY");

    for (size_t i = 0; i < 32; ++i) {
      xpos = -es.waf_active / 2. + i * es.waf_active / 32. + es.waf_active / 64.;
      sfwxLog.placeVolume(sfsxLog, i + 1, Position(xpos, 0., 0.));

      ypos = -es.waf_active / 2. + i * es.waf_active / 32. + es.waf_active / 64.;
      sfwyLog.placeVolume(sfsyLog, i + 1, Position(0., ypos, 0.));
    }
  }
  return 1;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_ecal_DDEcalPreshowerAlgo, algorithm)
