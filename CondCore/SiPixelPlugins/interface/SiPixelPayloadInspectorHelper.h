#ifndef CONDCORE_SIPIXELPLUGINS_SIPIXELPAYLOADINSPECTORHELPER_H
#define CONDCORE_SIPIXELPLUGINS_SIPIXELPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>
#include <string>
#include "TGraph.h"
#include "TH1.h"
#include "TH2.h"
#include "TLatex.h"
#include "TLine.h"
#include "TPave.h"
#include "TPaveStats.h"
#include "TPaveText.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "CondCore/CondDB/interface/Time.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"

//#define MMDEBUG
#ifdef MMDEBUG
#include <iostream>
#define COUT std::cout << "MM "
#else
#define COUT edm::LogVerbatim("")
#endif

namespace SiPixelPI {

  // size of the phase-0 pixel detID list
  static const unsigned int phase0size = 1440;

  std::pair<unsigned int, unsigned int> unpack(cond::Time_t since) {
    auto kLowMask = 0XFFFFFFFF;
    auto run = (since >> 32);
    auto lumi = (since & kLowMask);
    return std::make_pair(run, lumi);
  }

  //============================================================================
  // Taken from pixel naming classes
  // BmO (-z-x) = 1, BmI (-z+x) = 2 , BpO (+z-x) = 3 , BpI (+z+x) = 4
  int quadrant(const DetId& detid, const TrackerTopology* tTopo_, bool phase_) {
    if (detid.subdetId() == PixelSubdetector::PixelBarrel) {
      return PixelBarrelName(detid, tTopo_, phase_).shell();
    } else {
      return PixelEndcapName(detid, tTopo_, phase_).halfCylinder();
    }
  }

  //============================================================================
  // Online ladder convention taken from pixel naming class for barrel
  // Apply sign convention (- sign for BmO and BpO)
  int signed_ladder(const DetId& detid, const TrackerTopology& tTopo_, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelBarrel)
      return -9999;
    int signed_ladder = PixelBarrelName(detid, &tTopo_, phase_).ladderName();
    if (quadrant(detid, &tTopo_, phase_) % 2)
      signed_ladder *= -1;
    return signed_ladder;
  }

  //============================================================================
  // Online mdoule convention taken from pixel naming class for barrel
  // Apply sign convention (- sign for BmO and BmI)
  int signed_module(const DetId& detid, const TrackerTopology& tTopo_, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelBarrel)
      return -9999;
    int signed_module = PixelBarrelName(detid, &tTopo_, phase_).moduleName();
    if (quadrant(detid, &tTopo_, phase_) < 3)
      signed_module *= -1;
    return signed_module;
  }

  //============================================================================
  // Phase 0: Ring was not an existing convention
  //   but the 7 plaquettes were split by HV group
  //   --> Derive Ring 1/2 for them
  //   Panel 1 plq 1-2, Panel 2, plq 1   = Ring 1
  //   Panel 1 plq 3-4, Panel 2, plq 2-3 = Ring 2
  // Phase 1: Using pixel naming class for endcap
  int ring(const DetId& detid, const TrackerTopology& tTopo_, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int ring = -9999;
    if (phase_ == 0) {
      ring = 1 + (tTopo_.pxfPanel(detid) + tTopo_.pxfModule(detid) > 3);
    } else if (phase_ == 1) {
      ring = PixelEndcapName(detid, &tTopo_, phase_).ringName();
    }
    return ring;
  }

  //============================================================================
  // Online blade convention taken from pixel naming class for endcap
  // Apply sign convention (- sign for BmO and BpO)
  int signed_blade(const DetId& detid, const TrackerTopology& tTopo_, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int signed_blade = PixelEndcapName(detid, &tTopo_, phase_).bladeName();
    if (quadrant(detid, &tTopo_, phase_) % 2)
      signed_blade *= -1;
    return signed_blade;
  }

  //============================================================================
  int signed_blade_panel(const DetId& detid, const TrackerTopology& tTopo_, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int signed_blade_panel = signed_blade(detid, tTopo_, phase_) + (tTopo_.pxfPanel(detid) - 1);
    return signed_blade_panel;
  }

  //============================================================================
  // Online disk convention
  // Apply sign convention (- sign for BmO and BmI)
  int signed_disk(const DetId& detid, const TrackerTopology& tTopo_, bool phase_) {
    if (detid.subdetId() != PixelSubdetector::PixelEndcap)
      return -9999;
    int signed_disk = tTopo_.pxfDisk(DetId(detid));
    if (quadrant(detid, &tTopo_, phase_) < 3)
      signed_disk *= -1;
    return signed_disk;
  }

  //============================================================================
  void draw_line(double x1, double x2, double y1, double y2, int width = 2, int style = 1, int color = 1) {
    TLine* l = new TLine(x1, y1, x2, y2);
    l->SetBit(kCanDelete);
    l->SetLineWidth(width);
    l->SetLineStyle(style);
    l->SetLineColor(color);
    l->Draw();
  }

  //============================================================================
  void dress_occup_plot(TCanvas& canv,
                        TH2* h,
                        int lay,
                        int ring = 0,
                        int phase = 0,
                        bool half_shift = true,
                        bool mark_zero = true,
                        bool standard_palette = true) {
    std::string s_title;

    if (lay > 0) {
      canv.cd(lay);
      s_title = "Barrel Pixel Layer" + std::to_string(lay);
    } else {
      canv.cd(ring);
      s_title = "Forward Pixel Ring" + std::to_string(ring);
    }

    gStyle->SetPadRightMargin(0.125);

    if (standard_palette) {
      gStyle->SetPalette(1);
    } else {
      // this is the fine gradient palette
      const Int_t NRGBs = 5;
      const Int_t NCont = 255;

      Double_t stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
      Double_t red[NRGBs] = {0.00, 0.00, 0.87, 1.00, 0.51};
      Double_t green[NRGBs] = {0.00, 0.81, 1.00, 0.20, 0.00};
      Double_t blue[NRGBs] = {0.51, 1.00, 0.12, 0.00, 0.00};
      TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
      gStyle->SetNumberContours(NCont);
    }

    h->SetMarkerSize(0.7);
    h->Draw("colz");

    auto ltx = TLatex();
    ltx.SetTextFont(62);
    ltx.SetTextColor(1);
    ltx.SetTextSize(0.06);
    ltx.SetTextAlign(31);
    ltx.DrawLatexNDC(1 - gPad->GetRightMargin(), 1 - gPad->GetTopMargin() + 0.01, (s_title).c_str());

    // Draw Lines around modules
    if (lay > 0) {
      std::vector<std::vector<int> > nladder = {{10, 16, 22}, {6, 14, 22, 32}};
      int nlad = nladder[phase][lay - 1];
      for (int xsign = -1; xsign <= 1; xsign += 2)
        for (int ysign = -1; ysign <= 1; ysign += 2) {
          float xlow = xsign * (half_shift * 0.5);
          float xhigh = xsign * (half_shift * 0.5 + 4);
          float ylow = ysign * (half_shift * 0.5 + (phase == 0) * 0.5);
          float yhigh = ysign * (half_shift * 0.5 - (phase == 0) * 0.5 + nlad);
          // Outside box
          draw_line(xlow, xhigh, ylow, ylow, 1);    // bottom
          draw_line(xlow, xhigh, yhigh, yhigh, 1);  // top
          draw_line(xlow, xlow, ylow, yhigh, 1);    // left
          draw_line(xhigh, xhigh, ylow, yhigh, 1);  // right
          // Inner Horizontal lines
          for (int lad = 1; lad < nlad; ++lad) {
            float y = ysign * (lad + half_shift * 0.5);
            draw_line(xlow, xhigh, y, y, 1);
          }
          for (int lad = 1; lad <= nlad; ++lad)
            if (!(phase == 0 && (lad == 1 || lad == nlad))) {
              float y = ysign * (lad + half_shift * 0.5 - 0.5);
              draw_line(xlow, xhigh, y, y, 1, 3);
            }
          // Inner Vertical lines
          for (int mod = 1; mod < 4; ++mod) {
            float x = xsign * (mod + half_shift * 0.5);
            draw_line(x, x, ylow, yhigh, 1);
          }
          // Make a BOX around ROC 0
          // Phase 0 - ladder +1 is always non-flipped
          // Phase 1 - ladder +1 is always     flipped
          if (mark_zero) {
            for (int mod = 1; mod <= 4; ++mod)
              for (int lad = 1; lad <= nlad; ++lad) {
                bool flipped = ysign == 1 ? lad % 2 == 0 : lad % 2 == 1;
                if (phase == 1)
                  flipped = !flipped;
                int roc0_orientation = flipped ? -1 : 1;
                if (xsign == -1)
                  roc0_orientation *= -1;
                if (ysign == -1)
                  roc0_orientation *= -1;
                float x1 = xsign * (mod + half_shift * 0.5);
                float x2 = xsign * (mod + half_shift * 0.5 - 1. / 8);
                float y1 = ysign * (lad + half_shift * 0.5 - 0.5);
                float y2 = ysign * (lad + half_shift * 0.5 - 0.5 + roc0_orientation * 1. / 2);
                if (!(phase == 0 && (lad == 1 || lad == nlad) && xsign == -1)) {
                  if (lay == 1 && xsign <= -1) {
                    float x1 = xsign * ((mod - 1) + half_shift * 0.5);
                    float x2 = xsign * ((mod - 1) + half_shift * 0.5 + 1. / 8);
                    float y1 = ysign * (lad + half_shift * 0.5 - 0.5 + roc0_orientation);
                    float y2 = ysign * (lad + half_shift * 0.5 - 0.5 + roc0_orientation * 3. / 2);
                    draw_line(x1, x2, y1, y1, 1);
                    draw_line(x2, x2, y1, y2, 1);
                  } else {
                    draw_line(x1, x2, y1, y1, 1);
                    //draw_line(x1, x2, y2, y2, 1);
                    //draw_line(x1, x1, y1, y2, 1);
                    draw_line(x2, x2, y1, y2, 1);
                  }
                }
              }
          }
        }
    } else {
      // FPIX
      for (int dsk = 1, ndsk = 2 + (phase == 1); dsk <= ndsk; ++dsk) {
        for (int xsign = -1; xsign <= 1; xsign += 2)
          for (int ysign = -1; ysign <= 1; ysign += 2) {
            if (phase == 0) {
              int first_roc = 3, nbin = 16;
              for (int bld = 1, nbld = 12; bld <= nbld; ++bld) {
                // Horizontal lines
                for (int plq = 1, nplq = 7; plq <= nplq; ++plq) {
                  float xlow =
                      xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * plq + (plq == 1)) / (float)nbin);
                  float xhigh =
                      xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * (plq + 1) - (plq == 7)) / (float)nbin);
                  float ylow = ysign * (half_shift * 0.5 + (bld - 0.5) - (2 + plq / 2) * 0.1);
                  float yhigh = ysign * (half_shift * 0.5 + (bld - 0.5) + (2 + plq / 2) * 0.1);
                  draw_line(xlow, xhigh, ylow, ylow, 1);    // bottom
                  draw_line(xlow, xhigh, yhigh, yhigh, 1);  // top
                }
                // Vertical lines
                for (int plq = 1, nplq = 7 + 1; plq <= nplq; ++plq) {
                  float x = xsign * (half_shift * 0.5 + dsk - 1 +
                                     (first_roc - 3 + 2 * plq + (plq == 1) - (plq == 8)) / (float)nbin);
                  float ylow = ysign * (half_shift * 0.5 + (bld - 0.5) - (2 + (plq - (plq == 8)) / 2) * 0.1);
                  float yhigh = ysign * (half_shift * 0.5 + (bld - 0.5) + (2 + (plq - (plq == 8)) / 2) * 0.1);
                  draw_line(x, x, ylow, yhigh, 1);
                }
                // Panel 2 has dashed mid-plane
                for (int plq = 2, nplq = 6; plq <= nplq; ++plq)
                  if (plq % 2 == 0) {
                    float x = xsign * (half_shift * 0.5 + dsk - 1 +
                                       (first_roc - 3 + 2 * plq + (plq == 1) - (plq == 8) + 1) / (float)nbin);
                    float ylow = ysign * (half_shift * 0.5 + (bld - 0.5) - (2 + (plq - (plq == 8)) / 2) * 0.1);
                    float yhigh = ysign * (half_shift * 0.5 + (bld - 0.5) + (2 + (plq - (plq == 8)) / 2) * 0.1);
                    draw_line(x, x, ylow, yhigh, 1, 2);
                  }
                // Make a BOX around ROC 0
                for (int plq = 1, nplq = 7; plq <= nplq; ++plq) {
                  float x1 =
                      xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * plq + (plq == 1)) / (float)nbin);
                  float x2 =
                      xsign * (half_shift * 0.5 + dsk - 1 + (first_roc - 3 + 2 * plq + (plq == 1) + 1) / (float)nbin);
                  int sign = xsign * ysign * ((plq % 2) ? 1 : -1);
                  float y1 = ysign * (half_shift * 0.5 + (bld - 0.5) + sign * (2 + plq / 2) * 0.1);
                  float y2 = ysign * (half_shift * 0.5 + (bld - 0.5) + sign * (plq / 2) * 0.1);
                  //draw_line(x1, x2, y1, y1, 1);
                  draw_line(x1, x2, y2, y2, 1);
                  //draw_line(x1, x1, y1, y2, 1);
                  draw_line(x2, x2, y1, y2, 1);
                }
              }
            } else if (phase == 1) {
              if (ring == 0) {  // both
                for (int ring = 1; ring <= 2; ++ring)
                  for (int bld = 1, nbld = 5 + ring * 6; bld <= nbld; ++bld) {
                    float scale = (ring == 1) ? 1.5 : 1;
                    Color_t p1_color = 1, p2_color = 1;
                    // Horizontal lines
                    // Panel 2 has dashed mid-plane
                    float x1 = xsign * (half_shift * 0.5 + dsk - 1 + (ring - 1) * 0.5);
                    float x2 = xsign * (half_shift * 0.5 + dsk - 1 + ring * 0.5);
                    int sign = ysign;
                    float y1 = ysign * (half_shift * 0.5 - 0.5 + scale * bld + sign * 0.5);
                    //float yp1_mid = ysign * (half_shift*0.5 - 0.5 + scale*bld + sign*0.25);
                    float y2 = ysign * (half_shift * 0.5 - 0.5 + scale * bld);
                    float yp2_mid = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.25);
                    float y3 = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.5);
                    draw_line(x1, x2, y1, y1, 1, 1, p1_color);
                    //draw_line(x1, x2, yp1_mid, yp1_mid, 1, 3);
                    draw_line(x1, x2, y2, y2, 1, 1, p1_color);
                    draw_line(x1, x2, yp2_mid, yp2_mid, 1, 2);
                    draw_line(x1, x2, y3, y3, 1, 1, p2_color);
                    // Vertical lines
                    float x = xsign * (half_shift * 0.5 + dsk - 1 + (ring - 1) * 0.5);
                    draw_line(x, x, y1, y2, 1, 1, p1_color);
                    draw_line(x, x, y2, y3, 1, 1, p2_color);
                    if (ring == 2) {
                      //draw_line(x,  x,  y2,  y3, 1, 1, p1_color);
                      x = xsign * (half_shift * 0.5 + dsk);
                      draw_line(x, x, y1, y2, 1, 1, p1_color);
                      draw_line(x, x, y2, y3, 1, 1, p2_color);
                    }
                    // Make a BOX around ROC 0
                    x1 = xsign * (half_shift * 0.5 + dsk - 1 + ring * 0.5 - 1 / 16.);
                    x2 = xsign * (half_shift * 0.5 + dsk - 1 + ring * 0.5);
                    float y1_p1 = ysign * (half_shift * 0.5 - 0.5 + scale * bld + sign * 0.25);
                    float y2_p1 = ysign * (half_shift * 0.5 - 0.5 + scale * bld + sign * 0.25 + xsign * ysign * 0.25);
                    draw_line(x1, x2, y1_p1, y1_p1, 1);
                    //draw_line(x1, x2, y2_p1, y2_p1, 1);
                    draw_line(x1, x1, y1_p1, y2_p1, 1);
                    //draw_line(x2, x2, y1_p1, y2_p1, 1);
                    float y1_p2 = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.25);
                    float y2_p2 = ysign * (half_shift * 0.5 - 0.5 + scale * bld - sign * 0.25 - xsign * ysign * 0.25);
                    draw_line(x1, x2, y1_p2, y1_p2, 1);
                    //draw_line(x1, x2, y2_p2, y2_p2, 1);
                    draw_line(x1, x1, y1_p2, y2_p2, 1);
                    //draw_line(x2, x2, y1_p2, y2_p2, 1);
                  }
              } else {  // only one ring, 1 or 2
                for (int bld = 1, nbld = 5 + ring * 6; bld <= nbld; ++bld) {
                  Color_t p1_color = 1, p2_color = 1;
                  // Horizontal lines
                  // Panel 2 has dashed mid-plane
                  float x1 = xsign * (half_shift * 0.5 + dsk - 1);
                  float x2 = xsign * (half_shift * 0.5 + dsk);
                  int sign = ysign;
                  float y1 = ysign * (half_shift * 0.5 - 0.5 + bld + sign * 0.5);
                  //float yp1_mid = ysign * (half_shift*0.5 - 0.5 + bld + sign*0.25);
                  float y2 = ysign * (half_shift * 0.5 - 0.5 + bld);
                  float yp2_mid = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.25);
                  float y3 = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.5);
                  draw_line(x1, x2, y1, y1, 1, 1, p1_color);
                  //draw_line(x1, x2, yp1_mid, yp1_mid, 1, 3);
                  draw_line(x1, x2, y2, y2, 1, 1, p1_color);
                  draw_line(x1, x2, yp2_mid, yp2_mid, 1, 2);
                  draw_line(x1, x2, y3, y3, 1, 1, p2_color);
                  // Vertical lines
                  float x = xsign * (half_shift * 0.5 + dsk - 1);
                  draw_line(x, x, y1, y2, 1, 1, p1_color);
                  draw_line(x, x, y2, y3, 1, 1, p2_color);
                  if (ring == 2) {
                    //draw_line(x,  x,  y2,  y3, 1, 1, p1_color);
                    x = xsign * (half_shift * 0.5 + dsk);
                    draw_line(x, x, y1, y2, 1, 1, p1_color);
                    draw_line(x, x, y2, y3, 1, 1, p2_color);
                  }
                  // Make a BOX around ROC 0
                  x1 = xsign * (half_shift * 0.5 + dsk - 1 / 8.);
                  x2 = xsign * (half_shift * 0.5 + dsk);
                  float y1_p1 = ysign * (half_shift * 0.5 - 0.5 + bld + sign * 0.25);
                  float y2_p1 = ysign * (half_shift * 0.5 - 0.5 + bld + sign * 0.25 + xsign * ysign * 0.25);
                  draw_line(x1, x2, y1_p1, y1_p1, 1);
                  //draw_line(x1, x2, y2_p1, y2_p1, 1);
                  draw_line(x1, x1, y1_p1, y2_p1, 1);
                  //draw_line(x2, x2, y1_p1, y2_p1, 1);
                  float y1_p2 = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.25);
                  float y2_p2 = ysign * (half_shift * 0.5 - 0.5 + bld - sign * 0.25 - xsign * ysign * 0.25);
                  draw_line(x1, x2, y1_p2, y1_p2, 1);
                  //draw_line(x1, x2, y2_p2, y2_p2, 1);
                  draw_line(x1, x1, y1_p2, y2_p2, 1);
                  //draw_line(x2, x2, y1_p2, y2_p2, 1);
                }
              }
            }
          }
      }
      // Special shifted "rebin" for Phase 0
      // Y axis should always have at least half-roc granularity because
      // there are half-ROC size shifts implemented in the coordinates
      // To remove this and show full ROC granularity
      // We merge bin contents in each pair of bins corresponding to one ROC
      // TODO: make sure this works for Profiles
      if (phase == 0 && h->GetNbinsY() == 250 && h->GetNbinsX() == 80) {
        int nentries = h->GetEntries();
        for (int binx = 1; binx <= 80; ++binx) {
          double sum = 0;
          for (int biny = 1; biny <= 250; ++biny) {
            bool odd_nrocy = (binx - 1 < 40) != (((binx - 1) / 4) % 2);
            if (biny % 2 == odd_nrocy)
              sum += h->GetBinContent(binx, biny);
            else {
              sum += h->GetBinContent(binx, biny);
              if (sum) {
                h->SetBinContent(binx, biny, sum);
                h->SetBinContent(binx, biny - 1, sum);
              }
              sum = 0;
            }
          }
        }
        h->SetEntries(nentries);
      }
    }
  }

  /*--------------------------------------------------------------------*/
  std::pair<float, float> getExtrema(TH1* h1, TH1* h2)
  /*--------------------------------------------------------------------*/
  {
    float theMax(-9999.);
    float theMin(9999.);
    theMax = h1->GetMaximum() > h2->GetMaximum() ? h1->GetMaximum() : h2->GetMaximum();
    theMin = h1->GetMinimum() < h2->GetMaximum() ? h1->GetMinimum() : h2->GetMinimum();

    float add_min = theMin > 0. ? -0.05 : 0.05;
    float add_max = theMax > 0. ? 0.05 : -0.05;

    auto result = std::make_pair(theMin * (1 + add_min), theMax * (1 + add_max));
    return result;
  }

  /*--------------------------------------------------------------------*/
  void makeNicePlotStyle(TH1* hist)
  /*--------------------------------------------------------------------*/
  {
    hist->SetStats(kFALSE);
    hist->SetLineWidth(2);
    hist->GetXaxis()->CenterTitle(true);
    hist->GetYaxis()->CenterTitle(true);
    hist->GetXaxis()->SetTitleFont(42);
    hist->GetYaxis()->SetTitleFont(42);
    hist->GetXaxis()->SetTitleSize(0.05);
    hist->GetYaxis()->SetTitleSize(0.05);
    hist->GetXaxis()->SetTitleOffset(1.1);
    hist->GetYaxis()->SetTitleOffset(1.3);
    hist->GetXaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelFont(42);
    hist->GetYaxis()->SetLabelSize(.05);
    hist->GetXaxis()->SetLabelSize(.05);
  }

  enum regions {
    BPixL1o,        //0  Barrel Pixel Layer 1 outer
    BPixL1i,        //1  Barrel Pixel Layer 1 inner
    BPixL2o,        //2  Barrel Pixel Layer 2 outer
    BPixL2i,        //3  Barrel Pixel Layer 2 inner
    BPixL3o,        //4  Barrel Pixel Layer 3 outer
    BPixL3i,        //5  Barrel Pixel Layer 3 inner
    BPixL4o,        //6  Barrel Pixel Layer 4 outer
    BPixL4i,        //7  Barrel Pixel Layer 4 inner
    FPixmL1,        //8  Forward Pixel Minus side Disk 1
    FPixmL2,        //9  Forward Pixel Minus side Disk 2
    FPixmL3,        //10 Forward Pixel Minus side Disk 3
    FPixpL1,        //11 Forward Pixel Plus side Disk 1
    FPixpL2,        //12 Forward Pixel Plus side Disk 2
    FPixpL3,        //13 Forward Pixel Plus side Disk 3
    NUM_OF_REGIONS  //14 -- default
  };

  /*--------------------------------------------------------------------*/
  std::string getStringFromRegionEnum(SiPixelPI::regions e)
  /*--------------------------------------------------------------------*/
  {
    switch (e) {
      case SiPixelPI::BPixL1o:
        return "BPix L1/o";
      case SiPixelPI::BPixL1i:
        return "BPix L1/i";
      case SiPixelPI::BPixL2o:
        return "BPix L2/o";
      case SiPixelPI::BPixL2i:
        return "BPix L2/i";
      case SiPixelPI::BPixL3o:
        return "BPix L3/o";
      case SiPixelPI::BPixL3i:
        return "BPix L3/i";
      case SiPixelPI::BPixL4o:
        return "BPix L4/o";
      case SiPixelPI::BPixL4i:
        return "BPix L4/i";
      case SiPixelPI::FPixmL1:
        return "FPix- D1";
      case SiPixelPI::FPixmL2:
        return "FPix- D2";
      case SiPixelPI::FPixmL3:
        return "FPix- D3";
      case SiPixelPI::FPixpL1:
        return "FPix+ D1";
      case SiPixelPI::FPixpL2:
        return "FPix+ D2";
      case SiPixelPI::FPixpL3:
        return "FPix+ D3";
      default:
        edm::LogWarning("LogicError") << "Unknown partition: " << e;
        return "";
    }
  }

  /*--------------------------------------------------------------------*/
  bool isBPixOuterLadder(const DetId& detid, const TrackerTopology& tTopo, bool isPhase0)
  /*--------------------------------------------------------------------*/
  {
    bool isOuter = false;
    int layer = tTopo.pxbLayer(detid.rawId());
    bool odd_ladder = tTopo.pxbLadder(detid.rawId()) % 2;
    if (isPhase0) {
      if (layer == 2)
        isOuter = !odd_ladder;
      else
        isOuter = odd_ladder;
    } else {
      if (layer == 4)
        isOuter = odd_ladder;
      else
        isOuter = !odd_ladder;
    }
    return isOuter;
  }

  // ancillary struct to manage the topology
  // info in a more compact way

  struct topolInfo {
  private:
    uint32_t m_rawid;
    int m_subdetid;
    int m_layer;
    int m_side;
    int m_ring;
    bool m_isInternal;

  public:
    void init();
    void fillGeometryInfo(const DetId& detId, const TrackerTopology& tTopo, bool isPhase0);
    SiPixelPI::regions filterThePartition();
    bool sanityCheck();
    void printAll();
    virtual ~topolInfo() {}
  };

  /*--------------------------------------------------------------------*/
  void topolInfo::printAll()
  /*--------------------------------------------------------------------*/
  {
    std::cout << " detId:" << m_rawid << " subdetid: " << m_subdetid << " layer: " << m_layer << " side: " << m_side
              << " ring: " << m_ring << " isInternal:" << m_isInternal << std::endl;
  }

  /*--------------------------------------------------------------------*/
  void topolInfo::init()
  /*--------------------------------------------------------------------*/
  {
    m_rawid = 0;
    m_subdetid = -1;
    m_layer = -1;
    m_side = -1;
    m_ring = -1;
    m_isInternal = false;
  };

  /*--------------------------------------------------------------------*/
  bool topolInfo::sanityCheck()
  /*--------------------------------------------------------------------*/
  {
    if (m_layer == 0 || (m_subdetid == 1 && m_layer > 4) || (m_subdetid == 2 && m_layer > 3)) {
      return false;
    } else {
      return true;
    }
  }
  /*--------------------------------------------------------------------*/
  void topolInfo::fillGeometryInfo(const DetId& detId, const TrackerTopology& tTopo, bool isPhase0)
  /*--------------------------------------------------------------------*/
  {
    unsigned int subdetId = static_cast<unsigned int>(detId.subdetId());

    m_rawid = detId.rawId();
    m_subdetid = subdetId;
    if (subdetId == PixelSubdetector::PixelBarrel) {
      m_layer = tTopo.pxbLayer(detId.rawId());
      m_isInternal = !SiPixelPI::isBPixOuterLadder(detId, tTopo, isPhase0);
    } else if (subdetId == PixelSubdetector::PixelEndcap) {
      m_layer = tTopo.pxfDisk(detId.rawId());
      m_side = tTopo.pxfSide(detId.rawId());
    } else
      edm::LogWarning("LogicError") << "Unknown subdetid: " << subdetId;
  }

  // ------------ method to assign a partition based on the topology struct info ---------------

  /*--------------------------------------------------------------------*/
  SiPixelPI::regions topolInfo::filterThePartition()
  /*--------------------------------------------------------------------*/
  {
    SiPixelPI::regions ret = SiPixelPI::NUM_OF_REGIONS;

    // BPix
    if (m_subdetid == 1) {
      switch (m_layer) {
        case 1:
          m_isInternal > 0 ? ret = SiPixelPI::BPixL1o : ret = SiPixelPI::BPixL1i;
          break;
        case 2:
          m_isInternal > 0 ? ret = SiPixelPI::BPixL2o : ret = SiPixelPI::BPixL2i;
          break;
        case 3:
          m_isInternal > 0 ? ret = SiPixelPI::BPixL3o : ret = SiPixelPI::BPixL3i;
          break;
        case 4:
          m_isInternal > 0 ? ret = SiPixelPI::BPixL4o : ret = SiPixelPI::BPixL4i;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow BPix layer: " << m_layer;
          break;
      }
      // FPix
    } else if (m_subdetid == 2) {
      switch (m_layer) {
        case 1:
          m_side > 1 ? ret = SiPixelPI::FPixpL1 : ret = SiPixelPI::FPixmL1;
          break;
        case 2:
          m_side > 1 ? ret = SiPixelPI::FPixpL2 : ret = SiPixelPI::FPixmL2;
          break;
        case 3:
          m_side > 1 ? ret = SiPixelPI::FPixpL3 : ret = SiPixelPI::FPixmL3;
          break;
        default:
          edm::LogWarning("LogicError") << "Unknow FPix disk: " << m_layer;
          break;
      }
    }
    return ret;
  }

  // overloaded method: mask entire module
  /*--------------------------------------------------------------------*/
  std::vector<std::pair<int, int> > maskedBarrelRocsToBins(int layer, int ladder, int module)
  /*--------------------------------------------------------------------*/
  {
    std::vector<std::pair<int, int> > rocsToMask;

    int nlad_list[4] = {6, 14, 22, 32};
    int nlad = nlad_list[layer - 1];

    int start_x = module > 0 ? ((module + 4) * 8) + 1 : ((4 - (std::abs(module))) * 8) + 1;
    int start_y = ladder > 0 ? ((ladder + nlad) * 2) + 1 : ((nlad - (std::abs(ladder))) * 2) + 1;

    int end_x = start_x + 7;
    int end_y = start_y + 1;

    COUT << "module: " << module << " start_x:" << start_x << " end_x:" << end_x << std::endl;
    COUT << "ladder: " << ladder << " start_y:" << start_y << " end_y:" << end_y << std::endl;
    COUT << "==================================================================" << std::endl;

    for (int bin_x = 1; bin_x <= 72; bin_x++) {
      for (int bin_y = 1; bin_y <= (nlad * 4 + 2); bin_y++) {
        if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
          rocsToMask.push_back(std::make_pair(bin_x, bin_y));
        }
      }
    }
    return rocsToMask;
  }

  // overloaded method: mask single ROCs
  /*--------------------------------------------------------------------*/
  std::vector<std::tuple<int, int, int> > maskedBarrelRocsToBins(
      int layer, int ladder, int module, std::bitset<16> bad_rocs, bool isFlipped)
  /*--------------------------------------------------------------------*/
  {
    std::vector<std::tuple<int, int, int> > rocsToMask;

    int nlad_list[4] = {6, 14, 22, 32};
    int nlad = nlad_list[layer - 1];

    int start_x = module > 0 ? ((module + 4) * 8) + 1 : ((4 - (std::abs(module))) * 8) + 1;
    int start_y = ladder > 0 ? ((ladder + nlad) * 2) + 1 : ((nlad - (std::abs(ladder))) * 2) + 1;

    int roc0_x = ((layer == 1) || (layer > 1 && module > 0)) ? start_x + 7 : start_x;
    int roc0_y = start_y - 1;

    size_t idx = 0;
    while (idx < bad_rocs.size()) {
      if (bad_rocs.test(idx)) {
        //////////////////////////////////////////////////////////////////////////////////////
        //		                          |					    //
        // In BPix Layer1 and module>0 in L2,3,4  |   In BPix Layer 2,3,4 module > 0	    //
        //                                        |					    //
        // ROCs are ordered in the following      |   ROCs are ordered in the following     //
        // fashion for unplipped modules 	  |   fashion for unplipped modules         //
        //					  |  				            //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 8 |9  |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        //					  |  					    //
        // if the module is flipped the ordering  |   if the module is flipped the ordering //
        // is reveresed                           |   is reversed                           //
        //					  |                                         //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 8 | 9 |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        //////////////////////////////////////////////////////////////////////////////////////

        int roc_x(0), roc_y(0);

        if ((layer == 1) || (layer > 1 && module > 0)) {
          if (!isFlipped) {
            roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
            roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
          } else {
            roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
            roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
          }
        } else {
          if (!isFlipped) {
            roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
            roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
          } else {
            roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
            roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
          }
        }

        COUT << bad_rocs << " : (idx)= " << idx << std::endl;
        COUT << " layer:  " << layer << std::endl;
        COUT << "module: " << module << " roc_x:" << roc_x << std::endl;
        COUT << "ladder: " << ladder << " roc_y:" << roc_y << std::endl;
        COUT << "==================================================================" << std::endl;

        rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
      }
      ++idx;
    }
    return rocsToMask;
  }

  // overloaded method: mask entire module
  /*--------------------------------------------------------------------*/
  std::vector<std::pair<int, int> > maskedForwardRocsToBins(int ring, int blade, int panel, int disk)
  /*--------------------------------------------------------------------*/
  {
    std::vector<std::pair<int, int> > rocsToMask;

    //int nblade_list[2] = {11, 17};
    int nybins_list[2] = {92, 140};
    //int nblade = nblade_list[ring - 1];
    int nybins = nybins_list[ring - 1];

    int start_x = disk > 0 ? ((disk + 3) * 8) + 1 : ((3 - (std::abs(disk))) * 8) + 1;
    //int start_y = blade > 0 ? ((blade+nblade)*4)-panel*2  : ((nblade-(std::abs(blade)))*4)-panel*2;
    int start_y = blade > 0 ? (nybins / 2) + (blade * 4) - (panel * 2) + 3
                            : ((nybins / 2) - (std::abs(blade) * 4) - panel * 2) + 3;

    int end_x = start_x + 7;
    int end_y = start_y + 1;

    COUT << "==================================================================" << std::endl;
    COUT << "disk:  " << disk << " start_x:" << start_x << " end_x:" << end_x << std::endl;
    COUT << "blade: " << blade << " start_y:" << start_y << " end_y:" << end_y << std::endl;

    for (int bin_x = 1; bin_x <= 56; bin_x++) {
      for (int bin_y = 1; bin_y <= nybins; bin_y++) {
        if (bin_x >= start_x && bin_x <= end_x && bin_y >= start_y && bin_y <= end_y) {
          rocsToMask.push_back(std::make_pair(bin_x, bin_y));
        }
      }
    }
    return rocsToMask;
  }

  // overloaded method: mask single ROCs
  /*--------------------------------------------------------------------*/
  std::vector<std::tuple<int, int, int> > maskedForwardRocsToBins(
      int ring, int blade, int panel, int disk, std::bitset<16> bad_rocs, bool isFlipped)
  /*--------------------------------------------------------------------*/
  {
    std::vector<std::tuple<int, int, int> > rocsToMask;

    //int nblade_list[2] = {11, 17};
    int nybins_list[2] = {92, 140};
    //int nblade = nblade_list[ring - 1];
    int nybins = nybins_list[ring - 1];

    int start_x = disk > 0 ? ((disk + 3) * 8) + 1 : ((3 - (std::abs(disk))) * 8) + 1;
    //int start_y = blade > 0 ? ((blade+nblade)*4)-panel*2  : ((nblade-(std::abs(blade)))*4)-panel*2;
    int start_y = blade > 0 ? (nybins / 2) + (blade * 4) - (panel * 2) + 3
                            : ((nybins / 2) - (std::abs(blade) * 4) - panel * 2) + 3;

    int roc0_x = disk > 0 ? start_x + 7 : start_x;
    int roc0_y = start_y - 1;

    size_t idx = 0;
    while (idx < bad_rocs.size()) {
      if (bad_rocs.test(idx)) {
        int roc_x(0), roc_y(0);

        //////////////////////////////////////////////////////////////////////////////////////
        //		                          |					    //
        // In FPix + (Disk 1,2,3)                 |   In FPix - (Disk -1,-2,-3)	            //
        //                                        |					    //
        // ROCs are ordered in the following      |   ROCs are ordered in the following     //
        // fashion for unplipped modules 	  |   fashion for unplipped modules         //
        //					  |  				            //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 8 |9  |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        //					  |  					    //
        // if the module is flipped the ordering  |   if the module is flipped the ordering //
        // is reveresed                           |   is reversed                           //
        //					  |                                         //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 7 | 6 | 5 | 4 | 3 | 2 | 1 | 0 |      |   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        // | 8 | 9 |10 |11 |12 |13 |14 |15 |      |   |15 |14 |13 |12 |11 |10 | 9 | 8 |     //
        // +---+---+---+---+---+---+---+---+      |   +---+---+---+---+---+---+---+---+     //
        //////////////////////////////////////////////////////////////////////////////////////

        if (disk > 0) {
          if (!isFlipped) {
            roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
            roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
          } else {
            roc_x = idx < 8 ? roc0_x - idx : (start_x - 8) + idx;
            roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
          }
        } else {
          if (!isFlipped) {
            roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
            roc_y = idx < 8 ? roc0_y + 1 : roc0_y + 2;
          } else {
            roc_x = idx < 8 ? roc0_x + idx : (roc0_x + 7) - (idx - 8);
            roc_y = idx < 8 ? roc0_y + 2 : roc0_y + 1;
          }
        }

        COUT << bad_rocs << " : (idx)= " << idx << std::endl;
        COUT << " panel: " << panel << " isFlipped: " << isFlipped << std::endl;
        COUT << " disk:  " << disk << " roc_x:" << roc_x << std::endl;
        COUT << " blade: " << blade << " roc_y:" << roc_y << std::endl;
        COUT << "===============================" << std::endl;

        rocsToMask.push_back(std::make_tuple(roc_x, roc_y, idx));
      }
      ++idx;
    }
    return rocsToMask;
  }

};  // namespace SiPixelPI
#endif
