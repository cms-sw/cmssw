/*!
  \file SiPixelMapsRenderPlugin
  \brief RenderPlugin for plots based on SiPixelCoordinates

This RenderPlugin adds lines and markers about the ROC layout of Pixel modules
to 2D plots with a per-ROC granularity. Such plots can be created using the 
SiPixelCoordinates class and are included in the Phase1 Pixel DQM, but can also
be used in other contexts.

Based on code by Janos Karancsi (Janos.Karancsi@cern.ch)

  \author Marcel Schneider
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLine.h"
#include <cassert>
#include <cassert>
#include <string>

using namespace std;

class SiPixelMapsRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies( const VisDQMObject & o, const VisDQMImgInfo & )
    {
      if ((o.name.find( "PixelPhase1/Phase1_MechanicalView" ) != std::string::npos || o.name.find( "PixelPhase1/Tracks" ) != std::string::npos || o.name.find( "PixelPhase1/FED" ) != std::string::npos || o.name.find( "PixelPhase1Timing/" ) != std::string::npos  || o.name.find("PixelPhase1/SiPixelQualityPCL") != std::string::npos)
        && o.object && o.name.find( "Coord" ) != std::string::npos ) {
        return true;
      } else {
        return false;
      }
    }

  //virtual void preDraw( TCanvas * canvas, const VisDQMObject & o, const VisDQMImgInfo & , VisDQMRenderInfo & renderInfo)
    //{
      //canvas->cd();
      //TH2* obj = dynamic_cast<TH2*>( o.object );
      //if(!obj) return;
    //}

  virtual void postDraw( TCanvas *canvas, const VisDQMObject &o, const VisDQMImgInfo &/*ri*/ )
    {
      canvas->cd();
      TH2* obj = dynamic_cast<TH2*>( o.object );
      if(!obj) return;

      int h = obj->GetNbinsY(), w = obj->GetNbinsX();

      // Phase1, Layer 1-4
      if (w == 72 && h ==  26) dress_occup_plot(obj, 1, 0, 1);
      if (w == 72 && h ==  58) dress_occup_plot(obj, 2, 0, 1);
      if (w == 72 && h ==  90) dress_occup_plot(obj, 3, 0, 1);
      if (w == 72 && h == 130) dress_occup_plot(obj, 4, 0, 1);


      // Phase1 FPIX Ring 1-2
      if (w == 56 && h ==  92) dress_occup_plot(obj, 0, 1, 1);
      if (w == 56 && h == 140) dress_occup_plot(obj, 0, 2, 1);

      // Phase1 full FPIX
      if (w == 112 && h == 140) dress_occup_plot(obj, 0, 0, 1);

      // Phase0 BPIX  Layer 1-3
      if (w == 72 && h == 40) dress_occup_plot(obj, 1, 0, 0);
      if (w == 72 && h == 64) dress_occup_plot(obj, 2, 0, 0);
      if (w == 72 && h == 88) dress_occup_plot(obj, 3, 0, 0);

      // Phase0 full FPIX
      // TODO: Not sure how Janos got to w=80, which the rebinning code expects
      // Ignore the rebinning for now, the line overlay seems correct.
      if (w == 78 && h == 250) dress_occup_plot(obj, 0, 0, 0);

      //std::cout << "+++ if (w == " << w << " && h == " << h << ")\n";

    }

private:

  void draw_line(double x1, double x2, double y1, double y2, int width=2, int style=1, int color=1) {
    TLine* l = new TLine(x1, y1, x2, y2);
    l->SetBit(kCanDelete);
    l->SetLineWidth(width);
    l->SetLineStyle(style);
    l->SetLineColor(color);
    l->Draw();
  }

  void dress_occup_plot(TH2* h, int lay, int ring=0, int phase=0, bool half_shift = 1, bool mark_zero=1) {
    // Draw Lines around modules
    if (lay>0) {
      std::vector<std::vector<int> > nladder = { { 10, 16, 22 }, { 6, 14, 22, 32 } };
      int nlad = nladder[phase][lay-1];
      for (int xsign=-1; xsign<=1; xsign+=2) for (int ysign=-1; ysign<=1; ysign+=2) {
        float xlow  = xsign * (half_shift*0.5 );
        float xhigh = xsign * (half_shift*0.5 + 4 );
        float ylow  = ysign * (half_shift*0.5 + (phase==0)*0.5 );
        float yhigh = ysign * (half_shift*0.5 - (phase==0)*0.5 + nlad);
        // Outside box
        draw_line(xlow,  xhigh,  ylow,  ylow, 1); // bottom
        draw_line(xlow,  xhigh, yhigh, yhigh, 1); // top
        draw_line(xlow,   xlow,  ylow, yhigh, 1); // left
        draw_line(xhigh, xhigh,  ylow, yhigh, 1); // right
        // Inner Horizontal lines
        for (int lad=1; lad<nlad; ++lad) {
          float y = ysign * (lad + half_shift*0.5);
          draw_line(xlow, xhigh,  y,  y, 1);
        }
        for (int lad=1; lad<=nlad; ++lad) if (!(phase==0&&(lad==1||lad==nlad))) {
          float y = ysign * (lad + half_shift*0.5 - 0.5);
          draw_line(xlow, xhigh,  y,  y, 1, 3);
        }
        // Inner Vertical lines
        for (int mod=1; mod<4; ++mod) {
          float x = xsign * (mod + half_shift*0.5);
          draw_line(x, x,  ylow,  yhigh, 1);
        }
        // Make a BOX around ROC 0
        // Phase 0 - ladder +1 is always non-flipped
        // Phase 1 - ladder +1 is always     flipped
        if (mark_zero) {
          for (int mod=1; mod<=4; ++mod) for (int lad=1; lad<=nlad; ++lad) {
            bool flipped = ysign==1 ? lad%2==0 : lad%2==1;
            if (phase==1)  flipped = !flipped;
            int roc0_orientation = flipped ? -1 : 1;
            if (xsign==-1) roc0_orientation *= -1;
            if (ysign==-1) roc0_orientation *= -1;
            float x1 = xsign * (mod+half_shift*0.5);
            float x2 = xsign * (mod+half_shift*0.5 - 1./8);
            float y1 = ysign * (lad+half_shift*0.5-0.5);
            float y2 = ysign * (lad+half_shift*0.5-0.5 + roc0_orientation*1./2);
            if (!(phase==0&&(lad==1||lad==nlad)&&xsign==-1)) {
	      if (lay == 1 && xsign <= -1 ){
		float x1 = xsign * ((mod-1)+half_shift*0.5 );
		float x2 = xsign * ((mod-1)+half_shift*0.5 + 1./8);
		float y1 = ysign * (lad+half_shift*0.5-0.5 + roc0_orientation);
		float y2 = ysign * (lad+half_shift*0.5-0.5 + roc0_orientation*3./2);

		draw_line(x1, x2, y1, y1, 1);
		draw_line(x2, x2, y1, y2, 1);
	      }
	      else{
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
      for (int dsk=1, ndsk=2+(phase==1); dsk<=ndsk; ++dsk) {
        for (int xsign=-1; xsign<=1; xsign+=2) for (int ysign=-1; ysign<=1; ysign+=2) {
          if (phase==0) {
            int first_roc = 3, nbin = 16;
            for (int bld=1, nbld=12; bld<=nbld; ++bld) {
              // Horizontal lines
              for (int plq=1, nplq=7; plq<=nplq; ++plq) {
                float xlow  = xsign * (half_shift*0.5 + dsk - 1   + (first_roc-3+2*plq+(plq==1))/(float)nbin);
                float xhigh = xsign * (half_shift*0.5 + dsk - 1   + (first_roc-3+2*(plq+1)-(plq==7))/(float)nbin);
                float ylow  = ysign * (half_shift*0.5 + (bld-0.5) - (2+plq/2)*0.1);
                float yhigh = ysign * (half_shift*0.5 + (bld-0.5) + (2+plq/2)*0.1);
                draw_line(xlow,  xhigh,   ylow,  ylow, 1); // bottom
                draw_line(xlow,  xhigh,  yhigh, yhigh, 1); // top
              }
              // Vertical lines
              for (int plq=1, nplq=7+1; plq<=nplq; ++plq) {
                float x     = xsign * (half_shift*0.5 + dsk - 1   + (first_roc-3+2*plq+(plq==1)-(plq==8))/(float)nbin);
                float ylow  = ysign * (half_shift*0.5 + (bld-0.5) - (2+(plq-(plq==8))/2)*0.1);
                float yhigh = ysign * (half_shift*0.5 + (bld-0.5) + (2+(plq-(plq==8))/2)*0.1);
                draw_line(x,  x,  ylow,  yhigh, 1);
              }
              // Panel 2 has dashed mid-plane
              for (int plq=2, nplq=6; plq<=nplq; ++plq) if (plq%2==0) {
                float x     = xsign * (half_shift*0.5 + dsk - 1   + (first_roc-3+2*plq+(plq==1)-(plq==8)+1)/(float)nbin);
                float ylow  = ysign * (half_shift*0.5 + (bld-0.5) - (2+(plq-(plq==8))/2)*0.1);
                float yhigh = ysign * (half_shift*0.5 + (bld-0.5) + (2+(plq-(plq==8))/2)*0.1);
                draw_line(x,  x,  ylow,  yhigh, 1, 2);
              }
              // Make a BOX around ROC 0
              for (int plq=1, nplq=7; plq<=nplq; ++plq) {
                float x1 = xsign * (half_shift*0.5 + dsk - 1   + (first_roc-3+2*plq+(plq==1))/(float)nbin);
                float x2 = xsign * (half_shift*0.5 + dsk - 1   + (first_roc-3+2*plq+(plq==1)+1)/(float)nbin);
                int sign = xsign * ysign * ((plq%2) ? 1 : -1);
                float y1 = ysign * (half_shift*0.5 + (bld-0.5) + sign *(2+plq/2)*0.1);
                float y2 = ysign * (half_shift*0.5 + (bld-0.5) + sign *(  plq/2)*0.1);
                //draw_line(x1, x2, y1, y1, 1);
                draw_line(x1, x2, y2, y2, 1);
                //draw_line(x1, x1, y1, y2, 1);
                draw_line(x2, x2, y1, y2, 1);
              }
            }
          } else if (phase==1) {
            if (ring == 0) { // both
              for (int ring=1; ring<=2; ++ring) for (int bld=1, nbld=5+ring*6; bld<=nbld; ++bld) {
                float scale = (ring==1) ? 1.5 : 1;
                Color_t p1_color = 1, p2_color = 1;
                // Horizontal lines
                // Panel 2 has dashed mid-plane
                float x1      = xsign * (half_shift*0.5 + dsk - 1 + (ring-1)*0.5);
                float x2      = xsign * (half_shift*0.5 + dsk - 1 +  ring   *0.5);
                int sign = ysign;
                float y1      = ysign * (half_shift*0.5 - 0.5 + scale*bld + sign*0.5);
                //float yp1_mid = ysign * (half_shift*0.5 - 0.5 + scale*bld + sign*0.25);
                float y2      = ysign * (half_shift*0.5 - 0.5 + scale*bld);
                float yp2_mid = ysign * (half_shift*0.5 - 0.5 + scale*bld - sign*0.25);
                float y3      = ysign * (half_shift*0.5 - 0.5 + scale*bld - sign*0.5);
                draw_line(x1, x2, y1,      y1,      1, 1, p1_color);
                //draw_line(x1, x2, yp1_mid, yp1_mid, 1, 3);
                draw_line(x1, x2, y2,      y2,      1, 1, p1_color);
                draw_line(x1, x2, yp2_mid, yp2_mid, 1, 2);
                draw_line(x1, x2, y3,      y3,      1, 1, p2_color);
                // Vertical lines
                float x = xsign * (half_shift*0.5 + dsk - 1 + (ring-1)*0.5);
                draw_line(x,  x,  y1,  y2, 1, 1, p1_color);
                draw_line(x,  x,  y2,  y3, 1, 1, p2_color);
                if (ring==2) {
                  //draw_line(x,  x,  y2,  y3, 1, 1, p1_color);
                  x         = xsign * (half_shift*0.5 + dsk);
                  draw_line(x,  x,  y1,  y2, 1, 1, p1_color);
                  draw_line(x,  x,  y2,  y3, 1, 1, p2_color);
                }
                // Make a BOX around ROC 0
                x1          = xsign * (half_shift*0.5 + dsk - 1 + ring*0.5 - 1/16.);
                x2          = xsign * (half_shift*0.5 + dsk - 1 + ring*0.5);
                float y1_p1 = ysign * (half_shift*0.5 - 0.5 + scale*bld + sign*0.25);
                float y2_p1 = ysign * (half_shift*0.5 - 0.5 + scale*bld + sign*0.25 + xsign*ysign*0.25);
                draw_line(x1, x2, y1_p1, y1_p1, 1);
                //draw_line(x1, x2, y2_p1, y2_p1, 1);
                draw_line(x1, x1, y1_p1, y2_p1, 1);
                //draw_line(x2, x2, y1_p1, y2_p1, 1);
                float y1_p2 = ysign * (half_shift*0.5 - 0.5 + scale*bld - sign*0.25);
                float y2_p2 = ysign * (half_shift*0.5 - 0.5 + scale*bld - sign*0.25 - xsign*ysign*0.25);
                draw_line(x1, x2, y1_p2, y1_p2, 1);
                //draw_line(x1, x2, y2_p2, y2_p2, 1);
                draw_line(x1, x1, y1_p2, y2_p2, 1);
                //draw_line(x2, x2, y1_p2, y2_p2, 1);
              }
            } else { // only one ring, 1 or 2
              for (int bld=1, nbld=5+ring*6; bld<=nbld; ++bld) {
                Color_t p1_color = 1, p2_color = 1;
                // Horizontal lines
                // Panel 2 has dashed mid-plane
                float x1      = xsign * (half_shift*0.5 + dsk - 1);
                float x2      = xsign * (half_shift*0.5 + dsk);
                int sign = ysign;
                float y1      = ysign * (half_shift*0.5 - 0.5 + bld + sign*0.5);
                //float yp1_mid = ysign * (half_shift*0.5 - 0.5 + bld + sign*0.25);
                float y2      = ysign * (half_shift*0.5 - 0.5 + bld);
                float yp2_mid = ysign * (half_shift*0.5 - 0.5 + bld - sign*0.25);
                float y3      = ysign * (half_shift*0.5 - 0.5 + bld - sign*0.5);
                draw_line(x1, x2, y1,      y1,      1, 1, p1_color);
                //draw_line(x1, x2, yp1_mid, yp1_mid, 1, 3);
                draw_line(x1, x2, y2,      y2,      1, 1, p1_color);
                draw_line(x1, x2, yp2_mid, yp2_mid, 1, 2);
                draw_line(x1, x2, y3,      y3,      1, 1, p2_color);
                // Vertical lines
                float x = xsign * (half_shift*0.5 + dsk - 1);
                draw_line(x,  x,  y1,  y2, 1, 1, p1_color);
                draw_line(x,  x,  y2,  y3, 1, 1, p2_color);
                if (ring==2) {
                  //draw_line(x,  x,  y2,  y3, 1, 1, p1_color);
                  x         = xsign * (half_shift*0.5 + dsk);
                  draw_line(x,  x,  y1,  y2, 1, 1, p1_color);
                  draw_line(x,  x,  y2,  y3, 1, 1, p2_color);
                }
                // Make a BOX around ROC 0
                x1          = xsign * (half_shift*0.5 + dsk - 1/8.);
                x2          = xsign * (half_shift*0.5 + dsk);
                float y1_p1 = ysign * (half_shift*0.5 - 0.5 + bld + sign*0.25);
                float y2_p1 = ysign * (half_shift*0.5 - 0.5 + bld + sign*0.25 + xsign*ysign*0.25);
                draw_line(x1, x2, y1_p1, y1_p1, 1);
                //draw_line(x1, x2, y2_p1, y2_p1, 1);
                draw_line(x1, x1, y1_p1, y2_p1, 1);
                //draw_line(x2, x2, y1_p1, y2_p1, 1);
                float y1_p2 = ysign * (half_shift*0.5 - 0.5 + bld - sign*0.25);
                float y2_p2 = ysign * (half_shift*0.5 - 0.5 + bld - sign*0.25 - xsign*ysign*0.25);
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
      if (phase==0&&h->GetNbinsY()==250&&h->GetNbinsX()==80) {
        int nentries = h->GetEntries();
        for (int binx = 1; binx<=80; ++binx) {
          double sum = 0;
          for (int biny = 1; biny<=250; ++biny) {
            bool odd_nrocy = (binx-1<40) != (((binx-1)/4)%2);
            if (biny%2==odd_nrocy) sum+= h->GetBinContent(binx, biny);
            else {
              sum+= h->GetBinContent(binx, biny);
              if (sum) {
                h->SetBinContent(binx, biny, sum);
                h->SetBinContent(binx, biny-1, sum);
              }
              sum = 0;
            }
          }
        }
        h->SetEntries(nentries);
      }
    }
  }
};

static SiPixelMapsRenderPlugin instance;
