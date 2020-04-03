/*!
  \file SiPixelPhase1OnlineRenderPlugin
  \brief RenderPlugin for histograms showing some history in online

This Render Plugin was originally desinged for Pixel timing scans. It
operates on plots that have a time dimension (in a unit called OnlineBlocks,
a configurable amount of Lumisections, usually around 10).

The output is a set of overlaid, normalized histograms, that can be compared to
each other to figure which setting gives the best distribution.

The input is a 2D histogram, that has the distribution to be monitored on the
x-axis and time on the y-axis. As a 2D plot this is not really readable, that
is why this renderplugin turns each (non-0) bin on the y-axis into a 1D
histogram.

Usage:
  1. Make sure the 2D plot has time on the y-axis and contains "OnlineBlock" in
     the name.
     A non-trivial detail is that the block size of the OnlineBlocks (which is
     configurable on the CMSSW side) is passed as the min-value of the y-axis.

  2. This Render Plugin applies and shows lines for all non-0 time slices. This
     is very hard to read for more than 10 lines, since the color coding gets
     ambiguous then.

  3. Set a range on the z-Axis (!) using the "customize" box. You can limit the
     plot to some LS range there. x- and y-axis ranges work as well.

  4. Comparison to a reference is not possible for now (except "on side"), but
     by design, this plot is its own reference (comparison over time).

Future possibilities: This style of rendering could also be useful for other
purposes, e.g. to overlay trend plots (time on x-axis) for different partitions
(layer/disk on y-axis). This requieres a smarter applies-check and more fancy
logic for the Legend.

  \author Marcel Schneider
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TLegend.h"
#include <cassert>
#include <string>

using namespace std;

class SiPixelPhase1OnlineRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies( const VisDQMObject & o, const VisDQMImgInfo & )
    {
      if( ( o.name.find( "PixelPhase1" ) != std::string::npos)
            && o.object && o.name.find("OnlineBlock") != std::string::npos ){
        return true;
      } else {
        return false;
      }
    }

  // Here, we disable the rendering of the DQM GUI, using the AXIS draw option.
  virtual void preDraw( TCanvas * canvas, const VisDQMObject & o, const VisDQMImgInfo & , VisDQMRenderInfo & renderInfo)
    {
      canvas->cd();
      TH2* obj = dynamic_cast<TH2*>( o.object );

      obj->SetStats(1);
      gStyle->SetOptStat(111);
      renderInfo.drawOptions = "AXIS";
    }

  // Here, we render our own 1D-view. Some tricks are needed to undo settings that the GUI did.
  // We also at a Legend, for which the LS ranges of the OnlineBlocks are reconstructed.
  virtual void postDraw( TCanvas *canvas, const VisDQMObject &o, const VisDQMImgInfo &ri )
    {
      canvas->cd();
      TH2* obj = dynamic_cast<TH2*>( o.object );
      if(!obj) return;

      // Axis range is not shown anywhere, min abused for block size
      int blocksize = int(obj->GetYaxis()->GetXmin()+0.5);

      TLegend* leg = new TLegend(0.75,0.3,0.9,0.95);
      leg->SetHeader("LS Range");
      leg->SetBit(kCanDelete);

      double ref = obj->GetEntries();
      double max = 0;

      int n_color = 0;

      int lower = std::isnan(ri.zaxis.min) ? 1 : int(ri.zaxis.min/blocksize+1);
      int upper = std::isnan(ri.zaxis.max) ? obj->GetNbinsY() : int(ri.zaxis.max/blocksize);
      if (lower < 1 || lower > obj->GetNbinsY()) lower = 1;
      if (upper < 1 || upper > obj->GetNbinsY()) upper = obj->GetNbinsY();

      for (int i = lower; i <= upper; i++) {
        auto name = std::string(obj->GetName()) + "_" + std::to_string(i);
        TH1* h = new TH1F(name.c_str(), "", obj->GetNbinsX(),
                        obj->GetXaxis()->GetXmin(), obj->GetXaxis()->GetXmax());
        h->SetBit(kCanDelete);
        double entries = 0;
        int nonzerobins = 0;
        for (int x = 1; x < obj->GetNbinsX(); x++) {
          if (obj->GetBinContent(x, i) > 0) nonzerobins++;
          entries += obj->GetBinContent(x, i);
          h->SetBinContent(x, obj->GetBinContent(x, i));
        }

        // suppress single-bin distributions for non-zero-suppressed NDigis etc.
        if (nonzerobins <= 1) {
	  delete h;
	  continue;
	}
        h->Scale(ref/entries);

        n_color++;
        if (n_color == 10) n_color = 15; // skip low saturation stuff in between.

        h->SetLineColor(n_color);
	h->Draw("SAME");
        // i is bins, so 1 is 1st block = 0 to blocksize-1
        leg->AddEntry(h,(std::to_string((i-1)*blocksize) + "-" + std::to_string((i*blocksize)-1)).c_str(),"l");
        max = std::max(max, h->GetMaximum());
      }

      double ymin = std::isnan(ri.yaxis.min) ? 0 : ri.yaxis.min;
      double ymax = std::isnan(ri.yaxis.max) ? max*1.05 : ri.yaxis.max;

      obj->GetYaxis()->Set(1, ymin, ymax);
      obj->GetYaxis()->SetRange(0,0); // ROOT magic: unset range, use binning min/max set above
      obj->GetYaxis()->SetTitle("");
      leg->Draw();
    }

private:
};

static SiPixelPhase1OnlineRenderPlugin instance;
