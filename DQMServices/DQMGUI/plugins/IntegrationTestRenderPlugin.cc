/*!
  \file IntegrationTestRenderPlugin
  \\ This render plugin is used only in Continuous Integration of DQMGUI application.
  \\ It does some basic modifications to TH1 and TH2 histograms which are then compared
  \\ to an expected image. This ensures output of render plugin doesn't change unexpectedly.
  \\ Based on SiPixelRenderPlugin
  \\
  \author Antanas Sinica
  \version $Revision: 1.0 $
  \date $Date: 2017/04/06 18:15:09 $
*/

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"

#include "TProfile2D.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TGraphPolar.h"
#include "TColor.h"
#include "TText.h"
#include "TLine.h"
#include "TGaxis.h"
#include <cassert>
#include <cctype>
#include <map>
#include <vector>

class IntegrationTestRenderPlugin : public DQMRenderPlugin
{
public:
  virtual bool applies( const VisDQMObject &o, const VisDQMImgInfo & )
    {
      if( o.name.find( "IntegrationTest/" ) != std::string::npos )
        return true;

      return false;
    }

  virtual void preDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo & )
    {
      c->cd();

      if( dynamic_cast<TH2*>( o.object ) )
      {
        preDrawTH2( c, o );
      }

      else if( dynamic_cast<TH1*>( o.object ) )
      {
        preDrawTH1( c, o );
      }
    }

  virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo & )
    {
      c->cd();
      putName(o);

      if( dynamic_cast<TH2*>( o.object ) )
      {
        postDrawTH2( c, o );
      }
      else if( dynamic_cast<TH1*>( o.object ) )
      {
        postDrawTH1( c, o );
      }
    }

private:
  TText *name_text = nullptr;

  void putName(VisDQMObject const& o) {
    // we cannot delete this immediately, so wait for the next call
    if (name_text) delete name_text;

    name_text = new TText(0.05,0.01, o.name.c_str());
    name_text->SetTextColor(kBlack);
    name_text->SetTextSize(0.02);
    name_text->SetNDC();
    name_text->Draw("same");
  }

  // simple recursive descent parser for the "Column(0,30,50,)/Other(0,3,5,)/Last"
  // format used to carry the positions where histograms where concatenated by
  // EXTEND to this plugin.
  template<typename Iterator>
  struct LabelMarkerParser {
    std::vector<std::vector<int>> markers; // output
    std::string text; // cleaned-up label

    static std::pair<bool, int> parse_int(Iterator& start, Iterator end) {
      int out = 0;
      if (start == end || !std::isdigit(*start)) return std::make_pair(false, out);
      while (start != end && std::isdigit(*start))
        out = out * 10 + (*start++ - '0');
      return std::make_pair(true, out);
    }

    static std::string parse_text(Iterator& start, Iterator end) {
      std::string out;
      while (start != end && *start != '(' && *start != '/') out.push_back(*start++);
      return out;
    }

    static std::pair<bool, std::vector<int>> parse_list(Iterator& start, Iterator end) {
      std::vector<int> out;
      if (start == end || *start != '(') return std::make_pair(false, out);
      ++start;
      for(;;) {
        auto num = parse_int(start, end);
        if (num.first) out.push_back(num.second);
        else break;
        if (start != end && *start == ',') ++start;
      }
      if (start == end || *start != ')') return std::make_pair(false, out);
      ++start;
      return std::make_pair(true, out);
    }

    static std::pair<bool, LabelMarkerParser> parse_label(Iterator start, Iterator end) {
      std::vector<std::vector<int>> out;
      std::string text;
      for(;;) {
        text += parse_text(start, end);
        auto list = parse_list(start, end);
        // we might fail parsing here but consume input, but this is acceptable.
        if (list.first) out.emplace_back(list.second);
        if (start == end)  return std::make_pair(true, LabelMarkerParser{out, text});
        if (*start != '/') return std::make_pair(false, LabelMarkerParser{out, text});
        ++start;
        text += "/";
        if (start == end)  return std::make_pair(true, LabelMarkerParser{out, text});
      }
    }
  };

  void putMarkers(TH1* obj)
    {
      TAxis* ax = obj->GetXaxis();
      auto label = std::string(ax->GetTitle());
      auto res = LabelMarkerParser<std::string::iterator>::parse_label(label.begin(), label.end());
      if (!res.first || res.second.markers.size() == 0)
        return; // parse failed, probably no markers

      std::string newlabel = res.second.text;
      std::vector<std::vector<int>>& markers = res.second.markers;

      auto ymax = obj->GetMaximum() * 0.5;
      auto ymin = 0;
      auto step = (ymax-ymin)*0.5 / markers.size();
      // we only get one set of values each, but have to draw a full hierarchy
      putMarkersRecursive(markers.begin(), markers.end(), 0, ymin, ymax, step);

      ax->SetTitle(newlabel.c_str());
    }

  template<typename Iterator> // Iterator into vector of vectors
  void putMarkersRecursive(Iterator begin, Iterator end, int offset, double ymin, double ymax, double step) {
    if (begin == end) return;
    for (auto mark : *begin) {
      auto pos = double(mark) + 0.5 + double(offset);
      TLine tl;
      tl.SetLineColor(4);
      tl.DrawLine(pos, ymin, pos, ymax);
      putMarkersRecursive(begin+1, end, offset + mark, ymin, ymax - step, step);
    }
  }

  void preDrawTH2( TCanvas *, const VisDQMObject &o )
    {
      TH2* obj = dynamic_cast<TH2*>( o.object );
      assert( obj );

      // This applies to all
      if( o.name.find( "IntegrationTestTH2" ) != std::string::npos ) {
        // styles
        gStyle->SetCanvasBorderMode( 0 );
        gStyle->SetPadBorderMode( 0 );
        gStyle->SetPadBorderSize( 0 );
        gStyle->SetOptStat( 0 );
        gStyle->SetPalette(1);
        obj->SetOption("colztext");
        obj->SetStats( kTRUE );
        gStyle->SetOptStat( 1111111 );

        // margins and logarithmic scale
        gPad->SetRightMargin(0.15);
        gPad->SetGrid();
        gPad->SetLeftMargin(0.3);
        if( obj->GetEntries() > 0. ) gPad->SetLogz(1);

        // label sizes
        TAxis* xa = obj->GetXaxis();
        TAxis* ya = obj->GetYaxis();
        xa->SetTitleOffset(0.7);
        xa->SetTitleSize(0.065);
        xa->SetLabelSize(0.065);
        ya->SetTitleOffset(0.75);
        ya->SetTitleSize(0.065);
        ya->SetLabelSize(0.065);

        return;
      }
    }

  void postDrawTH2( TCanvas * /*c*/, const VisDQMObject &o )
    {
      TH2* obj = dynamic_cast<TH2*>( o.object );
      assert( obj );
      // Add Th2 post-draw code here.
      TGaxis::SetMaxDigits(3);

   }


  void preDrawTH1( TCanvas *, const VisDQMObject &o )
    {
      TH1* obj = dynamic_cast<TH1*>( o.object );
      assert( obj );

      // This applies to all
      gStyle->SetOptStat(111);
      TAxis* xa = obj->GetXaxis();
      TAxis* ya = obj->GetYaxis();
      xa->SetTitleOffset(0.7);
      xa->SetTitleSize(0.04);
      xa->SetLabelSize(0.03);
      ya->SetTitleOffset(0.75);
      ya->SetTitleSize(0.04);
      ya->SetLabelSize(0.03);
      TGaxis::SetMaxDigits(3);

      // Always include 0
      if( obj->GetMinimum() > 0.) obj->SetMinimum(0.);

      // Different ranges
      if( o.name.find( "IntegrationTestTH1" ) != std::string::npos ) {
        obj->SetMinimum(-5.); obj->SetMaximum(40.);
      }
    }

  void postDrawTH1( TCanvas *, const VisDQMObject &o )
    {
      if (o.flags == 0) return;

      TH1* obj = dynamic_cast<TH1*>( o.object );
      assert( obj );

      // put EXTEND marker for phase1
      putMarkers(obj);

      // Upper/Lower limit decoration for SUMOFFs.
      if( o.name.find( "IntegrationTestTH1" ) != std::string::npos ) {
        TLine tl1; tl1.SetLineColor(4); tl1.DrawLine(1.,12.0,193.,12.0);
        TLine tl2; tl2.SetLineColor(4); tl2.DrawLine(1.,35.0,193.,35.0);
      }
    }

};

static IntegrationTestRenderPlugin instance;
