/**
    DQMInfoRenderPlugin.cc
    Purpose: Will inject additional render code for specific plots in DQM the
    DQM "info" folders.

    @author Broen van Besien
    Originally written by multiple people in CMS before 2011, based on render
    plugins from other subsystems. (Actual authors are not known.)
    Completely rewritten in 2017, when adding new features and solving multiple
    bugs, hence restoring old functionality that was lost for many years.
    The main addition when rewriting the entire class is the logical ordering
    of the sub-systems in the high voltage plot.
*/

#include <cassert>
#include <vector>
#include <sstream>
#include <boost/algorithm/string.hpp>

#include "DQM/DQMRenderPlugin.h"
#include "utils.h"
#include "TMath.h"
#include "TH2F.h"
#include "TLine.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TColor.h"
#include "TText.h"
#include "TROOT.h"
#include "TObjString.h"

class DQMInfoRenderPlugin : public DQMRenderPlugin {
 public:

  // Test if this render plugin should be used for the given histogram
  virtual bool applies(const VisDQMObject &o, const VisDQMImgInfo &) {
    if (o.name.find("Info/EventInfo/reportSummaryMap") != std::string::npos
     || o.name.find("Info/LhcInfo/") != std::string::npos
     || o.name.find("Info/ProvInfo/Taglist") != std::string::npos) {
      return true;
    } else {
      return false;
    }
  }


  // Implementation of preDraw: What is done before calling the standard draw
  virtual void preDraw (TCanvas * c, const VisDQMObject &o, const VisDQMImgInfo &, VisDQMRenderInfo &) {
    // Only continue if there is an actual histogram
    if (not o.object) {
      return;
    }
    // Reset canvas
    c->cd();
    // Depending on histogram path, call different functions
    if (o.name.find("Info/EventInfo/reportSummaryMap") != std::string::npos) {
      preDrawReportSummaryMap(o);
    } else if (o.name.find("Info/LhcInfo/") != std::string::npos) {
      preDrawLhcInfo(o);
      if (o.name.find("Info/LhcInfo/beamMode") != std::string::npos) {
        preDrawBeamMode(o);
      }
    } else if (o.name.find("Info/ProvInfo/Taglist") != std::string::npos) {
      preDrawTagList(o);
    }
  }


  // Implementation of postDraw: What is done after calling the standard draw
  // (e.g. you can draw some more stuff, like lines or text)
  virtual void postDraw( TCanvas *c, const VisDQMObject &o, const VisDQMImgInfo & ) {
    // Only continue if there is an actual histogram
    if (not o.object) {
      return;
    }
    // Reset canvas
    c->cd();
    // Depending on histogram path, call different functions
    if (o.name.find("Info/EventInfo/reportSummaryMap" ) != std::string::npos) {
      postDrawReportSummaryMap(o);
    }
    else if (o.name.find("Info/LhcInfo/beamMode") != std::string::npos) {
      postDrawBeamMode(o);
    }
    else if (o.name.find("Info/LhcInfo/lhcFill") != std::string::npos) {
      postDrawLhcFill(o);
    }
    else if (o.name.find("Info/LhcInfo/momentum") != std::string::npos) {
      postDrawMomentum(o);
    }
    else if (o.name.find("Info/ProvInfo/Taglist") != std::string::npos) {
      postDrawTagList();
    }
  }

 private:

  ////////////////////////////////////////////////////////////////////////////
  // ReportSummaryMap
  ////////////////////////////////////////////////////////////////////////////

  // Order of the lines in the ReportSummaryMap
  // For the definition, see the bottom of this file.
  static const std::vector<std::string> summaryMapOrder_;

  // Because we don't want to trigger the shifters on subdetectors that
  // are "not in" for a longer time, we want to hide them from the HV plot.
  // In practice, this need was created by ZDC and CASTOR not being in for
  // a longer period.
  // - Either we remove these subdetectors from the plot in CMSSW at the
  //   source where the plot is created.
  // - Or we just remove them when the plot is displayed in the GUI.
  // We decided to go for the second option, since it makes more sence to
  // keep all data related to the DTS bits available in the data.
  // Doing the "masking" in the renderplugin in the GUI also gives us a
  // lot more flexibility in terms of fast turning it on or of.
  //
  // Additionally, we also change the order of the sub-systems to
  // something that makes more sense and is more user friendly. The current
  // ordering in the plot is basically random, based on the old
  // implementation of the CMSSW application. Again, we keep the order in
  // histogram exactly the way it is, but we just change the order in the
  // render plugin.
  //
  // The difficulty is in the hiding of certain horizontal lines when the
  // plot is rendered. There is not really any option for that in root.
  // The only way is to actually create a new plot with different data.
  // We will first clone the plot "obj" to have the original data.
  // Then we will determine the dimensions of our updated plot.
  // And finally copy the data from the cloned plot back to our original
  // plot, with a new order and leaving out the unwanted subdetectors.
  void preDrawReportSummaryMap(const VisDQMObject &o) {
    // The original plot:
    TH2F* originalPlot = dynamic_cast<TH2F*>(o.object);
    // Clone the plot:
    TH2F myClonedPlot(*originalPlot);
    // And reset the original:
    originalPlot->Reset();
    fixInconsistentPlotLimits(originalPlot);

    // Set the range of the new plot to span up to the last filled bin of
    // the top row + an extra column we will add:
    int lastBinWithData = getLastBinWithData(myClonedPlot);
    int maxBinX = TMath::Max(lastBinWithData + 1, 2); // Leave at least 2 bins
    int maxBinY = myClonedPlot.GetNbinsY();

    // Now we basically copy the data from the cloned plot back into the
    // original one, in the order defined in summaryMapOrder_.
    int newMaxBinY = copyDataReordered(myClonedPlot, originalPlot, maxBinX, maxBinY);

    // We set the ranges to plot, for Y this is the _new_ range:
    originalPlot->GetXaxis()->SetRange(1,maxBinX);
    originalPlot->GetYaxis()->SetRange(1,newMaxBinY);
    gPad->SetGrid(1,1);
    gPad->SetLeftMargin(0.12);
    dqm::utils::reportSummaryMapPalette(originalPlot);
    // We hide the tickmarks on the vertical axis
    originalPlot->GetYaxis()->SetTickSize(0);
    originalPlot->SetStats(kFALSE);
    originalPlot->SetOption("col");
  }


  // If you show Lumis 1 to 5, your limits should be 0 to 5, not 1 to 6.
  // We fix that here.
  void fixInconsistentPlotLimits(TH2F* histogram) {
    int firstBin = histogram->GetXaxis()->GetFirst();
    double minX = histogram->GetXaxis()->GetXmin();
    int lastBin = histogram->GetXaxis()->GetLast();
    double maxX = histogram->GetXaxis()->GetXmax();
    if (firstBin == 1 and minX == 1. and (maxX == lastBin + 1)) {
      histogram->GetXaxis()->SetLimits(minX - 1, maxX - 1);
    }
  }


  // Scan from right to left to see which is the rightmost bin with data
  int getLastBinWithData(TH2F const& histogram) {
    int maxBinX = histogram.GetNbinsX();
    int maxBinY = histogram.GetNbinsY();
    for (int binX = maxBinX; binX > 0; --binX) {
      // For this scan, we look at the top line, maxBinY, corresponding to
      // "Valid" for online.
      // For offline there is no "Valid", but the scan wouldn't really matter
      // since it shows only existing lumisections anyway.
      Double_t binContent = histogram.GetBinContent(binX, maxBinY);
      if (binContent != 0 && binContent != -1) {
        return binX;
      }
    }
    // For offline, we often see the last column being all 0 over the entire
    // line. This is fake data. When we see that, we remove it.
    for (int binY = 1; binY <= maxBinY; ++binY) {
      Double_t binContent = histogram.GetBinContent(maxBinX, binY);
      if (binContent != 0 && binContent != -1) {
        return maxBinX; // Not dropping last column
        break;
      }
    }
    return maxBinX - 1; // Assuming all are zero, dropping last column
  }


  // Copy the data from the source plot into the target plot in the order
  // defined in summaryMapOrder_.
  int copyDataReordered(TH2F const& source, TH2F* target, int maxBinX, int maxBinY) {
    int binYNew = 0;
    // Run over the labels in the order in which we want them
    // (Actually reverse, since we fill from the bottom)
    for (auto order_iter(summaryMapOrder_.rbegin());
        order_iter != summaryMapOrder_.rend();
        ++order_iter) {
      std::string wanted_label(*order_iter);
      // Run over the available data
      for (int binYOrig = 1; binYOrig <= maxBinY; binYOrig++) {
        std::string current_label(source.GetYaxis()->GetBinLabel(binYOrig));
        boost::trim(current_label);
        if (current_label == wanted_label) {
          binYNew++;
          for (int binX = 1; binX < maxBinX; binX++) {
            // We go to 1 less than maxBinX, because, remember that we added
            // an extra empty column.
            // Just copy the data from one row to the other:
            target->SetBinContent(binX, binYNew, source.GetBinContent(binX, binYOrig));
          }
          // We make an extra white band to the right of the plot for better
          // visibility. (Was originally done in the code, but this belongs
          // more in the render plugin.)
          target->SetBinContent(maxBinX, binYNew, -1.);
          // Of course, we have to update the label as well
          target->GetYaxis()->SetBinLabel(binYNew, TString(current_label));
        }
      }
    }
    // Blank out the labels of the unused lines:
    for (int binYrest = binYNew + 1; binYrest <= maxBinY; binYrest++) {
      target->GetYaxis()->SetBinLabel(binYrest, "<blank>");
    }
    return binYNew; // basically, the new _last_ bin on the Y axis
  }


  void postDrawReportSummaryMap(const VisDQMObject &o) {
    // object should be TH2F histogram
    TH2F* histogram = dynamic_cast<TH2F*>( o.object );
    // We draw some extra lines between the different sub-detectors
    drawSubsystemSeparators(histogram);
  // We also print the beam mode on top of the histogram
    printBeamMode(histogram);
  }


  // We draw some extra lines between the different sub-detectors
  void drawSubsystemSeparators(TH2F* histogram) {
    // Define beginning and ending of line
    int begin = histogram->GetXaxis()->GetFirst() - 1;
    int end = histogram->GetXaxis()->GetLast();
    int lineY = 0;
    TLine line;
    // Run over the summaryMapOrder vector to find the <line> entries.
    // (Actually reverse, since we fill from the bottom)
    for (auto order_iter(summaryMapOrder_.rbegin());
        order_iter != summaryMapOrder_.rend();
        ++order_iter) {
      std::string label(*order_iter);
      // If we encounter a <*_line>, we draw a line
      if (label == "<thick_line>") {
        line.SetLineWidth(2);
        line.SetLineColor(1); //Black
        line.DrawLine(begin, lineY, end, lineY);
      }
      else if (label == "<thin_line>") {
        line.SetLineWidth(1);
        line.SetLineColor(12); //Dark gray
        line.DrawLine(begin, lineY, end, lineY);
      }
      else {
        // If we encounter an actual label, we move up
        // But only if the label actually matches to something
        int maxBinY = histogram->GetYaxis()->GetLast();
        for (int binY = 1; binY<= maxBinY; binY++) {
          std::string current_label(histogram->GetYaxis()->GetBinLabel(binY));
          if (current_label == label) {
            lineY++;
          }
        }
      }
    }
  }


  // We print the beam mode on top of the histogram
  void printBeamMode(TH2F* histogram) {
    if (isGivenModeDeclaredDuringRun("PhysDecl", histogram)) {
      // We got a physics declared, so we show this.
      printMainText("Physics Declared");
    } else if (isGivenModeDeclaredDuringRun("Stable B", histogram)) {
      // We got stable beams, but no physics declared...
      printMainText("Stable Beams");
      // ...so we mention this as a warning.
      printSecondaryText("(No Physics Declared)");
    } else if (isGivenModeDeclaredDuringRun("13 TeV", histogram)) {
      // We got 13 TeV, but no stable beams...
      printMainText("13 TeV");
      // ...so we mention this as a warning.
      printSecondaryText("(No Stable Beams)");
    }
    // Otherwise we show nothing. Doing cosmics or commissioning.
  }


  // Test if for the given beam mode, we find at least one bin that is active/1
  bool isGivenModeDeclaredDuringRun(std::string mode, TH2F* histogram) {
    int maxBinY = histogram->GetYaxis()->GetLast();
    int maxBinX = histogram->GetXaxis()->GetLast();
    // scan up/down to find the right line
    for (int binY = 1; binY <= maxBinY; binY++) {
      TString label = TString(histogram->GetYaxis()->GetBinLabel(binY));
      if (label == mode) { // bingo!
        // scan left/right
        for (int binX = 1; binX <= maxBinX; binX++) {
          if (histogram->GetBinContent(binX, binY) == 1) {
            return true;
          }
        }
        return false;
      }
    }
    return false;
  }


  void printMainText(const char* message) {
    TText text;
    text.SetTextAlign(22); // center/center
    text.SetTextColor(13);
    text.SetTextSize(0.085);
    text.DrawTextNDC(.5, .54, message);
  }


  // A bit smaller, a bit lower
  void printSecondaryText(const char* message) {
    TText text;
    text.SetTextAlign(22); // center/center
    text.SetTextColor(13);
    text.SetTextSize(0.06);
    text.DrawTextNDC(.5, .46, message);
  }

  ////////////////////////////////////////////////////////////////////////////
  // LhcInfo
  ////////////////////////////////////////////////////////////////////////////

  void preDrawLhcInfo(const VisDQMObject &o) {
    TH1F* histogram = dynamic_cast<TH1F*>(o.object);
    // We scan from right to left to see which is the rightmost bin with data
    int maxBinX = histogram->GetNbinsX();
    for (int binX = histogram->GetNbinsX(); binX > 0; --binX) {
      if (histogram->GetBinContent(binX) != 0) {
        // Set the range of the plot to span up to the last filled bin + an
        // extra column
        maxBinX = TMath::Max(binX + 1, 2);  // leave at least 2 bins
        break;
      }
    }
    // We limit the X range of the histogram to where there is data
    histogram->GetXaxis()->SetRange(1,maxBinX);
    histogram->SetStats( kFALSE );
    histogram->SetMinimum(-1.e-15);
  }


  void preDrawBeamMode(const VisDQMObject &o) {
    TH1F* histogram = dynamic_cast<TH1F*>(o.object);
    histogram->SetMaximum(22.);
    histogram->SetMinimum(0.);
    gPad->SetLeftMargin(0.15);
    gPad->SetGrid(1,1);
  }


  // Draw the last beam mode in large letters on top of the plot
  void postDrawBeamMode(const VisDQMObject &o) {
    TH1F* histogram = dynamic_cast<TH1F*>( o.object );
    // Retrieve the last known beam mode from the histogram
    TString label("");
    for (int i = histogram->GetNbinsX(); i > 0; --i) {
      if (histogram->GetBinContent(i) != 0) {
        int value = (int)histogram->GetBinContent(i);
        label = TString(histogram->GetYaxis()->GetBinLabel(value));
        break;
      }
    }
    // Draw the beam mode on top of the histogram
    if (label) {
      std::stringstream message;
      message << "Beam mode: " << label;
      TText text;
      text.SetTextAlign(22); // center/center
      text.SetTextColor(4);
      text.SetTextSize(0.06);
      text.DrawTextNDC(.5, .6, message.str().c_str());
    }
  }


  // Draw the LHC fill number in large letters on top of the plot
  void postDrawLhcFill(const VisDQMObject &o) {
    TH1F* histogram = dynamic_cast<TH1F*>(o.object);
    // Retrieve the fill number from the histogram
    int fill_number = 0;
    for (int i = histogram->GetNbinsX(); i > 0; --i) {
      if (histogram->GetBinContent(i) != 0 ) {
        fill_number = (int)histogram->GetBinContent(i);
        break;
      }
    }
    // Draw the number on top of the histogram
    if (fill_number != 0 && fill_number < 1000000) // Optimistic about the future of CERN
    {
      std::stringstream message;
      message << "LHC Fill: " << fill_number;
      TText text;
      text.SetTextAlign(22); // center/center
      text.SetTextColor(4);
      text.SetTextSize(0.06);
      text.DrawTextNDC(.5, .5, message.str().c_str());
    }
  }


  // Draw the momentum in large letters on top of the plot
  void postDrawMomentum(const VisDQMObject &o) {
    TH1F* histogram = dynamic_cast<TH1F*>(o.object);
    // Retrieve momentum from the histogram
    int momentum = 0;
    for (int i = histogram->GetNbinsX(); i > 0; --i) {
      if (histogram->GetBinContent(i) != 0) {
        momentum = (int)histogram->GetBinContent(i);
        break;
      }
    }
    // Draw the momentum on top of the histogram
    if (momentum != 0 && momentum < 10000000) // We're optimistic about future LHC
    {
      std::stringstream message;
      message << "Beam Energy: " << momentum << " GeV";
      TText text;
      text.SetTextAlign(22); // center/center
      text.SetTextColor(4);
      text.SetTextSize(0.06);
      text.DrawTextNDC(.5, .5, message.str().c_str());
    }
  }

  ////////////////////////////////////////////////////////////////////////////
  // TagList
  ////////////////////////////////////////////////////////////////////////////

  // The TagList was something used during Run1 to show a <tag> version for
  // each DQM client. This data is still there if you look at very old runs,
  // so we keep it here for backward compatibility
  // In the predraw it extracts a string and then resets the plot to show
  // nothing.
  // In the postdraw it then reuses that string.

  std::string taglist;

  void preDrawTagList(const VisDQMObject &o) {
    TObjString* s = dynamic_cast<TObjString*>(o.object);
    taglist = s->String();
    s->SetString("");
  }


  void postDrawTagList() {
    // This code is dirty, but I'm leaving it as-is, given that is legacy.
    int max = 0;
    int length = 0;
    std::vector<std::string> tltable;
    tltable.push_back("");
    for (std::string::iterator it=taglist.begin(); it < taglist.end(); it++) {
      if (*it == ';') {
        tltable.push_back("");
        length=0;
        continue;
      }
      if (*it == ':') {
        if (max < length) {
          max = length;
        }
        length *= -1;
      }
      tltable.back().append(1,*it);
      if (length >=0) {
        length++;
      }
    }
    if (tltable.back().empty()) {
      tltable.pop_back();
    }
    int numRows = (int) tltable.size();
    float rowHight = 1.0 / numRows;
    if (rowHight >= 0.2) rowHight = 0.08;
    TText tt;
    tt.SetTextSize(rowHight);
    tt.SetTextFont(102);
    tt.SetTextAlign(13);
    for (int i = 0; i < (int) tltable.size(); i++) {
      int cpos = tltable[i].find(':');
      for (int j=0;j < max-cpos && cpos!=(int)std::string::npos; j++) {
        tltable[i].insert(cpos+j,1,' ');
      }
      tt.DrawText(0.01,1.0-(rowHight*i),tltable[i].c_str());
    }
  }
};


// Here we define the list and order of all the sub-systems we want to
// include in the ReportSummaryMap. It also includes some lines like
// "Stable beams", "13 TeV".
// The labels should correspond to the labels as they are set in the
// CMMSW application.
// <thick_line> and <thin_line> means to draw a horizontal line to
// group the different components of the subsystems together.
// To hide certain sub-systems (like CASTOR or ZDC) just comment those
// lines away.
const std::vector<std::string> DQMInfoRenderPlugin::summaryMapOrder_({
  "13 TeV",
  "Stable B",
  "PhysDecl",
  "<thick_line>",
  "BPIX",
  "FPIX",
  "<thin_line>",
  "TIBTID",
  "TOB",
  "TECm",
  "TECp",
  "<thin_line>",
  "EB-",
  "EB+",
  "EE-",
  "EE+",
  "ES-",
  "ES+",
  "<thin_line>",
  "HBHEa",
  "HBHEb",
  "HBHEc",
  "HO",
  "HF",
  "<thin_line>",
  "DT0",
  "DT-",
  "DT+",
  "<thin_line>",
  "CSC-",
  "CSC+",
  "<thin_line>",
  "RPC",
  "<thin_line>",
  "CT-PPS", // Simply not implemented yet
            // Will need to be implemented with this label in the CMSSW
            // DTS data structure and DQM info application first.
  //"<thin_line>",
  //"ZDC",
  //"<thin_line>",
  //"CASTOR",
  "<thick_line>",
  "Valid",
});

static DQMInfoRenderPlugin instance;
