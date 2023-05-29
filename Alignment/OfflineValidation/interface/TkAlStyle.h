// Common style file for TkAl public plots
// --------------------------------------------------------------
//
// Defines a set of functions to define the plotting style and
// provide commonly used style objects such as histogram title
// and legends. Can be used in compiled code and uncompiled scripts.
//
// Always call TkAlStyle::set(PublicationStatus) in the beginning
// of your plotting script to adjust the gStyle options. Also,
// the behaviour of some other methods of this class depend on
// this, e.g. the histogram title displays "CMS preliminary" etc.
// depending on the PublicationStatus. Call TkAlStyle::set before
// declaring any histograms (or call TH1::UseCurrentStyle()) to
// make sure the style settings are used.
// --------------------------------------------------------------

#ifndef ALIGNMENT_OFFLINEVALIDATION_TKAL_STYLE_H
#define ALIGNMENT_OFFLINEVALIDATION_TKAL_STYLE_H

#include "TColor.h"
#include "TError.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TString.h"
#include "TStyle.h"
#include <iostream>
#include <string>
#include <algorithm>

// Publication status: determines what is plotted in title
enum PublicationStatus {
  NO_STATUS,
  INTERNAL,
  INTERNAL_SIMULATION,
  PRELIMINARY,
  PUBLIC,
  SIMULATION,
  UNPUBLISHED,
  CUSTOM
};

// Data era: determines labels of data-taking periods, e.g. CRUZET
enum Era { NONE, CRUZET15, CRAFT15, COLL0T15 };

class TkAlStyle {
public:
  // Adjusts the gStyle settings and store the PublicationStatus
  static void set(const PublicationStatus status,
                  const Era era = NONE,
                  const TString customTitle = "",
                  const TString customRightTitle = "");
  static void set(const TString customTitle);
  static PublicationStatus status() { return publicationStatus_; }

  // Draws a title "<CMS label> 2015" on the current pad
  // dependending on the PublicationStatus
  //  INTERNAL    : no extra label (intended for AN-only plots with data)
  //  INTERNAL    : show "Simulation" label (intended for AN-only plots, no "CMS")
  //  PRELIMINARY : show "CMS preliminary 2015" label
  //  PUBLIC      : show "CMS 2015" label
  //  SIMULATION  : show "CMS Simulation" label
  //  UNPUBLISHED : show "CMS (unpublished)" label (intended for additional material on TWiki)
  // Note that this method does not allow for easy memory
  // handling. For that, use standardTitle().
  static void drawStandardTitle() {
    standardTitle()->Draw("same");
    standardRightTitle()->Draw("same");
  }
  static void drawStandardTitle(const Era era) {
    standardTitle()->Draw("same");
    standardRightTitle(era)->Draw("same");
  }
  static PublicationStatus toStatus(std::string _status);

  // Returns a TPaveText object that fits as a histogram title
  // with the current pad dimensions.
  // It has the same text as described in drawStandardTitle().
  // The idea of this method is that one has control over the
  // TPaveText object and can do proper memory handling.
  static TPaveText* standardTitle(PublicationStatus status) { return title(header(status)); }
  static TPaveText* standardTitle() { return standardTitle(publicationStatus_); }

  static TPaveText* standardRightTitle(const Era era) { return righttitle(rightheader(era)); }
  static TPaveText* standardRightTitle() { return standardRightTitle(era_); }

  // Returns a TPaveText object that fits as a histogram title
  // with the current pad dimensions and displays the specified text txt.
  static TPaveText* customTitle(const TString& txt) { return title(txt); }
  static TPaveText* customRightTitle(const TString& txt) { return righttitle(txt); }

  static TString legendheader;
  static TString legendoptions;
  static double textSize;
  // Returns a TLegend object that fits into the top-right corner
  // of the current pad. Its width, relative to the pad size (without
  // margins), can be specified. Its height is optimized for nEntries
  // entries.
  static TLegend* legend(const int nEntries, const double relWidth = 0.5) { return legendTR(nEntries, relWidth); }
  static TLegend* legend(TString position, const int nEntries, const double relWidth = 0.5) {
    position.ToLower();
    if (!(position.Contains("top") || position.Contains("bottom")))
      position += "top";
    if (!(position.Contains("left") || position.Contains("right")))
      position += "right";
    TLegend* leg = nullptr;
    if (position.Contains("top") && position.Contains("right")) {
      leg = legendTR(nEntries, relWidth);
    } else if (position.Contains("top") && position.Contains("left")) {
      leg = legendTL(nEntries, relWidth);
    } else if (position.Contains("bottom") && position.Contains("right")) {
      leg = legendBR(nEntries, relWidth);
    } else if (position.Contains("bottom") && position.Contains("left")) {
      leg = legendBL(nEntries, relWidth);
    } else {
      leg = legendTR(nEntries, relWidth);
    }

    return leg;
  }
  // Same but explicitly state position on pad
  static TLegend* legendTL(const int nEntries, const double relWidth = 0.5) {
    return legend(nEntries, relWidth, true, true);
  }
  static TLegend* legendTR(const int nEntries, const double relWidth = 0.5) {
    return legend(nEntries, relWidth, false, true);
  }
  static TLegend* legendBL(const int nEntries, const double relWidth = 0.5) {
    return legend(nEntries, relWidth, true, false);
  }
  static TLegend* legendBR(const int nEntries, const double relWidth = 0.5) {
    return legend(nEntries, relWidth, false, false);
  }

  // Returns a TPaveText object that fits into the top-right corner
  // of the current pad and that can be used for additional labels.
  // Its width, relative to the pad size (without margins), can be
  // specified. Its height is optimized for nEntries entries.
  static TPaveText* label(const int nEntries, const double relWidth = 0.5) { return labelTR(nEntries, relWidth); }

  static TPaveText* label(TString position, const int nEntries, const double relWidth = 0.5) {
    position.ToLower();
    if (!(position.Contains("top") || position.Contains("bottom")))
      position += "top";
    if (!(position.Contains("left") || position.Contains("right")))
      position += "right";
    TPaveText* label = nullptr;
    if (position.Contains("top") && position.Contains("right")) {
      label = labelTR(nEntries, relWidth);
    } else if (position.Contains("top") && position.Contains("left")) {
      label = labelTL(nEntries, relWidth);
    } else if (position.Contains("bottom") && position.Contains("right")) {
      label = labelBR(nEntries, relWidth);
    } else if (position.Contains("bottom") && position.Contains("left")) {
      label = labelBL(nEntries, relWidth);
    } else {
      label = labelTR(nEntries, relWidth);
    }

    return label;
  }

  // Same but explicitly state position on pad
  static TPaveText* labelTL(const int nEntries, const double relWidth = 0.5) {
    return label(nEntries, relWidth, true, true);
  }
  static TPaveText* labelTR(const int nEntries, const double relWidth = 0.5) {
    return label(nEntries, relWidth, false, true);
  }
  static TPaveText* labelBL(const int nEntries, const double relWidth = 0.5) {
    return label(nEntries, relWidth, true, false);
  }
  static TPaveText* labelBR(const int nEntries, const double relWidth = 0.5) {
    return label(nEntries, relWidth, false, false);
  }

  static double lineHeight() { return lineHeight_; }

private:
  static PublicationStatus publicationStatus_;
  static Era era_;
  static TString customTitle_;
  static TString customRightTitle_;
  static double lineHeight_;
  static double margin_;

  // creates a title
  static TString applyCMS(const TString& txt);
  static TPaveText* title(const TString& txt);
  static TPaveText* righttitle(const TString& txt);

  // returns the standard-title (CMS label 2015) depending
  // on the PublicationStatus
  static TString header(const PublicationStatus status);
  static TString rightheader(const Era era);

  // NDC coordinates for TPave, TLegend,...
  static void setXCoordinatesL(const double relWidth, double& x0, double& x1);
  static void setXCoordinatesR(const double relWidth, double& x0, double& x1);
  static void setYCoordinatesT(const int nEntries, double& y0, double& y1);
  static void setYCoordinatesB(const int nEntries, double& y0, double& y1);

  static TLegend* legend(const int nEntries, const double relWidth, const bool left, const bool top);
  static TPaveText* label(const int nEntries, const double relWidth, const bool leftt, const bool top);
};

#endif
