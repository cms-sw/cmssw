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

#ifndef TKAL_STYLE_CC
#define TKAL_STYLE_CC

#include "TColor.h"
#include "TError.h"
#include "TLegend.h"
#include "TPaveText.h"
#include "TString.h"
#include "TStyle.h"
#include <iostream>


// Publication status: determines what is plotted in title
enum PublicationStatus { NO_STATUS, INTERNAL, INTERNAL_SIMULATION, PRELIMINARY, PUBLIC, SIMULATION, UNPUBLISHED, CUSTOM };
TString toTString(const PublicationStatus status) {
  TString str = "";
  if(      status == NO_STATUS )           str = "Status not set yet!";
  else if( status == INTERNAL )            str = "internal";
  else if( status == INTERNAL_SIMULATION ) str = "simulation (internal)";
  else if( status == PRELIMINARY )         str = "preliminary";
  else if( status == PUBLIC      )         str = "public";
  else if( status == SIMULATION  )         str = "simulation (public)";
  else if( status == UNPUBLISHED )         str = "unpublished";
  else if( status == CUSTOM      )         str = "custom title set";

  return str;
}


// Data era: determines labels of data-taking periods, e.g. CRUZET
enum Era { NONE, CRUZET15, CRAFT15, COLL0T15 };
static TString toTString(const Era era) {
    TString str = "";
    if(      era == CRUZET15 ) str = "0T cosmic ray data 2015";
    else if( era == CRAFT15  ) str = "3.8T cosmic ray data 2015";
    else if( era == COLL0T15 ) str = "0T collision data 2015";

    return str;
}


class TkAlStyle {
public:
  // Adjusts the gStyle settings and store the PublicationStatus
  static void set(const PublicationStatus status, const Era era = NONE, const TString customTitle = "", const TString customRightTitle = "");
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
  static void drawStandardTitle() { standardTitle()->Draw("same"); standardRightTitle()->Draw("same"); }
  static void drawStandardTitle(const Era era) { standardTitle()->Draw("same"); standardRightTitle(era)->Draw("same"); }

  // Returns a TPaveText object that fits as a histogram title
  // with the current pad dimensions.
  // It has the same text as described in drawStandardTitle().
  // The idea of this method is that one has control over the
  // TPaveText object and can do proper memory handling.
  static TPaveText* standardTitle(PublicationStatus status) {
    return title(header(status));
  }
  static TPaveText* standardTitle() {
    return standardTitle(publicationStatus_);
  }

  static TPaveText* standardRightTitle(const Era era) {
    return righttitle(rightheader(era));
  }
  static TPaveText* standardRightTitle() {
    return standardRightTitle(era_);
  }

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
  static TLegend* legend(const int nEntries, const double relWidth=0.5) {
    return legendTR(nEntries,relWidth);
  }
  static TLegend* legend(TString position, const int nEntries, const double relWidth=0.5) {
    position.ToLower();
    if( !( position.Contains("top") || position.Contains("bottom") ) )
      position += "top";
    if( !( position.Contains("left") || position.Contains("right") ) )
      position += "right";
    TLegend* leg = 0;
    if(        position.Contains("top")    && position.Contains("right") ) {
      leg = legendTR(nEntries,relWidth);
    } else if( position.Contains("top")    && position.Contains("left")  ) {
      leg = legendTL(nEntries,relWidth);
    } else if( position.Contains("bottom") && position.Contains("right") ) {
      leg = legendBR(nEntries,relWidth);
    } else if( position.Contains("bottom") && position.Contains("left")  ) {
      leg = legendBL(nEntries,relWidth);
    } else {
      leg = legendTR(nEntries,relWidth);
    }

    return leg;
  }
  // Same but explicitly state position on pad
  static TLegend* legendTL(const int nEntries, const double relWidth=0.5) {
    return legend(nEntries,relWidth,true,true);
  }
  static TLegend* legendTR(const int nEntries, const double relWidth=0.5) {
    return legend(nEntries,relWidth,false,true);
  }
  static TLegend* legendBL(const int nEntries, const double relWidth=0.5) {
    return legend(nEntries,relWidth,true,false);
  }
  static TLegend* legendBR(const int nEntries, const double relWidth=0.5) {
    return legend(nEntries,relWidth,false,false);
  }


  // Returns a TPaveText object that fits into the top-right corner
  // of the current pad and that can be used for additional labels.
  // Its width, relative to the pad size (without margins), can be
  // specified. Its height is optimized for nEntries entries.
  static TPaveText* label(const int nEntries, const double relWidth=0.5) {
    return labelTR(nEntries,relWidth);
  }

  static TPaveText* label(TString position, const int nEntries, const double relWidth=0.5) {
    position.ToLower();
    if( !( position.Contains("top") || position.Contains("bottom") ) )
      position += "top";
    if( !( position.Contains("left") || position.Contains("right") ) )
      position += "right";
    TPaveText* label = 0;
    if(        position.Contains("top")    && position.Contains("right") ) {
      label = labelTR(nEntries,relWidth);
    } else if( position.Contains("top")    && position.Contains("left")  ) {
      label = labelTL(nEntries,relWidth);
    } else if( position.Contains("bottom") && position.Contains("right") ) {
      label = labelBR(nEntries,relWidth);
    } else if( position.Contains("bottom") && position.Contains("left")  ) {
      label = labelBL(nEntries,relWidth);
    } else {
      label = labelTR(nEntries,relWidth);
    }

    return label;
  }

  // Same but explicitly state position on pad
  static TPaveText* labelTL(const int nEntries, const double relWidth=0.5) {
    return label(nEntries,relWidth,true,true);
  }
  static TPaveText* labelTR(const int nEntries, const double relWidth=0.5) {
    return label(nEntries,relWidth,false,true);
  }
  static TPaveText* labelBL(const int nEntries, const double relWidth=0.5) {
    return label(nEntries,relWidth,true,false);
  }
  static TPaveText* labelBR(const int nEntries, const double relWidth=0.5) {
    return label(nEntries,relWidth,false,false);
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

PublicationStatus TkAlStyle::publicationStatus_ = NO_STATUS;
Era TkAlStyle::era_ = NONE;
TString TkAlStyle::legendheader = "";
TString TkAlStyle::legendoptions = "all";
TString TkAlStyle::customTitle_ = "";
TString TkAlStyle::customRightTitle_ = "";
double TkAlStyle::lineHeight_ = 0.042;
double TkAlStyle::margin_ = 0.04;
double TkAlStyle::textSize = 0.035;


// --------------------------------------------------------------
void TkAlStyle::setXCoordinatesL(const double relWidth, double& x0, double& x1) {
  x0 = gStyle->GetPadLeftMargin()+margin_;
  x1 = x0 + relWidth*(1.-gStyle->GetPadLeftMargin()-gStyle->GetPadRightMargin()-2.*margin_);
}


// --------------------------------------------------------------
void TkAlStyle::setXCoordinatesR(const double relWidth, double& x0, double& x1) {
  x0 = 1.-gStyle->GetPadRightMargin()-margin_-relWidth*(1.-gStyle->GetPadLeftMargin()-gStyle->GetPadRightMargin()-2.*margin_);
  x1 = 1.-gStyle->GetPadRightMargin()-margin_;
}


// --------------------------------------------------------------
void TkAlStyle::setYCoordinatesT(const int nEntries, double& y0, double& y1) {
  y1 = 1.-gStyle->GetPadTopMargin()-margin_;
  y0 = y1-nEntries*lineHeight_;
}


// --------------------------------------------------------------
void TkAlStyle::setYCoordinatesB(const int nEntries, double& y0, double& y1) {
  y1 = gStyle->GetPadBottomMargin()+margin_;
  y0 = y1+nEntries*lineHeight_;
}


// --------------------------------------------------------------
TLegend* TkAlStyle::legend(const int nEntries, const double relWidth, const bool left, const bool top) {
  double x0 = 0.;
  double x1 = 0.;
  double y0 = 0.;
  double y1 = 0.;
  bool hasheader = (TkAlStyle::legendheader != "");
  if( left ) setXCoordinatesL(relWidth,          x0,x1);
  else       setXCoordinatesR(relWidth,          x0,x1);
  if( top  ) setYCoordinatesT(nEntries+hasheader,y0,y1);
  else       setYCoordinatesB(nEntries+hasheader,y0,y1);

  TLegend* leg = new TLegend(x0,y0,x1,y1);
  leg->SetBorderSize(0);
  leg->SetFillColor(0);
  leg->SetFillStyle(0);
  leg->SetTextFont(42);
  leg->SetTextSize(textSize);
  if (hasheader) leg->SetHeader(TkAlStyle::legendheader);

  return leg;
}


// --------------------------------------------------------------
TPaveText* TkAlStyle::label(const int nEntries, const double relWidth, const bool left, const bool top) {
  double x0 = 0.;
  double x1 = 0.;
  double y0 = 0.;
  double y1 = 0.;
  if( left ) setXCoordinatesL(relWidth,x0,x1);
  else       setXCoordinatesR(relWidth,x0,x1);
  if( top  ) setYCoordinatesT(nEntries,y0,y1);
  else       setYCoordinatesB(nEntries,y0,y1);

  TPaveText* label = new TPaveText(x0,y0,x1,y1,"NDC");
  label->SetBorderSize(0);
  label->SetFillColor(0);
  label->SetFillStyle(0);
  label->SetTextFont(42);
  label->SetTextAlign(12);	// left adjusted and vertically centered
  label->SetTextSize(textSize);
  label->SetMargin(0.);

  return label;
}

// --------------------------------------------------------------
//unfortunately no #definecommand in TLatex...
//#CMS{text} gives CMS in big bold, text in italics
//#CMS with no "argument" just gives CMS in big bold
//#noCMS{text} gives text in italics
TString TkAlStyle::applyCMS(const TString& txt) {
  TString newtxt = txt;
  newtxt.ReplaceAll("#CMS{","#scale[1.4]{#font[61]{CMS}} #font[52]{");
  newtxt.ReplaceAll("#noCMS{","#font[52]{");
  newtxt.ReplaceAll("#CMS","#scale[1.4]{#font[61]{CMS}}");
  return newtxt;
}


// --------------------------------------------------------------
TPaveText* TkAlStyle::title(const TString& txt) {
  double x0 = gStyle->GetPadLeftMargin();
  double x1 = 1.-gStyle->GetPadRightMargin();
  double y0 = 1.-gStyle->GetPadTopMargin();
  double y1 = 1.;
  if (txt.Contains("#CMS")) y0 += .02;
  TPaveText* theTitle = new TPaveText(x0,y0,x1,y1,"NDC");
  theTitle->SetBorderSize(0);
  theTitle->SetFillColor(10);
  theTitle->SetFillStyle(0);
  theTitle->SetTextFont(42);
  theTitle->SetTextAlign(13);	// left bottom adjusted
  theTitle->SetTextSize(0.038);
  theTitle->SetMargin(0.);
  theTitle->AddText(applyCMS(txt));

  return theTitle;
}


// --------------------------------------------------------------
TPaveText* TkAlStyle::righttitle(const TString& txt) {
  TString newtxt = applyCMS(txt);
  double x0 = gStyle->GetPadLeftMargin();
  double x1 = 1.-gStyle->GetPadRightMargin();
  double y0 = 1.-gStyle->GetPadTopMargin();
  double y1 = 1.;
  TPaveText* theTitle = new TPaveText(x0,y0,x1,y1,"NDC");
  theTitle->SetBorderSize(0);
  theTitle->SetFillColor(10);
  theTitle->SetFillStyle(0);
  theTitle->SetTextFont(42);
  theTitle->SetTextAlign(33);	// right bottom adjusted
  theTitle->SetTextSize(0.038);
  theTitle->SetMargin(0.);
  theTitle->AddText(newtxt);
  
  return theTitle;
}


// --------------------------------------------------------------
TString TkAlStyle::header(const PublicationStatus status) {
  TString txt;
  if( status == NO_STATUS ) {
    std::cout << "Status not set yet!  Can't draw the title!" << std::endl;
  } else if( status == INTERNAL_SIMULATION ) {
    txt = "#noCMS{Simulation}";
  } else if( status == PRELIMINARY ) {
    txt = "#CMS{Preliminary}";
  } else if( status == PUBLIC ) {
    txt = "#CMS";
  } else if( status == SIMULATION ) {
    txt = "#CMS{Simulation}";
  } else if( status == UNPUBLISHED ) {
    txt = "#CMS{(unpublished)}";
  } else if( status == CUSTOM ) {
    txt = customTitle_;
  }

  return txt;
}

TString TkAlStyle::rightheader(const Era era)
{
  TString txt = "";
  if( era != NONE ) {
    txt = toTString(era);
  } else {
    txt = customRightTitle_;
  }
  return txt;
}


// --------------------------------------------------------------
void TkAlStyle::set(const PublicationStatus status, const Era era, const TString customTitle, const TString customRightTitle) {
  // Store the PublicationStatus for later usage, e.g. in the title
  publicationStatus_ = status;
  customTitle_ = customTitle;
  customRightTitle_ = customRightTitle;
  era_ = era;
  if (publicationStatus_ == CUSTOM && customTitle_ == "")
    std::cout << "Error: you are trying to use a custom title, but you don't provide it" << std::endl;
  if (publicationStatus_ != CUSTOM && customTitle_ != "")
    std::cout << "Error: you provide a custom title, but you don't indicate CUSTOM status.  Your title will not be used." << std::endl;

  // Suppress message when canvas has been saved
  gErrorIgnoreLevel = 1001;

  // Zero horizontal error bars
  gStyle->SetErrorX(0);

  //  For the canvas
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetCanvasDefH(800); //Height of canvas
  gStyle->SetCanvasDefW(800); //Width of canvas
  gStyle->SetCanvasDefX(0);   //Position on screen
  gStyle->SetCanvasDefY(0);
  
  //  For the frame
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(10);
  gStyle->SetFrameFillColor(kBlack);
  gStyle->SetFrameFillStyle(0);
  gStyle->SetFrameLineColor(kBlack);
  gStyle->SetFrameLineStyle(0);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetLineWidth(3);
    
  //  For the Pad
  gStyle->SetPadBorderMode(0);
  gStyle->SetPadColor(kWhite);
  gStyle->SetPadGridX(false);
  gStyle->SetPadGridY(false);
  gStyle->SetGridColor(0);
  gStyle->SetGridStyle(3);
  gStyle->SetGridWidth(1);
  
  //  Margins
  gStyle->SetPadTopMargin(0.08);
  gStyle->SetPadBottomMargin(0.13);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadRightMargin(0.05);

  //  For the histo:
  gStyle->SetHistLineColor(kBlack);
  gStyle->SetHistLineStyle(0);
  gStyle->SetHistLineWidth(3);
  gStyle->SetMarkerSize(0.8);
  gStyle->SetEndErrorSize(4);
  gStyle->SetHatchesLineWidth(1);

  //  For the statistics box:
  gStyle->SetOptStat(0);
  
  //  For the axis
  gStyle->SetAxisColor(1,"XYZ");
  gStyle->SetTickLength(0.03,"XYZ");
  gStyle->SetNdivisions(510,"XYZ");
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetStripDecimals(kFALSE);
  
  //  For the axis labels and titles
  gStyle->SetTitleColor(1,"XYZ");
  gStyle->SetLabelColor(1,"XYZ");
  gStyle->SetLabelFont(42,"XYZ");
  gStyle->SetLabelOffset(0.007,"XYZ");
  gStyle->SetLabelSize(0.04,"XYZ");
  gStyle->SetTitleFont(42,"XYZ");
  gStyle->SetTitleSize(0.047,"XYZ");
  gStyle->SetTitleXOffset(1.2);
  gStyle->SetTitleYOffset(1.5);

  //  For the legend
  gStyle->SetLegendBorderSize(0);
}

void TkAlStyle::set(const TString customTitle)
{
  set(CUSTOM, NONE, customTitle);
}

#endif
