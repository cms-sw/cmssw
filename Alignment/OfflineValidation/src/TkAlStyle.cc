#include "Alignment/OfflineValidation/interface/TkAlStyle.h"

TString toTString(const PublicationStatus status) {
  TString str = "";
  if (status == NO_STATUS)
    str = "Status not set yet!";
  else if (status == INTERNAL)
    str = "internal";
  else if (status == INTERNAL_SIMULATION)
    str = "simulation (internal)";
  else if (status == PRELIMINARY)
    str = "preliminary";
  else if (status == PUBLIC)
    str = "public";
  else if (status == SIMULATION)
    str = "simulation (public)";
  else if (status == UNPUBLISHED)
    str = "unpublished";
  else if (status == CUSTOM)
    str = "custom title set";

  return str;
}

static TString toTString(const Era era) {
  TString str = "";
  if (era == CRUZET15)
    str = "0T cosmic ray data 2015";
  else if (era == CRAFT15)
    str = "3.8T cosmic ray data 2015";
  else if (era == COLL0T15)
    str = "0T collision data 2015";

  return str;
}

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
PublicationStatus TkAlStyle::toStatus(std::string status) {
  PublicationStatus st;
  std::for_each(status.begin(), status.end(), [](char& c) { c = ::toupper(c); });
  if (status == "NO_STATUS")
    st = NO_STATUS;
  else if (status == "INTERNAL")
    st = INTERNAL;
  else if (status == "SIMULATION (INTERNAL)")
    st = INTERNAL_SIMULATION;
  else if (status == "PRELIMINARY")
    st = PRELIMINARY;
  else if (status == "PUBLIC")
    st = PUBLIC;
  else if (status == "SIMULATION")
    st = SIMULATION;
  else if (status == "UNPUBLISHED")
    st = UNPUBLISHED;
  else
    st = CUSTOM;

  return st;
}

// --------------------------------------------------------------
void TkAlStyle::setXCoordinatesL(const double relWidth, double& x0, double& x1) {
  x0 = gStyle->GetPadLeftMargin() + margin_;
  x1 = x0 + relWidth * (1. - gStyle->GetPadLeftMargin() - gStyle->GetPadRightMargin() - 2. * margin_);
}

// --------------------------------------------------------------
void TkAlStyle::setXCoordinatesR(const double relWidth, double& x0, double& x1) {
  x0 = 1. - gStyle->GetPadRightMargin() - margin_ -
       relWidth * (1. - gStyle->GetPadLeftMargin() - gStyle->GetPadRightMargin() - 2. * margin_);
  x1 = 1. - gStyle->GetPadRightMargin() - margin_;
}

// --------------------------------------------------------------
void TkAlStyle::setYCoordinatesT(const int nEntries, double& y0, double& y1) {
  y1 = 1. - gStyle->GetPadTopMargin() - margin_;
  y0 = y1 - nEntries * lineHeight_;
}

// --------------------------------------------------------------
void TkAlStyle::setYCoordinatesB(const int nEntries, double& y0, double& y1) {
  y1 = gStyle->GetPadBottomMargin() + margin_;
  y0 = y1 + nEntries * lineHeight_;
}

// --------------------------------------------------------------
TLegend* TkAlStyle::legend(const int nEntries, const double relWidth, const bool left, const bool top) {
  double x0 = 0.;
  double x1 = 0.;
  double y0 = 0.;
  double y1 = 0.;
  bool hasheader = (TkAlStyle::legendheader != "");
  if (left)
    setXCoordinatesL(relWidth, x0, x1);
  else
    setXCoordinatesR(relWidth, x0, x1);
  if (top)
    setYCoordinatesT(nEntries + hasheader, y0, y1);
  else
    setYCoordinatesB(nEntries + hasheader, y0, y1);

  TLegend* leg = new TLegend(x0, y0, x1, y1);
  leg->SetBorderSize(0);
  leg->SetFillColor(0);
  leg->SetFillStyle(0);
  leg->SetTextFont(42);
  leg->SetTextSize(textSize);
  if (hasheader)
    leg->SetHeader(TkAlStyle::legendheader);

  return leg;
}

// --------------------------------------------------------------
TPaveText* TkAlStyle::label(const int nEntries, const double relWidth, const bool left, const bool top) {
  double x0 = 0.;
  double x1 = 0.;
  double y0 = 0.;
  double y1 = 0.;
  if (left)
    setXCoordinatesL(relWidth, x0, x1);
  else
    setXCoordinatesR(relWidth, x0, x1);
  if (top)
    setYCoordinatesT(nEntries, y0, y1);
  else
    setYCoordinatesB(nEntries, y0, y1);

  TPaveText* label = new TPaveText(x0, y0, x1, y1, "NDC");
  label->SetBorderSize(0);
  label->SetFillColor(0);
  label->SetFillStyle(0);
  label->SetTextFont(42);
  label->SetTextAlign(12);  // left adjusted and vertically centered
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
  newtxt.ReplaceAll("#CMS{", "#scale[1.4]{#font[61]{CMS}} #font[52]{");
  newtxt.ReplaceAll("#noCMS{", "#font[52]{");
  newtxt.ReplaceAll("#CMS", "#scale[1.4]{#font[61]{CMS}}");
  return newtxt;
}

// --------------------------------------------------------------
TPaveText* TkAlStyle::title(const TString& txt) {
  double x0 = gStyle->GetPadLeftMargin();
  double x1 = 1. - gStyle->GetPadRightMargin();
  double y0 = 1. - gStyle->GetPadTopMargin();
  double y1 = 1.;
  if (txt.Contains("#CMS"))
    y0 += .02;
  TPaveText* theTitle = new TPaveText(x0, y0, x1, y1, "NDC");
  theTitle->SetBorderSize(0);
  theTitle->SetFillColor(10);
  theTitle->SetFillStyle(0);
  theTitle->SetTextFont(42);
  theTitle->SetTextAlign(13);  // left bottom adjusted
  theTitle->SetTextSize(0.038);
  theTitle->SetMargin(0.);
  theTitle->AddText(applyCMS(txt));

  return theTitle;
}

// --------------------------------------------------------------
TPaveText* TkAlStyle::righttitle(const TString& txt) {
  TString newtxt = applyCMS(txt);
  double x0 = gStyle->GetPadLeftMargin();
  double x1 = 1. - gStyle->GetPadRightMargin();
  double y0 = 1. - gStyle->GetPadTopMargin();
  double y1 = 1.;
  TPaveText* theTitle = new TPaveText(x0, y0, x1, y1, "NDC");
  theTitle->SetBorderSize(0);
  theTitle->SetFillColor(10);
  theTitle->SetFillStyle(0);
  theTitle->SetTextFont(42);
  theTitle->SetTextAlign(33);  // right bottom adjusted
  theTitle->SetTextSize(0.038);
  theTitle->SetMargin(0.);
  theTitle->AddText(newtxt);

  return theTitle;
}

// --------------------------------------------------------------
TString TkAlStyle::header(const PublicationStatus status) {
  TString txt;
  if (status == NO_STATUS) {
    std::cout << "Status not set yet!  Can't draw the title!" << std::endl;
  } else if (status == INTERNAL_SIMULATION) {
    txt = "#noCMS{Simulation}";
  } else if (status == INTERNAL) {
    txt = "#CMS{Internal}";
  } else if (status == PRELIMINARY) {
    txt = "#CMS{Preliminary}";
  } else if (status == PUBLIC) {
    txt = "#CMS";
  } else if (status == SIMULATION) {
    txt = "#CMS{Simulation}";
  } else if (status == UNPUBLISHED) {
    txt = "#CMS{(unpublished)}";
  } else if (status == CUSTOM) {
    txt = customTitle_;
  }

  return txt;
}

TString TkAlStyle::rightheader(const Era era) {
  TString txt = "";
  if (era != NONE) {
    txt = toTString(era);
  } else {
    txt = customRightTitle_;
  }
  return txt;
}

// --------------------------------------------------------------
void TkAlStyle::set(const PublicationStatus status,
                    const Era era,
                    const TString customTitle,
                    const TString customRightTitle) {
  // Store the PublicationStatus for later usage, e.g. in the title
  publicationStatus_ = status;
  customTitle_ = customTitle;
  customRightTitle_ = customRightTitle;
  era_ = era;
  if (publicationStatus_ == CUSTOM && customTitle_ == "")
    std::cout << "Error: you are trying to use a custom title, but you don't provide it" << std::endl;
  if (publicationStatus_ != CUSTOM && customTitle_ != "")
    std::cout
        << "Error: you provide a custom title, but you don't indicate CUSTOM status.  Your title will not be used."
        << std::endl;

  // Suppress message when canvas has been saved
  gErrorIgnoreLevel = 1001;

  // Zero horizontal error bars
  gStyle->SetErrorX(0);

  //  For the canvas
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetCanvasDefH(800);  //Height of canvas
  gStyle->SetCanvasDefW(800);  //Width of canvas
  gStyle->SetCanvasDefX(0);    //Position on screen
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
  gStyle->SetAxisColor(1, "XYZ");
  gStyle->SetTickLength(0.03, "XYZ");
  gStyle->SetNdivisions(510, "XYZ");
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetStripDecimals(kFALSE);

  //  For the axis labels and titles
  gStyle->SetTitleColor(1, "XYZ");
  gStyle->SetLabelColor(1, "XYZ");
  gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetLabelOffset(0.007, "XYZ");
  gStyle->SetLabelSize(0.04, "XYZ");
  gStyle->SetTitleFont(42, "XYZ");
  gStyle->SetTitleSize(0.047, "XYZ");
  gStyle->SetTitleXOffset(1.2);
  gStyle->SetTitleYOffset(1.5);

  //  For the legend
  gStyle->SetLegendBorderSize(0);
}

void TkAlStyle::set(const TString customTitle) { set(CUSTOM, NONE, customTitle); }
