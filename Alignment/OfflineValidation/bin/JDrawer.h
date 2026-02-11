#ifndef ALIGNMENT_OFFLINEVALIDATION_JDRAWER_H
#define ALIGNMENT_OFFLINEVALIDATION_JDRAWER_H

// C++ includes
#include <iostream>

// Root includes
#include "TROOT.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TString.h"
#include "TH1.h"
#include "TF1.h"
#include "TH2.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TLegend.h"

/*
 * Class for drawing histograms and graphs in the standard style of the group
 */
class JDrawer {
private:
  // Margins for the drawn canvases
  double fMarginLeft;    // left margin for the canvas
  double fMarginRight;   // right margin for the canvas
  double fMarginBottom;  // bottom margin for the canvas
  double fMarginTop;     // top margin for the canvas

  // Place and size for the canvas
  int fTopLeftX;      // Number of pixels the the canvas is out of the top left corner of the screen in x-direction
  int fTopLeftY;      // Number of pixels the the canvas is out of the top left corner of the screen in y-direction
  int fCanvasWidth;   // Width of the canvas
  int fCanvasHeight;  // Height of the canvas

  // Appearance settings for histogram
  double fTitleOffsetX;  // Offset of the x-axis title
  double fTitleOffsetY;  // Offset of the y-axis title
  double fTitleOffsetZ;  // Offset of the z-axis title
  double fTitleSizeX;    // Size of the x-axis title
  double fTitleSizeY;    // Size of the y-axis title
  double fTitleSizeZ;    // Size of the z-axis title
  double fLabelOffsetX;  // Offset of the x-axis label
  double fLabelOffsetY;  // Offset of the y-axis label
  double fLabelOffsetZ;  // Offset of the z-axis label
  double fLabelSizeX;    // Size of the x-axis label
  double fLabelSizeY;    // Size of the y-axis label
  double fLabelSizeZ;    // Size of the z-axis label
  double fDivisionsX;    // The number of divisions in x-axis
  double fDivisionsY;    // The number of divisions in y-axis
  int fFont;             // Font index for titles and labels

  // TPad appearance settings
  int fLogX;   // 1: Logarithmic x-axis, 0: linear x-axis
  int fLogY;   // 1: Logarithmic y-axis, 0: linear y-axis
  int fLogZ;   // 1: Logarithmic z-axis, 0: linear z-axis
  int fGridX;  // 1: Grid in x-axis, 0: no grid
  int fGridY;  // 1: Grid in y-axis, 0: no grid
  int fTickX;  // 1: Ticks in x-axis, 0: no ticks
  int fTickY;  // 1: Ticks in y-axis, 0: no ticks

  // Canvas displacement settings
  int fCanvasDisplacementX;  // Displacement of the canvas for one index in x-direction
  int fCanvasDisplacementY;  // Displacement of the canvas for one index in y-direction
  int fCanvasesInOneRow;     // Number of canvases in a row before a new row is started

  // Handles needed for custom canvas creation
  TCanvas *fCanvas;      // If only a custom canvas is drawn, we need to keep on handle on that
  TPad *fSinglePad;      // Single pad for single canvas
  TPad *fUpperSplitPad;  // Upper pad in split canvas
  TPad *fLowerSplitPad;  // Lower pad in split canvas
  TPad *fLeftRowPad;     // Left pad in a row of three pads
  TPad *fMiddleRowPad;   // Middle pad in a row of three pads
  TPad *fRightRowPad;    // Right pad in the row of three pads

  // Settings for canvas splitting
  double fSplitRatio;  // The percentage of the lower pad from the total canvas when splitting the canvas

  // Index for generating names
  int fNameIndex;  // Names for canvases and pads are generated using this index. Static so that unique naming is ensured
      // This used to be static, but for some reason does not compile if defined as static. Should work like this also though
      // Remember: static means that only one fNameIndex is used by all the JDrawers that are created in the code

  /*
   * Trim the styles of given axes
   *
   * TAxis *xAxis = The styled x-axis
   * TAxis *yAxis = The styled y-axis
   * TAxis *zAxis = The styled z-axis
   * TString xtit = Title of the x-axis
   * TString ytit = Title of the y-axis
   */
  void SetAxisStyles(TAxis *xAxis, TAxis *yAxis, TAxis *zAxis, TString xtit, TString ytit) {
    xAxis->CenterTitle(true);  // Axis titles are centered
    yAxis->CenterTitle(true);  // Axis titles are centered
    if (zAxis)
      zAxis->CenterTitle(true);  // Axis titles are centered

    xAxis->SetTitleOffset(fTitleOffsetX);  // Give a small offset to the title so that it does overlap with axis
    yAxis->SetTitleOffset(fTitleOffsetY);  // Give a small offset to the title so that it does overlap with axis
    if (zAxis)
      zAxis->SetTitleOffset(fTitleOffsetZ);  // Give a small offset to the title so that it does overlap with axis

    xAxis->SetTitleSize(fTitleSizeX);  // Define the size of the title
    yAxis->SetTitleSize(fTitleSizeY);  // Define the size of the title
    if (zAxis)
      zAxis->SetTitleSize(fTitleSizeZ);  // Define the size of the title

    xAxis->SetLabelOffset(fLabelOffsetX);  // Give a small offset to the label so that it does overlap with axis
    yAxis->SetLabelOffset(fLabelOffsetY);  // Give a small offset to the label so that it does overlap with axis
    if (zAxis)
      zAxis->SetLabelOffset(fLabelOffsetZ);  // Give a small offset to the label so that it does overlap with axis

    xAxis->SetLabelSize(fLabelSizeX);  // Define the sixe of the label
    yAxis->SetLabelSize(fLabelSizeY);  // Define the sixe of the label
    if (zAxis)
      zAxis->SetLabelSize(fLabelSizeZ);  // Define the sixe of the label

    xAxis->SetNdivisions(fDivisionsX);  // Set the number of division markers
    yAxis->SetNdivisions(fDivisionsY);  // Set the number of division markers

    xAxis->SetTitle(xtit);  // Set the axis title
    yAxis->SetTitle(ytit);  // Set the axis title

    xAxis->SetLabelFont(fFont);  // Set the label font
    yAxis->SetLabelFont(fFont);  // Set the label font
    if (zAxis)
      zAxis->SetLabelFont(fFont);  // Set the label font
    xAxis->SetTitleFont(fFont);    // Set the title font
    yAxis->SetTitleFont(fFont);    // Set the title font
    if (zAxis)
      zAxis->SetTitleFont(fFont);  // Set the title font
  }

  /*
   * Trim the style of a histogram using the defined style values
   *
   * TH1 *hid = Histogram which is styled
   * TString xtit = Title of the x-axis
   * TString ytit = Title of the y-axis
   */
  void SetHistogramStyle(TH1 *hid, TString xtit, TString ytit) {
    SetAxisStyles(hid->GetXaxis(), hid->GetYaxis(), hid->GetZaxis(), xtit, ytit);
  }

  /*
   * Trim the style of a function using the defined style values
   *
   * TF1 *hid = Function which is styled
   * TString xtit = Title of the x-axis
   * TString ytit = Title of the y-axis
   */
  void SetFunctionStyle(TF1 *hid, TString xtit, TString ytit) {
    SetAxisStyles(hid->GetXaxis(), hid->GetYaxis(), hid->GetZaxis(), xtit, ytit);
  }

  /*
   * Trim the style of a graph using the defined style values
   *
   * TGraph *hid = Histogram which is styled
   * TString xtit = Title of the x-axis
   * TString ytit = Title of the y-axis
   */
  void SetGraphStyle(TGraph *hid, TString xtit, TString ytit) {
    SetAxisStyles(hid->GetXaxis(), hid->GetYaxis(), nullptr, xtit, ytit);
  }

  /*
   *  Set the pad specific values to the given pad
   */
  void SetPadValues(TPad *pad) {
    pad->SetLogy(fLogY);
    pad->SetLogx(fLogX);
    pad->SetLogz(fLogZ);
    pad->SetGridx(fGridX);
    pad->SetGridy(fGridY);
    pad->SetTickx(fTickX);
    pad->SetTicky(fTickY);
  }

  /*
   * Generate a name for an object that does not exist yet
   */
  TString GenerateName(const char *objectName) {
    TString name = Form("%s%d%d%d", objectName, fNameIndex++, fTopLeftX, fTopLeftY);
    return name.Data();
  }

  /*
   * Generate a name for a canvas
   */
  TString GenerateNameForCanvas() {
    TString name;
    name = Form("canvas%d%d", fTopLeftX, fTopLeftY);
    while (true) {
      // If a canvas with a specified name does not exist, accept the name
      if (gROOT->GetListOfCanvases()->FindObject(name) == nullptr) {
        return name.Data();
      }

      // If a canvas with a specified name does exist, try a new name
      name = Form("canvas%d%d%d", fTopLeftX, fTopLeftY, fNameIndex++);
    }
  }

public:
  /*
   * Constructor for JDrawer
   */
  JDrawer() {
    Reset();  // Set all the style values to default
    fNameIndex = 0;
  }

  /*
   * Destructor for JDrawer
   */
  ~JDrawer() = default;

  /*
   *  Draw a histogram to a canvas
   *
   *  TH1 *histo = histogram to be drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawHistogramToCurrentCanvas(
      TH1 *histo, const char *xTitle, const char *yTitle, const char *title = "", const char *drawOption = "") {
    // If no titles are given, keep the original ones
    if (strcmp(xTitle, "") == 0)
      xTitle = histo->GetXaxis()
                   ->GetTitle();  // To compare char*:s we need to use strcmp function provided by <cstring> library
    if (strcmp(yTitle, "") == 0)
      yTitle = histo->GetYaxis()->GetTitle();
    if (strcmp(title, "") == 0)
      title = histo->GetTitle();

    // Set up the histogram and draw it to current canvas
    histo->SetTitle(title);
    SetHistogramStyle(histo, xTitle, yTitle);
    histo->Draw(drawOption);
  }

  /*
   *  Draw a histogram to a canvas
   *
   *  TH1 *histo = histogram to be drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawHistogram(
      TH1 *histo, const char *xTitle, const char *yTitle, const char *title = "", const char *drawOption = "") {
    // If no titles are given, keep the original ones
    if (strcmp(xTitle, "") == 0)
      xTitle = histo->GetXaxis()
                   ->GetTitle();  // To compare char*:s we need to use strcmp function provided by <cstring> library
    if (strcmp(yTitle, "") == 0)
      yTitle = histo->GetYaxis()->GetTitle();
    if (strcmp(title, "") == 0)
      title = histo->GetTitle();

    // Set up the histogram and draw it to canvas
    CreateCanvas();
    histo->SetTitle(title);
    SetHistogramStyle(histo, xTitle, yTitle);
    histo->Draw(drawOption);
  }

  /*
   * Draw histogram without changing the titles
   *
   *  TH1 *histo = histogram to be drawn
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawHistogram(TH1 *histo, const char *drawOption = "") { DrawHistogram(histo, "", "", "", drawOption); }

  /*
   *  Draw a histogram to a pad of a split canvas
   *
   *  TH1 *histo = histogram to be drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawHistogramToPad(TH1 *histo,
                          TPad *drawPad,
                          const char *xTitle = "",
                          const char *yTitle = "",
                          const char *title = "",
                          const char *drawOption = "") {
    // If no titles are given, keep the original ones
    if (strcmp(xTitle, "") == 0)
      xTitle = histo->GetXaxis()
                   ->GetTitle();  // To compare char*:s we need to use strcmp function provided by <cstring> library
    if (strcmp(yTitle, "") == 0)
      yTitle = histo->GetYaxis()->GetTitle();
    if (strcmp(title, "") == 0)
      title = histo->GetTitle();

    // Change to the desired pad
    SetPadValues(drawPad);
    drawPad->cd();

    // Remove statistics box and title
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    // Set up the histogram and draw it to the pad
    histo->SetTitle(title);
    SetHistogramStyle(histo, xTitle, yTitle);
    histo->Draw(drawOption);
  }

  /*
   *  Draw a histogram to an upper pad of a split canvas
   *
   *  TH1 *histo = histogram to be drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawHistogramToUpperPad(TH1 *histo,
                               const char *xTitle = "",
                               const char *yTitle = "",
                               const char *title = "",
                               const char *drawOption = "") {
    // If there is no upper pad, create one
    if (fUpperSplitPad == nullptr) {
      //SetDefaultAppearanceSplitCanvas();
      CreateSplitCanvas();
    }

    DrawHistogramToPad(histo, fUpperSplitPad, xTitle, yTitle, title, drawOption);
  }

  /*
   *  Draw a histogram to an upper pad of a split canvas
   *
   *  TH1 *histo = histogram to be drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawHistogramToLowerPad(TH1 *histo,
                               const char *xTitle = "",
                               const char *yTitle = "",
                               const char *title = "",
                               const char *drawOption = "") {
    // If there is no lower pad, create one
    if (fLowerSplitPad == nullptr) {
      //SetDefaultAppearanceSplitCanvas();
      CreateSplitCanvas();
    }

    DrawHistogramToPad(histo, fLowerSplitPad, xTitle, yTitle, title, drawOption);
  }

  /*
   *  Draw a graph to a canvas
   *
   *  TGraph *graph = graph to be drawn
   *  double xlow = lowest x-axis value drawn
   *  double xhigh = highest x-axis value drawn
   *  double ylow = lowest y-axis value drawn
   *  double yhigh = highest y-axis value drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawGraph(TGraph *graph,
                 double xlow,
                 double xhigh,
                 double ylow,
                 double yhigh,
                 const char *xTitle = "",
                 const char *yTitle = "",
                 const char *title = "",
                 const char *drawOption = "") {
    // Create the canvas with right specs and a graph into it
    CreateCanvas(xlow, xhigh, ylow, yhigh, xTitle, yTitle, title);
    graph->Draw(drawOption);
  }

  /*
   *  Draw a graph to a canvas. Notice that with custom axes the option "a" should be given for the first graph in each canvas.
   *
   *  TGraph *graph = graph to be drawn
   *  double xlow = lowest x-axis value drawn
   *  double xhigh = highest x-axis value drawn
   *  double ylow = lowest y-axis value drawn
   *  double yhigh = highest y-axis value drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawGraphCustomAxes(TGraph *graph,
                           double xlow,
                           double xhigh,
                           double ylow,
                           double yhigh,
                           const char *xTitle = "",
                           const char *yTitle = "",
                           const char *title = "",
                           const char *drawOption = "") {
    // If no titles are given, keep the original ones
    if (strcmp(xTitle, "") == 0)
      xTitle = graph->GetXaxis()
                   ->GetTitle();  // To compare char*:s we need to use strcmp function provided by <cstring> library
    if (strcmp(yTitle, "") == 0)
      yTitle = graph->GetYaxis()->GetTitle();
    if (strcmp(title, "") == 0)
      title = graph->GetTitle();

    // Set up the histogram and draw it to canvas
    CreateCanvas();
    graph->SetTitle(title);
    graph->GetXaxis()->SetRangeUser(xlow, xhigh);
    graph->GetYaxis()->SetRangeUser(ylow, yhigh);
    SetGraphStyle(graph, xTitle, yTitle);
    graph->Draw(drawOption);
  }

  /*
   *  Draw a function to a canvas
   *
   *  TF1 *myFun = histogram to be drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawFunction(
      TF1 *myFun, const char *xTitle, const char *yTitle, const char *title = "", const char *drawOption = "") {
    // If no titles are given, keep the original ones
    if (strcmp(xTitle, "") == 0)
      xTitle = myFun->GetXaxis()
                   ->GetTitle();  // To compare char*:s we need to use strcmp function provided by <cstring> library
    if (strcmp(yTitle, "") == 0)
      yTitle = myFun->GetYaxis()->GetTitle();
    if (strcmp(title, "") == 0)
      title = myFun->GetTitle();

    // Set up the histogram and draw it to canvas
    CreateCanvas();
    myFun->SetTitle(title);
    SetFunctionStyle(myFun, xTitle, yTitle);
    myFun->Draw(drawOption);
  }

  /*
   * Draw histogram without changing the titles
   *
   *  TF1 *myFun = function to be drawn
   *  char *drawOption = options for drawing given in root documentation
   */
  void DrawFunction(TF1 *myFun, const char *drawOption = "") { DrawFunction(myFun, "", "", "", drawOption); }

  /*
   * Set all the values back to default
   */
  void Reset() {
    // Set default values for margins
    fMarginLeft = 0.15;    // left margin
    fMarginRight = 0.06;   // right margin
    fMarginBottom = 0.15;  // bottom margin
    fMarginTop = 0.06;     // top margin

    // Set default values for the size and place of the canvas
    fTopLeftX = 10;       // Number of pixels the the canvas is out of the top left corner of the screen in x-direction
    fTopLeftY = 10;       // Number of pixels the the canvas is out of the top left corner of the screen in y-direction
    fCanvasWidth = 700;   // Width of the canvas
    fCanvasHeight = 500;  // Height of the canvas

    // Set default values for histogram appearance settings
    fTitleOffsetX = 1.1;    // Offset of the x-axis title
    fTitleOffsetY = 1.1;    // Offset of the y-axis title
    fTitleOffsetZ = 1.3;    // Pffset of the z-axis title
    fTitleSizeX = 0.06;     // Size of the x-axis title
    fTitleSizeY = 0.06;     // Size of the y-axis title
    fTitleSizeZ = 0.06;     // Size of the z-axis title
    fLabelOffsetX = 0.01;   // Offset of the x-axis label
    fLabelOffsetY = 0.001;  // Offset of the y-axis label
    fLabelOffsetZ = 0.001;  // Offset of the z-axis label
    fLabelSizeX = 0.05;     // Size of the x-axis label
    fLabelSizeY = 0.05;     // Size of the y-axis label
    fLabelSizeZ = 0.05;     // Size of the z-axis label
    fDivisionsX = 505;      // The number of divisions in x-axis
    fDivisionsY = 505;      // The number of divisions in y-axis
    fFont = 42;             // Font index for titles and labels

    // Axes are linear by default
    fLogX = 0;  // 1: Logarithmic x-axis, 0: linear x-axis
    fLogY = 0;  // 1: Logarithmic y-axis, 0: linear y-axis
    fLogZ = 0;  // 1: Logarithmic z-axis, 0: linear z-axis

    // Set default values for canves displacement information
    fCanvasDisplacementX = 100;  // Displacement of the canvas for one index in x-direction
    fCanvasDisplacementY = 100;  // Displacement of the canvas for one index in y-direction
    fCanvasesInOneRow = 10;      // Number of canvases in a row before a new row is started

    // Set defauls values for pad properties
    fGridX = 0;  // 1: Grid in x-axis, 0: no grid
    fGridY = 0;  // 1: Grid in y-axis, 0: no grid
    fTickX = 1;  // 1: Ticks in x-axis, 0: no ticks
    fTickY = 1;  // 1: Ticks in y-axis, 0: no ticks

    // Default setting values for splitted canvas
    fSplitRatio = 0.4;  // The percentage of the lower pad from the total canvas when splitting the canvas

    // Set the canvas handles to null
    fCanvas = nullptr;
    fSinglePad = nullptr;
    fUpperSplitPad = nullptr;
    fLowerSplitPad = nullptr;
    fLeftRowPad = nullptr;
    fMiddleRowPad = nullptr;
    fRightRowPad = nullptr;
  }

  /*
   * Create a canvas and one pad to it
   */
  void CreateCanvas() {
    // Create a canvas and set up its appearance
    gStyle->SetOptStat(0);  // remove statistics box
    TString cname = GenerateNameForCanvas();
    fCanvas = new TCanvas(cname.Data(), cname.Data(), fTopLeftX, fTopLeftY, fCanvasWidth, fCanvasHeight);
    fSinglePad = new TPad("pad", "pad", 0.01, 0.01, 0.99, 0.99, 0, 0, 0);
    fSinglePad->SetLeftMargin(fMarginLeft);
    fSinglePad->SetBottomMargin(fMarginBottom);
    fSinglePad->SetTopMargin(fMarginTop);
    fSinglePad->SetRightMargin(fMarginRight);
    SetPadValues(fSinglePad);
    fSinglePad->Draw();
    fSinglePad->cd();
  }

  /*
   *  Create a canvas with a grid of desired size and titles
   *
   *  double xlow = lowest x-axis value drawn
   *  double xhigh = highest x-axis value drawn
   *  double ylow = lowest y-axis value drawn
   *  double yhigh = highest y-axis value drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   */
  void CreateCanvas(double xlow,
                    double xhigh,
                    double ylow,
                    double yhigh,
                    const char *xTitle = "",
                    const char *yTitle = "",
                    const char *title = "") {
    // First make a canvas
    CreateCanvas();

    // Create a dummy 2D-histogram to set the drawing ranges and titles
    TString dummyName = GenerateName("histo");
    TH2F *dummyHisto = new TH2F(dummyName.Data(), title, 10, xlow, xhigh, 10, ylow, yhigh);
    SetHistogramStyle(dummyHisto, xTitle, yTitle);
    dummyHisto->Draw();
  }

  /*
   * Create a canvas that is split to two pads that are on top of each other
   */
  void CreateSplitCanvas() {
    TString dummyName;  // dummy name generator
    dummyName = GenerateNameForCanvas();

    // Create the canvas
    fCanvas = new TCanvas(dummyName.Data(), dummyName.Data(), fTopLeftX, fTopLeftY, fCanvasWidth, fCanvasHeight);
    fCanvas->SetFillStyle(4000);     // The created canvas is completely transparent
    fCanvas->SetFillColor(10);       // Canvas is filled with white color
    gStyle->SetOptStat(0);           // Disable the drawing of the statistics box to the canvas
    gStyle->SetOptTitle(0);          // Disable the drawing of titles to the canvas
    fCanvas->SetMargin(0, 0, 0, 0);  // Remove the margins from the bottom canvas
    fCanvas->Draw();                 // Draw the canvas to the screen

    // ---- Create the two pads to the canvas ----

    // The upper pad stretches from the top of the canvas to the defined split ratio
    dummyName = GenerateName("UpperPad");
    fUpperSplitPad = new TPad(dummyName.Data(), dummyName.Data(), 0, fSplitRatio, 1, 1, 0);
    fUpperSplitPad->SetTopMargin(fMarginTop / (1 - fSplitRatio));  // Adjust top margin according to the split ratio
    fUpperSplitPad->SetBottomMargin(
        0.0015);  // Make the bottom margin small so that the pads fit nicely on top of each other
    fUpperSplitPad->SetLeftMargin(fMarginLeft);
    fUpperSplitPad->SetRightMargin(fMarginRight);
    fUpperSplitPad->Draw();

    // The lower pad stretches from the defined split ratio to the bottom of the canvas
    dummyName = GenerateName("LowerPad");
    fLowerSplitPad = new TPad(dummyName.Data(), dummyName.Data(), 0, 0, 1, fSplitRatio, 0);
    fLowerSplitPad->SetTopMargin(0.0015);  // Make the top margin small so that the pads fit nicely on top of each other
    fLowerSplitPad->SetBottomMargin(fMarginBottom / fSplitRatio);  // Adjust bottom margin according to the split ratio
    fLowerSplitPad->SetLeftMargin(fMarginLeft);
    fLowerSplitPad->SetRightMargin(fMarginRight);
    fLowerSplitPad->Draw();
  }

  /*
   * Create a canvas where three graphs can be drawn in one row (to show all xlong bins in one figure for example)
   */
  void CreateCanvasGraphRow() {
    TString dummyName;  // dummy name generator
    dummyName = GenerateNameForCanvas();

    // Create the canvas
    fCanvas = new TCanvas(dummyName.Data(), dummyName.Data(), fTopLeftX, fTopLeftY, fCanvasWidth, fCanvasHeight);
    fCanvas->SetFillStyle(4000);     // The created canvas is completely transparent
    fCanvas->SetFillColor(10);       // Canvas is filled with white color
    gStyle->SetOptStat(0);           // Disable the drawing of the statistics box to the canvas
    gStyle->SetOptTitle(0);          // Disable the drawing of titles to the canvas
    fCanvas->SetMargin(0, 0, 0, 0);  // Remove the margins from the bottom canvas
    fCanvas->Draw();                 // Draw the canvas to the screen

    // ---- Create the two pads to the canvas ----

    // First, find the correct relation between different pads based on the required margins
    double smallMargin = 0.0006;
    double evenSmallerMargin = 0.0002;
    double relativePlotSizeLeftPad = 1 - (fMarginLeft + evenSmallerMargin);
    double relativePlotSizeMiddlePad = 1 - (2 * smallMargin);
    double relativePlotSizeRightPad = 1 - (fMarginRight + smallMargin);
    double firstSeparator = 1 / (1 + relativePlotSizeLeftPad / relativePlotSizeMiddlePad +
                                 relativePlotSizeLeftPad / relativePlotSizeRightPad);
    double secondSeparator = firstSeparator + (relativePlotSizeLeftPad / relativePlotSizeMiddlePad) * firstSeparator;

    // The first pad stretches from the left corner 1/3 of the total horizontal space, taking into account marginals
    dummyName = GenerateName("LeftPad");
    fLeftRowPad = new TPad(dummyName.Data(), dummyName.Data(), 0, 0, firstSeparator, 1, 0);
    fLeftRowPad->SetTopMargin(fMarginTop);
    fLeftRowPad->SetBottomMargin(fMarginBottom);
    fLeftRowPad->SetLeftMargin(fMarginLeft);
    fLeftRowPad->SetRightMargin(
        evenSmallerMargin);  // Make the right margin small so that the pads fit nicely on top of each other
    SetPadValues(fLeftRowPad);
    fLeftRowPad->Draw();

    // The second pad stretches from 1/3 of the total horizontal space to 2/3, taking into account marginals
    dummyName = GenerateName("MiddlePad");
    fMiddleRowPad = new TPad(dummyName.Data(), dummyName.Data(), firstSeparator, 0, secondSeparator, 1, 0);
    fMiddleRowPad->SetTopMargin(fMarginTop);
    fMiddleRowPad->SetBottomMargin(fMarginBottom);
    fMiddleRowPad->SetLeftMargin(
        smallMargin);  // Make the left margin small so that the pads fit nicely on top of each other
    fMiddleRowPad->SetRightMargin(
        smallMargin);  // Make the right margin small so that the pads fit nicely on top of each other
    SetPadValues(fMiddleRowPad);
    fMiddleRowPad->Draw();

    // The third pad stretches from 2/3 of the total horizontal space right corner, taking into account marginals
    dummyName = GenerateName("RightPad");
    fRightRowPad = new TPad(dummyName.Data(), dummyName.Data(), secondSeparator, 0, 1, 1, 0);
    fRightRowPad->SetTopMargin(fMarginTop);
    fRightRowPad->SetBottomMargin(fMarginBottom);
    fRightRowPad->SetLeftMargin(
        smallMargin);  // Make the left margin small so that the pads fit nicely on top of each other
    fRightRowPad->SetRightMargin(fMarginRight);
    SetPadValues(fRightRowPad);
    fRightRowPad->Draw();
  }

  /*
   *  Create a canvas with a grid of desired size and titles
   *
   *  double xlow = lowest x-axis value drawn
   *  double xhigh = highest x-axis value drawn
   *  double ylow = lowest y-axis value drawn
   *  double yhigh = highest y-axis value drawn
   *  char *xTitle = title for the x-axis
   *  char *yTitle = title for the y-axis
   *  char *title = title of the histogram
   */
  void CreateCanvasGraphRow(double xlow,
                            double xhigh,
                            double ylow,
                            double yhigh,
                            const char *xTitle = "",
                            const char *yTitle = "",
                            const char *title = "") {
    // First make a canvas
    CreateCanvasGraphRow();

    // Create a dummy 2D-histogram to set the drawing ranges and titles
    SelectPad(0);
    TString dummyName = GenerateName("histo");
    TH2F *dummyHisto = new TH2F(dummyName.Data(), "", 10, xlow, xhigh, 10, ylow, yhigh);
    SetHistogramStyle(dummyHisto, xTitle, yTitle);
    dummyHisto->Draw();

    // Create a dummy 2D-histogram to set the drawing ranges and titles
    SelectPad(1);
    dummyName = GenerateName("histo");
    dummyHisto = new TH2F(dummyName.Data(), title, 10, xlow, xhigh, 10, ylow, yhigh);
    SetHistogramStyle(dummyHisto, xTitle, "");
    dummyHisto->Draw();

    // Create a dummy 2D-histogram to set the drawing ranges and titles
    SelectPad(2);
    dummyName = GenerateName("histo");
    dummyHisto = new TH2F(dummyName.Data(), "", 10, xlow, xhigh, 10, ylow, yhigh);
    SetHistogramStyle(dummyHisto, xTitle, "");
    dummyHisto->Draw();
  }

  /*
   * Create split canvas with default settings defining the canvas displacement info
   *
   *  int canvasIndex = index number for the canvas, this is used to calculate the canvas displacement
   *  int canvasesInRow = number of canvases in a row before the next row is started
   */
  void CreateSplitCanvas(int canvasIndex, int canvasesInRow) {
    SetDefaultAppearanceSplitCanvas();
    SetNumberOfCanvasesInOneRow(canvasesInRow);
    SetCanvasDisplacement(canvasIndex);
    CreateSplitCanvas();
  }

  // Setter for left margin
  void SetLeftMargin(double margin) { fMarginLeft = margin; }

  // Setter for right margin
  void SetRightMargin(double margin) { fMarginRight = margin; }

  // Setter for bottom margin
  void SetBottomMargin(double margin) { fMarginBottom = margin; }

  // Setter for top margin
  void SetTopMargin(double margin) { fMarginTop = margin; }

  /*
   * Set all the margins for canvas
   *
   * double left = new left margin
   * double right = new right margin
   * double top = new top margin
   * doulbe bottom = new bottom margin
   */
  void SetMargins(double left, double right, double top, double bottom) {
    fMarginLeft = left;
    fMarginRight = right;
    fMarginTop = top;
    fMarginBottom = bottom;
  }

  // Setter for x-axis title offset
  void SetTitleOffsetX(double offset) { fTitleOffsetX = offset; }

  // Setter for y-axis title offset
  void SetTitleOffsetY(double offset) { fTitleOffsetY = offset; }

  // Setter for x-axis title size
  void SetTitleSizeX(double size) { fTitleSizeX = size; }

  // Setter for y-axis title size
  void SetTitleSizeY(double size) { fTitleSizeY = size; }

  // Setter for x-axis label offset
  void SetLabelOffsetX(double offset) { fLabelOffsetX = offset; }

  // Setter for y-axis label offset
  void SetLabelOffsetY(double offset) { fLabelOffsetY = offset; }

  // Setter for x-axis label size
  void SetLabelSizeX(double size) { fLabelSizeX = size; }

  // Setter for y-axis label size
  void SetLabelSizeY(double size) { fLabelSizeY = size; }

  // Setter for number of divisions in x-axis
  void SetNDivisionsX(int div) { fDivisionsX = div; }

  // Setter for number of divisions in y-axis
  void SetNDivisionsY(int div) { fDivisionsY = div; }

  // Setter for the font
  void SetFont(int fontIndex) { fFont = fontIndex; }

  /*
   * Apply the settings that are given to JDrawer to a given histogram
   *
   *  TH1 *histo = Histogram that is styled
   *  TString xTitle = title for the x axis
   *  TString yTitle = title for the y axis
   */
  void ApplyStyleSettings(TH1 *histo, TString xTitle = "", TString yTitle = "") {
    SetHistogramStyle(histo, xTitle, yTitle);
  }

  /*
   * Set all the appearance factors for the histogram
   *
   * double titoffx = Offset of the x-axis title
   * double titoffy = Offset of the y-axis title
   * double titsizex = Size of the x-axis title
   * double titsizey = Size of the y-axis title
   * double labeloffx = Offset of the x-axis label
   * double labeloffy = Offset of the y-axis label
   * double labelsizex = Size of the x-axis label
   * double labelsizey = Size of the y-axis label
   * int divx = The number of divisions in x-axis
   * int divy = The number of divisions in y-axis
   * int fontIndex = Font index for titles and labels
   */
  void SetHistogramAppearance(double titoffx,
                              double titoffy,
                              double titsizex,
                              double titsizey,
                              double labeloffx,
                              double labeloffy,
                              double labelsizex,
                              double labelsizey,
                              int divx,
                              int divy,
                              int fontIndex) {
    fTitleOffsetX = titoffx;    // Offset of the x-axis title
    fTitleOffsetY = titoffy;    // Offset of the y-axis title
    fTitleSizeX = titsizex;     // Size of the x-axis title
    fTitleSizeY = titsizey;     // Size of the y-axis title
    fLabelOffsetX = labeloffx;  // Offset of the x-axis label
    fLabelOffsetY = labeloffy;  // Offset of the y-axis label
    fLabelSizeX = labelsizex;   // Size of the x-axis label
    fLabelSizeY = labelsizey;   // Size of the y-axis label
    fDivisionsX = divx;         // The number of divisions in x-axis
    fDivisionsY = divy;         // The number of divisions in y-axis
    fFont = fontIndex;          // Font index for titles and labels
  }

  /*
   * Setter for the scale of the x-axis
   *
   * bool log = true, if logarithmic axis, false if linear axis
   */
  void SetLogX(bool log = true) {
    if (log) {
      fLogX = 1;
    } else {
      fLogX = 0;
    }
  }

  /*
   * Setter for the scale of the y-axis
   *
   * bool log = true, if logarithmic axis, false if linear axis
   */
  void SetLogY(bool log = true) {
    if (log) {
      fLogY = 1;
    } else {
      fLogY = 0;
    }
  }

  /*
   * Setter for the scale of the z-axis
   *
   * bool log = true, if logarithmic axis, false if linear axis
   */
  void SetLogZ(bool log = true) {
    if (log) {
      fLogZ = 1;
    } else {
      fLogZ = 0;
    }
  }

  // Getter for the displacement of the drawn canvas in x-direction
  int GetCanvasDisplacementX() { return fTopLeftX; }

  // Getter for the displacement of the drawn canvas in y-direction
  int GetCanvasDisplacementY() { return fTopLeftY; }

  // Setter for the displacement of the drawn canvas in x-direction
  void SetCanvasDisplacementX(int displacement) { fTopLeftX = displacement; }

  // Setter for the displacement of the drawn canvas in y-direction
  void SetCanvasDisplacementY(int displacement) { fTopLeftY = displacement; }

  /*
   * Setter for the displacement of the canvas from the top left corner
   *
   * int x = displacement in x-direction
   * int y = displacement in y-direction
   */
  void SetCanvasDisplacement(int x, int y) {
    SetCanvasDisplacementX(x);
    SetCanvasDisplacementY(y);
  }

  /*
   * Setter for displacement using an index
   *
   *  int index = index number for the canvas
   *
   *  The position of the canvas is decided in the following way:
   *  The displacement of two consecutive canvases in the x-direction is fCanvasDisplacementX.
   *  The canvases are put it one row, until there is fCanvasesInOneRow canvases in that row.
   *  After this, the next canvas is put in the next row fCanvasDisplacementY under the first
   *  canvas in the preceeding row. Then the next row is filled as the previous.
   */
  void SetCanvasDisplacement(int index) {
    SetCanvasDisplacementX(fCanvasDisplacementX * (index % fCanvasesInOneRow) + 10);
    SetCanvasDisplacementY(fCanvasDisplacementY * (index - index % fCanvasesInOneRow) / fCanvasesInOneRow + 10);
  }

  /*
   * Setter for canvas displacement settings
   *
   *  int displacementX = Displacement of the canvas for one index in x-direction
   *  int displacementY = Displacement of the canvas for one index in y-direction
   *  int canvasesInRow = Number of canvases in a row before a new row is started
   */
  void SetCanvasDisplacementSettings(int displacementX, int displacementY, int canvasesInRow) {
    fCanvasDisplacementX = displacementX;
    fCanvasDisplacementY = displacementY;
    fCanvasesInOneRow = canvasesInRow;
  }

  /*
   * Setter for number of canvases in one row
   *
   *  int canvasesInRow = Number of canvases in a row before a new row is started
   */
  void SetNumberOfCanvasesInOneRow(int canvasesInRow) { fCanvasesInOneRow = canvasesInRow; }

  /*
   * Setter for the size of the drawn canvas
   *
   * int width = width of the canvas
   * int height = height of the canvas
   */
  void SetCanvasSize(int width, int height) {
    fCanvasWidth = width;
    fCanvasHeight = height;
  }

  /*
   * Setter for grid in x-axis
   */
  void SetGridX(bool grid = true) {
    if (grid) {
      fGridX = 1;
    } else {
      fGridX = 0;
    }
  }

  /*
   * Setter for grid in y-axis
   */
  void SetGridY(bool grid = true) {
    if (grid) {
      fGridY = 1;
    } else {
      fGridY = 0;
    }
  }

  /*
   * Setter for grids in both axes
   */
  void SetGrid(int gridX, int gridY) {
    SetGridX(gridX);
    SetGridY(gridY);
  }

  /*
   * Setter for grid in x-axis
   */
  void SetTickX(bool tick = true) {
    if (tick) {
      fTickX = 1;
    } else {
      fTickX = 0;
    }
  }

  /*
   * Setter for grid in y-axis
   */
  void SetTickY(bool tick = true) {
    if (tick) {
      fTickY = 1;
    } else {
      fTickY = 0;
    }
  }

  /*
   * Setter for grids in both axes
   */
  void SetTick(int tickX, int tickY) {
    SetTickX(tickX);
    SetTickY(tickY);
  }

  /*
   * Setter for split ratio
   */
  void SetSplitRatio(double split) { fSplitRatio = split; }

  /*
   * Set the appearance values for splitted canvas as their default values
   * Different values than for single canvas are needed, so there is a different setter too
   *
   * Note: SetHistogramAppearance(titoffx, titoffy, titsizex, titsizey, labeloffx, labeloffy, labelsizex, labelsizey, divx, divy, fontIndex)
   */
  void SetDefaultAppearanceSplitCanvas() {
    Reset();
    //SetHistogramAppearance(2.5, 2.5, 20, 20, 0.01, 0.001, 16, 16, 505, 505, 43); // Smaller label size for axes
    SetHistogramAppearance(2.5, 2.3, 25, 25, 0.01, 0.001, 20, 20, 505, 505, 43);  // Increased label size for axes
    SetSplitRatio(0.4);
    SetRelativeCanvasSize(1.1, 0.7);
    SetMargins(0.21, 0.05, 0.09, 0.1);  // Left, right, top, bottom
  }

  /*
   * Set the appearance values for graph to default values
   */
  void SetDefaultAppearanceGraph() {
    Reset();
    SetHistogramAppearance(0.9, 1.3, 0.06, 0.05, 0.01, 0.001, 0.05, 0.05, 505, 505, 42);
    SetRelativeCanvasSize(0.8, 1);
    SetLeftMargin(0.17);
  }

  /*
   * Set the appearance values for row of graphs
   */
  void SetDefaultAppearanceGraphRow() {
    Reset();
    SetHistogramAppearance(0.9, 1.3, 0.06, 0.05, 0.01, 0.001, 0.05, 0.05, 505, 505, 42);
    SetRelativeCanvasSize(0.8, 3);
    SetLeftMargin(0.17);
  }

  /*
   * Getter for pad according to the ID number
   */
  TPad *GetPad(int padNumber) {
    switch (padNumber) {
      case 0:
        return fLeftRowPad;

      case 1:
        return fMiddleRowPad;

      case 2:
        return fRightRowPad;

      case 3:
        return fUpperSplitPad;

      case 4:
        return fLowerSplitPad;

      default:
        return nullptr;
    }
  }

  /*
   * Getter for pad according to the ID number
   */
  void SelectPad(int padNumber) {
    switch (padNumber) {
      case 0:
        fLeftRowPad->cd();
        break;

      case 1:
        fMiddleRowPad->cd();
        break;

      case 2:
        fRightRowPad->cd();
        break;

      case 3:
        fUpperSplitPad->cd();
        break;

      case 4:
        fLowerSplitPad->cd();
        break;

      default:
        std::cout << "Pad " << padNumber << " not found" << std::endl;
        break;
    }
  }

  /*
   * Getter for the upper pad
   */
  TPad *GetUpperPad() { return fUpperSplitPad; }

  /*
   * Getter for the lower pad
   */
  TPad *GetLowerPad() { return fLowerSplitPad; }

  /*
   * Select the upper pad
   */
  void SelectUpperPad() { fUpperSplitPad->cd(); }

  /*
   * Select the upper pad
   */
  void SelectLowerPad() { fLowerSplitPad->cd(); }

  /*
   * Set the canvas size using scaling factor and aspect ratio
   *
   *  double relativeSize = multiplication factor for the default canvas height of 600 pixels
   *  double aspectRatio = the ratio canvas width / canvas height
   */
  void SetRelativeCanvasSize(double relativeSize, double aspectRatio) {
    fCanvasHeight = 600 * relativeSize;
    fCanvasWidth = aspectRatio * fCanvasHeight;
  }

  /*
   * Get the name of the considered canvas
   */
  TString GetCanvasName() {
    if (fCanvas == nullptr) {
      std::cout << "Error: No canvas defined! Cannot return name." << std::endl;
      return "";
    }
    TString name = Form("%s", fCanvas->GetName());
    return name;
  }

  /*
   * Set the drawing range for a pad to draw geometric objects to it
   *
   *  double xLow = minimum x-axis range
   *  double yLow = minimum y-axis range
   *  double xHigh = maximum x-axis range
   *  double yHigh = maximum y-axis range
   */
  void SetPadRange(double xLow, double yLow, double xHigh, double yHigh) {
    // TODO: Setting range also for split canvas
    fSinglePad->Range(xLow, yLow, xHigh, yHigh);
    fSinglePad->Clear();
  }
};

#endif
