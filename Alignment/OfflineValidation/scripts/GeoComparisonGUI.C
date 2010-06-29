

#include <TGClient.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TGraph.h>
#include <TRandom.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TFrame.h>
#include <TRootEmbeddedCanvas.h>
#include <RQ_OBJECT.h>


//////////////////////////////////////////////////////////////////////
//
// GeoComparisonGUI.C
//
// Geometry comparison GUI
//
// Usage is easy. Invoke the following command:
//
// root 'GeoComparisonGUI.C("geometry.root")'
// 
// where "geometry.root" is an output of the Geometry Comparison Tool.
//
// (in one line). Instead of the given geometry.root you can use any of the
// standard validation output geometry.root files. A canvas will open and draw
// the known 3x3 plots. In each of the nine plots, you can mark with the
// MIDDLE mouse button a region: Click the MIDDLE mouse button on the first
// edge, and drag to the second edge. (The region is only shown as a black box
// after you finish dragging). The, press the "draw" button at the bottom. The
// chosen modules will be shown in read in each of the nine views. That's
// all. You can zoom into the plots. 
// 
// N.B.: This tool uses TH2F with fine-grained binnig for displaying the
// modules.  This means that if you zoom in with a very high zoom factor, you
// are going to see the single bins. This can be annoying since the binning is
// different for all modules and the selected modules. Therefore, with a very
// high zoom factor, do not expect the distributions to match exactly.  There
// is a way out of that: The tool can be extended such that the second draw
// command uses the exact binning of the first draw command.
//
// Author      : Martin Weber
// Revision    : $Revision: 1.17 $
// Last update : $Date: 2009/10/10 14:17:44 $
// by          : $Author: bonato $
//
//////////////////////////////////////////////////////////////////////


// global variables, necessary for the 3x3 plots

// x-axis variable names
const char * varx[3] = { "r", "z", "phi" }; 
// y-axis variable names
const char * vary[3] = { "dr", "dz", "dphi" };
// selection to apply
char selection[255];
// name of input file
char * gName = 0;

// duplicate string with help of new
char * strdup_new(const char * text)
{
  char * temp =  new char[strlen(text)+1];
  strcpy(temp, text);
  return temp;
}

// This class contains the main GUI frame with all embedded canvases.
class MyMainFrame {
  RQ_OBJECT("MyMainFrame")
  private:
  TGMainFrame         * fMain;
  TRootEmbeddedCanvas * fEcanvas[3][3];

public:
  MyMainFrame(const TGWindow *p);
  virtual ~MyMainFrame();

  void DoDraw(); // main draw routine 
  void Reset();  // clear selection
  void Exec(Int_t event, Int_t x, Int_t y, TObject *selected);
};

void MyMainFrame::Exec(Int_t event, Int_t x, Int_t y, TObject *selected)
{
  static double x1, y1, x2, y2;
  static TGraph * gr = 0;
  TCanvas * c = (TCanvas *) gTQSender;
  // printf("Canvas %s: event=%d, x=%d, y=%d, selected=%s\n", c->GetName(),
  // 	 event, x, y, selected->IsA()->GetName());

  // get from current canvas the x and y indices necessary for drawing the variables
  int theNr = atoi(c->GetName()+7)-1;
  int xval = theNr % 3;
  int yval = theNr / 3;

  // printf("Canvas nr. %d, xval = %d, yval = %d\n", theNr, xval, yval);

  // when middle mouse button is pressed inside the histogram or frame, start marking area
  if (event == 2 && (selected->IsA() == TH2F::Class() || selected->IsA() == TFrame::Class())) {
    // 1 = left button pressed
    // 2 = middle button pressed
    x1 = c->AbsPixeltoX(x); 
    y1 = c->AbsPixeltoY(y);
  }

  // when dragging with mouse or in the moment of releasing, update the coordinates and draw a frame
  if ((event == 22 || event == 12) && ((selected->IsA() == TH2F::Class()) || selected->IsA() == TFrame::Class())) {
    // 21 = left button drag action, 11 = left button released
    // 22 = middle button drag action, 12 = middle button released

    // remove old rectangular area from plot
    if (gr != 0) {
      delete gr;
      gr = 0;
    }
    c->Modified();
    c->Update();

    // compute new coordinates for new area
    x2 = c->AbsPixeltoX(x); 
    y2 = c->AbsPixeltoY(y);
    // cout << "x1 = " << x1 << " y1 = " << y1 << " x2 = " << x2 << " y2 = " << y2 << endl;
    c->cd();

    // draw new rectangular area
    double xf[5] = { x1, x2, x2, x1, x1 };
    double yf[5] = { y1, y1, y2, y2, y1 };
    gr = new TGraph(5, xf, yf);
    gr->Draw();
    c->Update();

    // cout << "Draw frame" << endl;

    // when middle mouse button released, update the selection string with the
    // chosen coordinates
    if (event == 12) {
      // 11 = left button released
      // 12 = middle button released
      // cout << "creating selection" << endl;
      sprintf(selection, 
	      "(%s>%f)&&(%s<%f)&&(%s>%f)&&(%s<%f)",
	      vary[xval],
	      TMath::Min(y2, y1),
	      vary[xval],
	      TMath::Max(y2, y1),
	      varx[yval],
	      TMath::Min(x2, x1),
	      varx[yval],
	      TMath::Max(x2, x1));
      // cout << "selection = " << selection << endl;
    }
  }
}

MyMainFrame::MyMainFrame(const TGWindow *p) {
  // Create a main frame (vertical layout by default)
  fMain = new TGMainFrame(p, 800, 600);
  
  // Create first row horizontal frame
  TGHorizontalFrame * hframe = new TGHorizontalFrame(fMain,200,200);
  // Create three canvas widgets in row
  fEcanvas[0][0] = new TRootEmbeddedCanvas("Ecanvas1",hframe,200,200);
  hframe->AddFrame(fEcanvas[0][0], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[0][0]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  fEcanvas[1][0] = new TRootEmbeddedCanvas("Ecanvas2",hframe,200,200);
  hframe->AddFrame(fEcanvas[1][0], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[1][0]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  fEcanvas[2][0] = new TRootEmbeddedCanvas("Ecanvas3",hframe,200,200);
  hframe->AddFrame(fEcanvas[2][0], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[2][0]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  // add to main frame
  fMain->AddFrame(hframe, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					    5, 5, 5, 5));


  // Create second row horizontal frame
  TGHorizontalFrame * hframe = new TGHorizontalFrame(fMain,200,200);
  // Create three canvas widgets in row
  fEcanvas[0][1] = new TRootEmbeddedCanvas("Ecanvas4",hframe,200,200);
  hframe->AddFrame(fEcanvas[0][1], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[0][1]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  fEcanvas[1][1] = new TRootEmbeddedCanvas("Ecanvas5",hframe,200,200);
  hframe->AddFrame(fEcanvas[1][1], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[1][1]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  fEcanvas[2][1] = new TRootEmbeddedCanvas("Ecanvas6",hframe,200,200);
  hframe->AddFrame(fEcanvas[2][1], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[2][1]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  // add to main frame
  fMain->AddFrame(hframe, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					    5, 5, 5, 5));

  // Create third row horizontal frame
  TGHorizontalFrame * hframe = new TGHorizontalFrame(fMain,200,200);
  // Create three canvas widgets in row
  fEcanvas[0][2] = new TRootEmbeddedCanvas("Ecanvas7",hframe,200,200);
  hframe->AddFrame(fEcanvas[0][2], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[0][2]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  fEcanvas[1][2] = new TRootEmbeddedCanvas("Ecanvas8",hframe,200,200);
  hframe->AddFrame(fEcanvas[1][2], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[1][2]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  fEcanvas[2][2] = new TRootEmbeddedCanvas("Ecanvas9",hframe,200,200);
  hframe->AddFrame(fEcanvas[2][2], new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					       5, 5, 5, 5));
  fEcanvas[2][2]->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)", "MyMainFrame", "",
				       "Exec(Int_t,Int_t,Int_t,TObject*)");
  // add to main frame
  fMain->AddFrame(hframe, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
					    5, 5, 5, 5));

  // Create a horizontal frame widget with buttons
  hframe = new TGHorizontalFrame(fMain,200,40);
  TGTextButton *draw = new TGTextButton(hframe,"&Draw");
  draw->Connect("Clicked()","MyMainFrame",this,"DoDraw()");
  hframe->AddFrame(draw, new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  TGTextButton *reset = new TGTextButton(hframe,"&Reset");
  reset->Connect("Clicked()","MyMainFrame",this,"Reset()");
  hframe->AddFrame(reset, new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  TGTextButton *exit = new TGTextButton(hframe,"&Exit",
                                        "gApplication->Terminate(0)");
  // add buttons to frame
  hframe->AddFrame(exit, new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  // add button frame to main frame
  fMain->AddFrame(hframe, new TGLayoutHints(kLHintsCenterX,2,2,2,2));


  // Set a name to the main frame
  fMain->SetWindowName("Geometry Comparison GUI");
  // Map all subwindows of main frame
  fMain->MapSubwindows();
  // Initialize the layout algorithm
  fMain->Resize(fMain->GetDefaultSize());
  // Map main frame
  fMain->MapWindow();
}

// This is the main drawing routine. It draws to histograms in each canvas:
// First the geometry comparison for all modules, and in a different color on
// top the selected modules.
void MyMainFrame::DoDraw()
{
  // variables
  float dr, dphi, dz, r, phi, z;

  // open file and get tree from file
  TFile * infile = new TFile(gName);
  TTree * t = (TTree *) infile->Get("alignTree");
  if (t == 0) {
    cout << "ERR: could not find \"alignTree\" in file " << gName << endl;
    gApplication->Terminate(0);
  }
  // cout << "selection = " << selection << endl;


  // We need to store histogram names and the commands to pass to
  // TTree::Draw()
  char histname[20];
  char histname2[20];
  char drawstring[64];
  char drawstring2[64];

  // loop over the 3x3 matrix of plots
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // get and select canvas
      TCanvas * fCanvas = fEcanvas[i][j]->GetCanvas();
      fCanvas->cd();

      // prepare draw strings
      sprintf(histname, "%s vs %s", vary[i], varx[j]);
      sprintf(histname2, "%s vs %s_2", vary[i], varx[j]);
      sprintf(drawstring, "%s:%s>>%s", vary[i], varx[j], histname);
      sprintf(drawstring2, "%s:%s>>%s", vary[i], varx[j], histname2);
      // cout << "drawstring = " << drawstring << endl;

      // first plot: draw without selection, draw all modules into histogram. 
      t->Draw(drawstring, "", "goff");

      // get histogram and set draw options
      TObject * obj = gDirectory->Get(histname);
      // cout << "obj is a " << obj->IsA()->GetName() << endl;
      TH2F * h2 = (TH2F *) obj;
      if (h2 != 0) {
      	h2->SetTitle(histname);
      	h2->Draw();
      	h2->SetDirectory(0);
      	h2->SetMarkerStyle(1);
      	h2->Draw("p");
      }
      else {
      	cout << "Histo " << histname << " not found" << endl;
      }

      // second plot on top: use the selection
      t->Draw(drawstring2, selection, "goff");

      // get histogram and set draw options
      TH2F * h22 = (TH2F *) gDirectory->Get(histname2);
      if (h22 != 0) {
      	// cout << "Got histo" << endl;
      	h22->SetDirectory(0);
      	h22->SetTitle(histname2);
      	h22->SetMarkerColor(kRed);
      	h22->SetMarkerStyle(7);
      	h22->Draw("psame");
      }
      else {
      	cout << "Histo " << histname2 << " not found" << endl;
      }
      fCanvas->Update();
    }
  }
  // close file
  delete infile;
}

void MyMainFrame::Reset()
{
  // set the selection in such a way that no module gets selected
  strcpy(selection, "z==3000");
  DoDraw();
}

MyMainFrame::~MyMainFrame() 
{
  // Clean up used widgets: frames, buttons, layouthints
  fMain->Cleanup();
  delete fMain;
}

// main routine: set global options, create frame, draw initial plots
void GeoComparisonGUI(const char * fname) {
  // setup binning for plots
  gEnv->SetValue("Hist.Binning.2D.x", "1000");
  gEnv->SetValue("Hist.Binning.2D.y", "1000");
  gStyle->SetOptStat(0);
  // Popup the GUI...
  gName = strdup_new(fname);
  MyMainFrame * frame = new MyMainFrame(gClient->GetRoot());
  frame->Reset();
}
