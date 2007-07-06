// example.C
#include <TGClient.h>
#include <TCanvas.h>
#include <TF1.h>
#include <TRandom.h>
#include <TGButton.h>
#include <TGButtonGroup.h>
#include <TGTextEntry.h>
#include <TGFrame.h>
#include <TGLabel.h>
#include <TGFileDialog.h>
#include <TRootEmbeddedCanvas.h>
#include <RQ_OBJECT.h>
#include "PlotAllDisplay.h"

#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

class PlotAllMenu {
  RQ_OBJECT("PlotAllMenu")
    private:
  TGMainFrame *fMain;
  TRootEmbeddedCanvas *fEcanvas;
  TGRadioButton* fET[5];
  TGRadioButton* fFT[3];
  TGCheckButton *checkbut;
  TGTextEntry* iphiEntry, *ietaEntry, *runnoEntry;
  PlotAllDisplay* theDisplay;
  const char *username_;
public:
  PlotAllMenu(const TGWindow *p,UInt_t w,UInt_t h, const char *username);
  virtual ~PlotAllMenu();
  void DoSelector();
  void DoDraw();
  void DoBrowse();
  void DoProcessRunNumber();
};

PlotAllMenu::PlotAllMenu(const TGWindow *p,UInt_t w,UInt_t h,
			 const char *username) {
  theDisplay=0;
  // Create a main frame
  fMain = new TGMainFrame(p,w,h);

  // Create a vertical frame widget for File Selection objects
  TGGroupFrame* fileselgf=new TGGroupFrame(fMain,"File Selection",kVerticalFrame);
  fileselgf->SetLayoutManager(new TGMatrixLayout(fileselgf,0,2,5,5));

  // first row
  TGTextButton *processrn = new TGTextButton(fileselgf,"Process");
  processrn->Connect("Clicked()","PlotAllMenu",this,"DoProcessRunNumber()");
  fileselgf->AddFrame(processrn, new TGLayoutHints(kLHintsCenterX,5,5,3,4));

  TGHorizontalFrame *hframe1 = new TGHorizontalFrame(fileselgf,200,40);
  hframe1->AddFrame(new TGLabel(hframe1," Run Number "));
  runnoEntry=new TGTextEntry(hframe1,"12345   ");
  hframe1->AddFrame(runnoEntry);

  fileselgf->AddFrame(hframe1, new TGLayoutHints(kLHintsCenterX,5,5,3,4));

  // second row
  fileselgf->AddFrame(new TGLabel(fileselgf," or "));
  fileselgf->AddFrame(new TGLabel(fileselgf," "));

  // third row
  TGTextButton *browse = new TGTextButton(fileselgf,"Browse for file...");
  browse->Connect("Clicked()","PlotAllMenu",this,"DoBrowse()");
  fileselgf->AddFrame(browse, new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  checkbut = new TGCheckButton(fileselgf,"Force re-processing");
  fileselgf->AddFrame(checkbut, new TGLayoutHints(kLHintsCenterX,5,5,3,4));

  fMain->AddFrame(fileselgf, new TGLayoutHints(kLHintsCenterX,2,2,2,2));

  // Create Selection widget
  
  TGButtonGroup* br=new TGButtonGroup(fMain,"Event Type",kVerticalFrame);
  
  fET[0]=new TGRadioButton(br,"Other");
  fET[1]=new TGRadioButton(br,"Pedestal");
  fET[2]=new TGRadioButton(br,"LED");
  fET[3]=new TGRadioButton(br,"Laser");
  fET[4]=new TGRadioButton(br,"Beam");  

  TGButtonGroup* fbgr=new TGButtonGroup(fMain,"Flavor Type",kVerticalFrame);
  fFT[0]=new TGRadioButton(fbgr,"Energy");
  fFT[1]=new TGRadioButton(fbgr,"Time");
  fFT[2]=new TGRadioButton(fbgr,"Pulse Shape");

  TGGroupFrame* gf=new TGGroupFrame(fMain,"Channel Selection",kVerticalFrame);
  gf->SetLayoutManager(new TGMatrixLayout(gf,0,2,10,10));
  gf->AddFrame(new TGLabel(gf," IPhi "));
  iphiEntry=new TGTextEntry(gf,"0   ");
  gf->AddFrame(iphiEntry);
  gf->AddFrame(new TGLabel(gf," IEta "));
  ietaEntry=new TGTextEntry(gf,"0   ");
  gf->AddFrame(ietaEntry);
  TGTextButton *selector = new TGTextButton(gf,"Visual Selector");
  selector->Connect("Clicked()","PlotAllMenu",this,"DoSelector()");
  gf->AddFrame(selector,new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  
  fMain->AddFrame(br);
  fMain->AddFrame(fbgr);
  fMain->AddFrame(gf);

  // Create a horizontal frame widget with buttons
  TGHorizontalFrame *hframe = new TGHorizontalFrame(fMain,200,40);

  TGTextButton *draw = new TGTextButton(hframe,"&Draw");
  draw->Connect("Clicked()","PlotAllMenu",this,"DoDraw()");
  hframe->AddFrame(draw, new TGLayoutHints(kLHintsCenterX,5,5,3,4));


  TGTextButton *exit = new TGTextButton(hframe,"&Exit",
					"gApplication->Terminate(0)");
  hframe->AddFrame(exit, new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  fMain->AddFrame(hframe, new TGLayoutHints(kLHintsCenterX,2,2,2,2));
  // Set a name to the main frame
  fMain->SetWindowName("Plot Menu for HCAL");
  // Map all subwindows of main frame
  fMain->MapSubwindows();
  // Initialize the layout algorithm
  fMain->Resize(fMain->GetDefaultSize());
  // Map main frame
  fMain->MapWindow();

  username_ = username;
}

void PlotAllMenu::DoSelector() {

  int iev=0;
  int ifl=0;
  for(int i=0; i<=4; i++) {
    if (fET[i]->IsOn()) iev=i;
    if (i<3 && fFT[i]->IsOn()) ifl=i;  
  }
  theDisplay->displaySelector(iev,ifl);
}
void PlotAllMenu::DoDraw() {
  int iev=0;
  int ifl=0;
  for (int i=0; i<=4; i++) {
    if (fET[i]->IsOn()) iev=i;
    if (i<3 && fFT[i]->IsOn()) ifl=i;
  }

  int ieta=atoi(ietaEntry->GetText());
  int iphi=atoi(iphiEntry->GetText());

  if (ieta==0 || iphi==0) {
    theDisplay->displaySummary(ieta,iphi,iev,ifl);
  } else {
    theDisplay->displayOne(ieta,iphi,1,iev,ifl);
  }
}

static const char *filetypes[] = { "POOL files",    "*.root", 0,  0 };

void PlotAllMenu::DoBrowse() {
  char line[1200];
  char outfn[1200];

  static TString dir(".");
  TGFileInfo fi;
  fi.fFileTypes = filetypes;
  fi.fIniDir    = StrDup(dir);
  new TGFileDialog(gClient->GetRoot(), fMain, kFDOpen, &fi);

  // Compose temporary root output filename
  strcpy(line,fi.fFilename);
  *(strstr(line,".root")) = 0;         // null out ".root" suffix

 // get rid of path prefix, put it in tmp
  char *fn=strrchr(line,'/');
  if (fn != NULL)
    sprintf (outfn,"/tmp/%s.%s.plotall.root", fn+1, username_);
  else
    sprintf (outfn,"/tmp/%s.%s.plotall.root", fi.fFilename, username_);

  struct stat buf;
  if (!checkbut->IsOn() &&
      !stat(outfn,&buf)) {
    std::cout << "File already processed, loading results." << std::endl;
  }
  else {
    std::cout << "Processing..." << std::endl;
    sprintf(line,".! ./runCMSSWReco.sh %s %s",fi.fFilename, outfn);
    gROOT->ProcessLine(line);
    std::cout << "Done." << std::endl;
  }

  if (theDisplay!=0) delete theDisplay;
  theDisplay=new PlotAllDisplay(outfn);
  
  dir = fi.fIniDir;
}

void PlotAllMenu::DoProcessRunNumber() {
  char line[1200];
  char outfn[1200];
  int runno=atoi(runnoEntry->GetText());

  // Compose temporary root output filename
  sprintf (outfn,"/tmp/%s.%d.plotall.root", username_,runno);

  struct stat buf;
  if (!checkbut->IsOn() &&
      !stat(outfn,&buf)) {
    std::cout << "File already processed, loading results." << std::endl;
  }
  else {
    std::cout << "Processing..." << std::endl;
    sprintf(line,".! ./runCMSSWReco.sh %d %s",runno, outfn);
    gROOT->ProcessLine(line);
    std::cout << "Done." << std::endl;
  }

  if (theDisplay!=0) delete theDisplay;
  theDisplay=new PlotAllDisplay(outfn);
}

PlotAllMenu::~PlotAllMenu() {
  // Clean up used widgets: frames, buttons, layouthints
  fMain->Cleanup();
  delete fMain;
}
