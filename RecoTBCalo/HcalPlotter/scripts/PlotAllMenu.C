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
#include "HistoManager.h"

#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

class PlotAllMenu {
  RQ_OBJECT("PlotAllMenu")
    private:
  TGMainFrame *fMain;
  TRootEmbeddedCanvas *fEcanvas;
  TGRadioButton* fET[HistoManager::NUMEVTTYPES];
  TGRadioButton* fFT[HistoManager::NUMHISTTYPES];
  TGRadioButton* fVT[2];
  TGRadioButton* fTBT[2];
  TGRadioButton* fDoE[1];
  TGCheckButton *checkbut,*elecbut,*topbut;
  TGTextEntry* iphiEntry, *ietaEntry, *runnoEntry, * fiberEntry, *fiberChanEntry, *crateEntry, *SlotEntry;
  PlotAllDisplay* theDisplay;
  const char *username_;
public:
  PlotAllMenu(const TGWindow *p,UInt_t w,UInt_t h, const char *username);
  virtual ~PlotAllMenu();
  void DoSelector();
  void DoCrateSelector();
  void DoDraw();
  void DoBrowse();
  void DoProcessRunNumber();
};

PlotAllMenu::PlotAllMenu(const TGWindow *p,UInt_t w,UInt_t h,
			 const char *username) {
  theDisplay=0;
  // Create a main frame
  fMain = new TGMainFrame(p,w,h);

  /***************************************************************
   * Create a vertical frame widget for File Selection objects   *
   ***************************************************************/

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

  /*************************************************************************
   * Create a vertical frame widget for Plot parameter Selection objects   *
   *************************************************************************/

  TGGroupFrame* plotselgf=new TGGroupFrame(fMain,"Plot Selection",kHorizontalFrame);
  //plotselgf->SetLayoutManager(new TGMatrixLayout(plotselgf,0,2,5,5));

  // first row:
  // Create Selection widget
  
  TGButtonGroup* ebg=new TGButtonGroup(plotselgf,"Event Type",kVerticalFrame);
  
  fET[0]=new TGRadioButton(ebg,"Other");
  fET[1]=new TGRadioButton(ebg,"Pedestal");
  fET[2]=new TGRadioButton(ebg,"LED");
  fET[3]=new TGRadioButton(ebg,"Laser");
  fET[4]=new TGRadioButton(ebg,"Beam");  

  plotselgf->AddFrame(ebg, new TGLayoutHints(kLHintsCenterX));  // ,5,5,3,4));

  TGButtonGroup* fbg=new TGButtonGroup(plotselgf,"Flavor Type",kVerticalFrame);
  fFT[0]=new TGRadioButton(fbg,"Energy");
  fFT[1]=new TGRadioButton(fbg,"Time");
  fFT[2]=new TGRadioButton(fbg,"Pulse Shape");
  fFT[3]=new TGRadioButton(fbg,"ADC");

  plotselgf->AddFrame(fbg, new TGLayoutHints(kLHintsCenterX)); // ,5,5,3,4));

  TGButtonGroup* vssbg=new TGButtonGroup(plotselgf,"VS Plot Stat",kHorizontalFrame);
  fVT[0]=new TGRadioButton(vssbg,"Mean");
  fVT[1]=new TGRadioButton(vssbg,"RMS");

  plotselgf->AddFrame(vssbg, new TGLayoutHints(kLHintsCenterX)); // ,5,5,3,4));

  fMain->AddFrame(plotselgf);

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
  
 fMain->AddFrame(gf);
  

  TGGroupFrame* ef=new TGGroupFrame(fMain,"Electronics Selection",kVerticalFrame);
  ef->SetLayoutManager(new TGMatrixLayout(ef,0,4,10,10));
  
  
 elecbut = new TGCheckButton(ef,"Use ElecId");
  ef->AddFrame(elecbut, new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  ef->AddFrame(new TGLabel(ef,""));

   
  ef->AddFrame(new TGLabel(ef," Fiber "));
  fiberEntry=new TGTextEntry(ef,"-1  ");
  ef->AddFrame(fiberEntry);
  ef->AddFrame(new TGLabel(ef," Fiber Chan"));
  fiberChanEntry=new TGTextEntry(ef,"-1  ");
  ef->AddFrame(fiberChanEntry);
  ef->AddFrame(new TGLabel(ef," Crate "));
  crateEntry=new TGTextEntry(ef," 2 ");
  ef->AddFrame(crateEntry);
  ef->AddFrame(new TGLabel(ef," HTR FPGA "));
  SlotEntry=new TGTextEntry(ef,"16b  ");
  ef->AddFrame(SlotEntry);

#if 0
  TGButtonGroup* topbot=new TGButtonGroup(ef,"Top or Bottom",kHorizontalFrame);
  fTBT[0]=new TGRadioButton(topbot,"Bottom");
  fTBT[1]=new TGRadioButton(topbot,"Top");

  ef->AddFrame(topbot, new TGLayoutHints(kLHintsCenterX));
#endif

 TGTextButton *CrateSelector = new TGTextButton(ef,"Visual Selector");
 CrateSelector->Connect("Clicked()","PlotAllMenu",this,"DoCrateSelector()");
  ef->AddFrame(CrateSelector,new TGLayoutHints(kLHintsCenterX,5,5,3,4));
  
  
    fMain->AddFrame(ef);
 
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
  int ipstat=0;
  for(int i=0; i<HistoManager::NUMEVTTYPES; i++) {
    if (fET[i]->IsOn()) iev=i;
    if (i<HistoManager::NUMHISTTYPES && fFT[i]->IsOn()) ifl=i;  
    if (i<2 && fVT[i]->IsOn()) ipstat=i;
  }
  theDisplay->displaySelector(iev,ifl,ipstat);
}

void PlotAllMenu::DoCrateSelector() {

  int iev=0;
  int ifl=0;
  int ipstat=0;
  for(int i=0; i<HistoManager::NUMEVTTYPES; i++) {
    if (fET[i]->IsOn()) iev=i;
    if (i<HistoManager::NUMHISTTYPES && fFT[i]->IsOn()) ifl=i;  
    if (i<2 && fVT[i]->IsOn()) ipstat=i;
  }
  
  int crate = atoi(crateEntry->GetText());
  if (crate ==-1){std::cout<<"Please enter a crate number to use Electronics Visual Selector"<<std::endl;
  }else{
    theDisplay->CrateDisplaySelector(crate,iev,ifl,ipstat);}
}


void PlotAllMenu::DoDraw() {
  int iev=0;
  int ifl=0;
  for (int i=0; i<HistoManager::NUMEVTTYPES; i++) {
    if (fET[i]->IsOn()) iev=i;
    if (i<HistoManager::NUMHISTTYPES && fFT[i]->IsOn()) ifl=i;
  }

  int ieta=atoi(ietaEntry->GetText());
  int iphi=atoi(iphiEntry->GetText());
  int fiber=atoi(fiberEntry->GetText());
  int fiberChan=atoi(fiberChanEntry->GetText());
  int crate=atoi(crateEntry->GetText());

  int slot=0,tb=0;
  char tbc='t';
  sscanf(SlotEntry->GetText(),"%d%c",&slot,&tbc);

  if (tbc=='t'){tb=1;}else{tb=0;}


  if(!elecbut->IsOn()){
    if (ieta==0 || iphi==0) {
      theDisplay->displaySummary(ieta,iphi,iev,ifl);
    } else {
      theDisplay->displayOne(ieta,iphi,1,iev,ifl);
    }
  }else {
    if (fiber==-1||fiberChan==-1){
     
      theDisplay->displayElecSummary(crate,slot,tb,iev,ifl);
    }else{
      theDisplay->displayElecOne(fiber,fiberChan,crate,slot,tb,iev,ifl);
    }
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
