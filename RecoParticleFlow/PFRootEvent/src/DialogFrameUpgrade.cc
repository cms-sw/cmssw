#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "RecoParticleFlow/PFProducer/interface/PFGeometry.h"
#include "RecoParticleFlow/PFRootEvent/interface/DialogFrameUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayCommon.h"

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManagerUpgrade.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManagerUpgrade.h"

#include <TROOT.h>
#include <TTree.h>
#include "TLine.h"
#include "TList.h"
#include "TCanvas.h"
#include "TButton.h"
#include "TGraph.h"
#include "TMarker.h"
#include "TText.h"
#include <TApplication.h>




DialogFrameUpgrade::DialogFrameUpgrade(PFRootEventManagerUpgrade *evman,DisplayManagerUpgrade *dm,const TGWindow *p,UInt_t w,UInt_t h)
  :TGMainFrame(p, w, h),evMan_(evman),display_(dm),attrView_(0),
   thisClusPattern_(0),trackPattern_(0),simplePartPattern_(0)
{
  
  mainFrame_= new TGCompositeFrame(this,200,300,kVerticalFrame);
  createCmdFrame();
  AddFrame(mainFrame_, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));
  
  // Set a name to the main frame
  SetWindowName("PFRootEvent Dialog");
  // Map all subwindows of main frame
  MapSubwindows();
  // Initialize the layout algorithm
  Resize(mainFrame_->GetDefaultSize());
  // Map main frame
  MapWindow();
    
}
//__________________________________________________________________________________________________
void DialogFrameUpgrade::closeCanvasAttr()
{
  if (gROOT->GetListOfCanvases()->FindObject("graphic Attributes")) 
    attrView_->Close();
  attrView_=0;
  thisClusPattern_=0;
  trackPattern_=0;
  simplePartPattern_=0;
}
//__________________________________________________________________________________________________
void DialogFrameUpgrade::createCanvasAttr()
{
  //creates an attributes canvas which enable modifications
  //of cluster and track graphic attributes 
  
  closeCanvasAttr(); 
  attrView_=0;
  attrView_ = new TCanvas("graphic Attributes","graphic Attributes",180,300);
  const char *action1="win->DialogFrameUpgrade::modifyAttr()";
  const char *action2="win->DialogFrameUpgrade::closeCanvasAttr()";
  TButton *accept_ = new TButton("modify",action1,0.1,0.2,0.5,0.3);
  TButton *cancel_ = new TButton("cancel",action2,0.54,0.2,0.9,0.3);
  double x[3];
  double y[3];
  x[0]=0.1;x[1]=0.3;x[2]=0.5;
  y[0]=0.6;y[1]=0.6;y[2]=0.6;
  thisClusPattern_= new TMarker(0.3,0.8,display_->clusPattern_->GetMarkerStyle());
  thisClusPattern_->SetMarkerColor(display_->clusPattern_->GetMarkerColor());
  thisClusPattern_->SetMarkerSize(display_->clusPattern_->GetMarkerSize());
  thisClusPattern_->Draw();
  TText * tt=new TText(0.6,0.78,"clusters");
  tt->SetTextSize(.08);
  tt->Draw();
  trackPattern_= new TGraph(3,x,y);
  trackPattern_->SetLineColor(display_->trackPatternL_->GetLineColor());
  trackPattern_->SetMarkerColor(display_->trackPatternM_->GetMarkerColor());
  trackPattern_->SetMarkerStyle(display_->trackPatternM_->GetMarkerStyle());
  trackPattern_->SetMarkerSize(display_->trackPatternM_->GetMarkerSize());
  trackPattern_->Draw("pl");
  TText *tt2= new TText(0.6,0.58,"recTracks");
  tt2->SetTextSize(.08);
  tt2->Draw();
 
  accept_->Draw();
  cancel_->Draw();
  attrView_->Update();
}
//__________________________________________________________________________________________________
void DialogFrameUpgrade::createCmdFrame() 
{
  TGCompositeFrame *h1Frame1 = new TGCompositeFrame(mainFrame_, 100, 100, kHorizontalFrame | kRaisedFrame);
  mainFrame_->AddFrame(h1Frame1,new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
  TGLayoutHints *lo;
  lo = new TGLayoutHints(kLHintsLeft|kLHintsExpandX |kLHintsExpandY ,5,2,5,2);
  
  //create object selection buttons
  TGGroupFrame *gr1= new TGGroupFrame(h1Frame1,"Draw Selection",kVerticalFrame); 
  gr1->SetLayoutManager(new TGMatrixLayout(gr1,9,3,5));
  
  selectObject_[0] = new TGCheckButton(gr1,"Hits");
  selectObject_[0]->SetState(display_->drawHits_ ? kButtonDown :kButtonUp);
  selectObject_[0]->Connect("Clicked()","DialogFrameUpgrade",this,"doModifyOptions(=0)");
  selectObject_[1] = new TGCheckButton(gr1,"Clusters");
  selectObject_[1]->SetState(display_->drawClus_ ? kButtonDown :kButtonUp);
  selectObject_[1]->Connect("Clicked()","DialogFrameUpgrade",this,"doModifyOptions(=1)");
  selectObject_[2] = new TGCheckButton(gr1,"Tracks");
  selectObject_[2]->SetState(display_->drawTracks_ ? kButtonDown :kButtonUp);
  selectObject_[2]->Connect("Clicked()","DialogFrameUpgrade",this,"doModifyOptions(=2)");
  selectObject_[3] = new TGCheckButton(gr1,"SimParticles");
  selectObject_[3]->SetState(display_->drawParticles_ ? kButtonDown :kButtonUp);
  selectObject_[3]->Connect("Clicked()","DialogFrameUpgrade",this,"doModifyOptions(=3)");
  selectObject_[4] = new TGCheckButton(gr1,"GenParticles");
  selectObject_[4]->SetState(display_->drawGenParticles_ ? kButtonDown :kButtonUp);
  selectObject_[4]->Connect("Clicked()","DialogFrameUpgrade",this,"doModifyOptions(=4)");
  selectObject_[5] = new TGCheckButton(gr1,"GsfTracks");
  selectObject_[5]->SetState(display_->drawGsfTracks_ ? kButtonDown :kButtonUp);
  selectObject_[5]->Connect("Clicked()","DialogFrameUpgrade",this,"doModifyOptions(=5)");
  selectObject_[6] = new TGCheckButton(gr1,"Brems visible");
  selectObject_[6]->SetState(display_->drawBrems_ ? kButtonDown :kButtonUp);
  selectObject_[6]->Connect("Clicked()","DialogFrameUpgrade",this,"areBremVisible()");
  selectObject_[7] = new TGCheckButton(gr1,"PFBlock visible");
  selectObject_[7]->SetState(display_->drawPFBlocks_ ? kButtonDown :kButtonUp);
  selectObject_[7]->Connect("Clicked()","DialogFrameUpgrade",this,"isPFBlockVisible()");


  // create threshold fields
  TGNumberFormat::ELimit lim = TGNumberFormat::kNELLimitMinMax;  
  float limit=100;
  for (int i=0;i<6;++i){
    thresholdS_[i] = new TGDoubleHSlider(gr1,100,kDoubleScaleNo,ENER+i);
    thresholdS_[i]->Associate(this);
    thresholdS_[i]->SetRange(0,limit);
   
    threshEntry_[i] = new TGNumberEntryField(gr1,EN+i,0);
    threshEntry_[i]->Associate(this);
    threshEntry_[i]->SetLimits(lim,0,limit);
    threshEntry_[i]->SetFormat((TGNumberFormat::EStyle)2);
  }
  thresholdS_[0]->SetPosition((float) display_->hitEnMin_,(float) display_->hitEnMin_);
  thresholdS_[1]->SetPosition((float) display_->clusEnMin_,(float)display_->clusEnMin_);
  thresholdS_[2]->SetPosition((float) display_->trackPtMin_,(float)display_->trackPtMin_);
  thresholdS_[3]->SetPosition((float) display_->particlePtMin_,(float)display_->particlePtMin_);
  thresholdS_[4]->SetPosition((float) display_->genParticlePtMin_,(float)display_->genParticlePtMin_);
  thresholdS_[5]->SetPosition((float) display_->gsfPtMin_,(float)display_->gsfPtMin_);
  
  
  int charw= threshEntry_[0]->GetCharWidth("O");
  int size=charw*4;
  for (int i=0;i<6;++i) {
    threshEntry_[i]->SetNumber(thresholdS_[i]->GetMinPosition());
    threshEntry_[i]->Resize(size,threshEntry_[i]->GetDefaultHeight());
  }
  
  //
  TGLayoutHints *lo1=new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 2, 2);
  TGLabel *label=new TGLabel(gr1,"  ");
  gr1->AddFrame(label,lo1);
  label=new TGLabel(gr1," En/Pt  Threshold");
  gr1->AddFrame(label,lo1);
  label=new TGLabel(gr1," (Gev) ");  
  gr1->AddFrame(label,lo1);
  
  for (int i=0;i<6;++i) {
    gr1->AddFrame(selectObject_[i],lo1);
    gr1->AddFrame(thresholdS_[i],lo1);
    gr1->AddFrame(threshEntry_[i],lo1);
  }
  
//gr1->AddFrame(selectObject_[6],lo1);   
//  h1Frame1->AddFrame(gr1,lo1);
//  gr1->AddFrame(selectObject_[7],lo1);   
//  h1Frame1->AddFrame(gr1,lo1);

// MURIEL - replace the previous four  lines by :
  TGLabel *label0 = new TGLabel(gr1,"  ");
  gr1->AddFrame(selectObject_[6],lo1);   
  gr1->AddFrame(label0,lo1);   
  gr1->AddFrame(label0,lo1);
  gr1->AddFrame(selectObject_[7],lo1);   
  h1Frame1->AddFrame(gr1,lo);
  
  
  //add options frame
  TGVerticalFrame *optionFrame = new TGVerticalFrame(h1Frame1,10,10,kSunkenFrame);
  
  //print space
  TGLabel *lab1,*lab2;
  TGHorizontalFrame *h2 = new TGHorizontalFrame(optionFrame,10,10);
  TGGroupFrame *printGroup = new TGGroupFrame(h2, " Print", kVerticalFrame);
  lab1 = new TGLabel(printGroup," ");
  lab2 = new TGLabel(printGroup," ");
  //TGLabel *lab3 = new TGLabel(printGroup," ");
  printGroup->SetLayoutManager(new TGMatrixLayout(printGroup, 7,3,3));
  printButton_[0] = new TGCheckButton(printGroup,"RecHits ");
  printButton_[0]->SetState(evMan_->printRecHits_ ? kButtonDown :kButtonUp); 
  printButton_[0]->Connect("Clicked()","DialogFrameUpgrade",this,"selectPrintOption(=0)");
  printButton_[1] = new TGCheckButton(printGroup,"Clusters");
  printButton_[1]->SetState(evMan_->printClusters_ ? kButtonDown :kButtonUp); 
  printButton_[1]->Connect("Clicked()","DialogFrameUpgrade",this,"selectPrintOption(=1)");
  printButton_[2] = new TGCheckButton(printGroup,"PFBlocks");
  printButton_[2]->SetState(evMan_->printPFBlocks_ ? kButtonDown :kButtonUp); 
  printButton_[2]->Connect("Clicked()","DialogFrameUpgrade",this,"selectPrintOption(=2)");
  printButton_[3] = new TGCheckButton(printGroup,"PFCandidates ");
  printButton_[3]->SetState(evMan_->printPFCandidates_ ? kButtonDown :kButtonUp); 
  printButton_[3]->Connect("Clicked()","DialogFrameUpgrade",this,"selectPrintOption(=3)");
  printButton_[4] = new TGCheckButton(printGroup,"PFJets ");
  printButton_[4]->SetState(evMan_->printPFJets_ ? kButtonDown :kButtonUp); 
  printButton_[4]->Connect("Clicked()","DialogFrameUpgrade",this,"selectPrintOption(=4)");
  printButton_[5] = new TGCheckButton(printGroup,"SimParticles ");
  printButton_[5]->SetState(evMan_->printSimParticles_ ? kButtonDown :kButtonUp); 
  printButton_[5]->Connect("Clicked()","DialogFrameUpgrade",this,"selectPrintOption(=5)");
  printButton_[6] = new TGCheckButton(printGroup,"GenParticles");
  TGLabel *maxl = new TGLabel(printGroup,"max lines:");
  maxLineEntry_= new TGNumberEntryField(printGroup,MAXL,30);
  maxLineEntry_->Associate(this);
  maxLineEntry_->SetFormat((TGNumberFormat::EStyle)0);
  maxLineEntry_->Resize(charw*3,maxLineEntry_->GetDefaultHeight());
  printButton_[6]->SetState(evMan_->printGenParticles_ ? kButtonDown :kButtonUp); 
  printButton_[6]->Connect("Clicked()","DialogFrameUpgrade",this,"selectPrintOption(=6)");
  
    
  for(UInt_t i = 0 ;i<6 ; ++i){
    printGroup->AddFrame(printButton_[i],lo1);
    printGroup->AddFrame(lab1,lo1);
    printGroup->AddFrame(lab2,lo1);
  }
  printGroup->AddFrame(printButton_[6],lo1);
  printGroup->AddFrame(maxl,lo1);
  printGroup->AddFrame(maxLineEntry_,lo1);
  
  
  TGTextButton *sendPrintButton = new TGTextButton(h2,"Print");
  sendPrintButton->Connect("Clicked()","DialogFrameUpgrade",this,"doPrint()");
  
  h2->AddFrame(printGroup,lo1);
  h2->AddFrame(sendPrintButton,new TGLayoutHints(kLHintsLeft|kLHintsCenterY,2,2,2,2));
  
  TGGroupFrame *viewGroup = new TGGroupFrame(optionFrame,"View",kHorizontalFrame);
  lab1 = new TGLabel(viewGroup," ");
  lab2 = new TGLabel(viewGroup," ");
  viewGroup->SetLayoutManager(new TGMatrixLayout(viewGroup, 3,3,3));
  
  TGTextButton *lookFor = new TGTextButton(viewGroup,"Look for");
  lookFor->Connect("Clicked()","DialogFrameUpgrade",this,"doLookForGenParticle()");
  TGLabel *genPartNb = new TGLabel(viewGroup,"Gen Particle Nb:");
  particleTypeEntry_ = new TGNumberEntryField(viewGroup,PARTTYPE,1);
  particleTypeEntry_->Associate(this);
  particleTypeEntry_->SetFormat((TGNumberFormat::EStyle)0);
  particleTypeEntry_->Resize(charw*3,particleTypeEntry_->GetDefaultHeight());
 
  TGTextButton *unZoomButton = new TGTextButton(viewGroup,"Unzoom");
  unZoomButton->Connect("Clicked()","DialogFrameUpgrade",this,"unZoom()");

  TGTextButton *newAttrBis = new TGTextButton(viewGroup,"Modify Graphic Attr");
  newAttrBis->Connect("Clicked()","DialogFrameUpgrade",this,"createCanvasAttr()");
  
  viewGroup->AddFrame(lookFor,lo1);
  viewGroup->AddFrame(genPartNb,lo1),
    viewGroup->AddFrame(particleTypeEntry_,lo1);
  viewGroup->AddFrame(unZoomButton,lo1);
  viewGroup->AddFrame(lab1,lo1);
  viewGroup->AddFrame(lab2,lo1);
  viewGroup->AddFrame(newAttrBis,lo1); 
  
  //
  optionFrame->AddFrame(h2,lo);
  optionFrame->AddFrame(viewGroup,lo1);
  h1Frame1->AddFrame(optionFrame,lo);

  
  // Next/Pevious/exit buttons
  
  TGHorizontalFrame *h1 = new TGHorizontalFrame(mainFrame_,20,30);
  mainFrame_->AddFrame(h1,new TGLayoutHints(kLHintsCenterX,2,2,2,2));
  
  nextButton = new TGTextButton(h1,"Draw Next");
  nextButton->Connect("Clicked()","DialogFrameUpgrade",this,"doNextEvent()");
  h1->AddFrame(nextButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
  
  previousButton = new TGTextButton(h1,"Draw Previous");
  previousButton->Connect("Clicked()","DialogFrameUpgrade",this,"doPreviousEvent()");
  h1->AddFrame(previousButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
  
  
  reProcessButton = new TGTextButton(h1,"Re-Process");
  reProcessButton->Connect("Clicked()","DialogFrameUpgrade",this,"doReProcessEvent()");
  h1->AddFrame(reProcessButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
  
  //Modifie Graphic attributes in option file
  //  TGTextButton *newAttr = new TGTextButton(h1,"new GAttr");
  //  newAttr->Connect("Clicked()","DialogFrameUpgrade",this,"modifyGraphicAttributes()");
  //  h1->AddFrame(newAttr,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));

    
  exitButton = new TGTextButton(h1,"&Exit","gApplication->Terminate(0)");
  h1->AddFrame(exitButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
}  
  
//________________________________________________________________________________
void DialogFrameUpgrade::CloseWindow()
{
  //!!!WARNING keep the first letter of the method uppercase.It is an overriden ROOT method  
  gApplication->Terminate(0);
}
//_________________________________________________________________________________
void DialogFrameUpgrade::doLookForGenParticle()
{
  int num = particleTypeEntry_->GetIntNumber();
  display_->lookForGenParticle((unsigned)num);
}

//_________________________________________________________________________________
void DialogFrameUpgrade::doModifyOptions(unsigned objNb)
{
  switch (objNb) {
  case 0:
    display_->drawHits_ = (selectObject_[0]->IsDown()) ?true :false;
    break;
  case 1:
    display_->drawClus_ = (selectObject_[1]->IsDown()) ?true :false;
    break; 
  case 2:
    display_->drawTracks_ = (selectObject_[2]->IsDown()) ?true :false;
    break;
  case 3: 
    display_->drawParticles_ = (selectObject_[3]->IsDown()) ?true :false;
    break;
  case 4:
    display_->drawGenParticles_ = (selectObject_[4]->IsDown()) ?true :false;
    break;    
  case 5:
    display_->drawGsfTracks_ = (selectObject_[5]->IsDown()) ?true :false;
    break;    
  }
  display_->displayAll();    
}
//_______________________________________________________________________________
DialogFrameUpgrade::~DialogFrameUpgrade()
{
  mainFrame_->Cleanup();
}
//________________________________________________________________________________
void DialogFrameUpgrade::doModifyPtThreshold(unsigned objNb,double pt)
{
  switch(objNb) {
  case 0: 
    display_->hitEnMin_= pt;break;
  case 1:
    display_->clusEnMin_= pt;break;
  case 2:
    display_->trackPtMin_= pt;break;
  case 3:
    display_->particlePtMin_= pt;break;
  case 4:
    display_->genParticlePtMin_= pt;break;
  case 5:
    display_->gsfPtMin_= pt;break;
    
  default:break;
  }  
  display_->displayAll();
}
//_________________________________________________________________________________
void DialogFrameUpgrade::doNextEvent()
{
  display_->displayNext();
  doLookForGenParticle();  
  //   int eventNumber = evMan_->eventNumber();
  //TODOLIST:display new value of eventNumber in the futur reserve field
} 
//_________________________________________________________________________________
void DialogFrameUpgrade::doPreviousEvent()
{
  display_->displayPrevious();
  doLookForGenParticle();  
  //   int eventNumber = evMan_->eventNumber();
  //TODOLIST:display new value of eventNumber in the futur reserve field
}
//_________________________________________________________________________________
void DialogFrameUpgrade::doPrint()
{
  evMan_->print(std::cout,maxLineEntry_->GetIntNumber());
}
//________________________________________________________________________________
void DialogFrameUpgrade::doPrintGenParticles()
{
  evMan_->printGenParticles(std::cout,maxLineEntry_->GetIntNumber());
}
//_________________________________________________________________________________
void DialogFrameUpgrade::doReProcessEvent()
{
  int eventNumber = evMan_->eventNumber();
  display_->display(eventNumber);
}
//_________________________________________________________________________________
void DialogFrameUpgrade::isPFBlockVisible()
{
  display_->enableDrawPFBlock((selectObject_[7]->IsDown()) ?true :false);
  
}
//_________________________________________________________________________________
void DialogFrameUpgrade::areBremVisible()
{
  display_->enableDrawBrem((selectObject_[6]->IsDown()) ?true :false);
  display_->displayAll();
}


//_________________________________________________________________________________
void DialogFrameUpgrade::selectPrintOption(int opt)
{
  switch (opt) {
  case 0:
    evMan_->printRecHits_ = (printButton_[0]->IsDown()) ?true :false;
    break;
  case 1:
    evMan_->printClusters_ = (printButton_[1]->IsDown()) ?true :false;
    break;
  case 2:
    evMan_->printPFBlocks_ = (printButton_[2]->IsDown()) ?true :false;
    break;
  case 3:
    evMan_->printPFCandidates_ = (printButton_[3]->IsDown()) ?true :false;
    break;
  case 4:
    evMan_->printPFJets_ = (printButton_[4]->IsDown()) ?true :false;
    break;
  case 5:
    evMan_->printSimParticles_ = (printButton_[5]->IsDown()) ?true :false;
    break;
  case 6:
    evMan_->printGenParticles_ = (printButton_[6]->IsDown()) ?true :false;
    break;
  default: break;  
    
  }
} 
//________________________________________________________________________________
void DialogFrameUpgrade::updateDisplay()
{
  display_->updateDisplay();
}

//________________________________________________________________________________
void DialogFrameUpgrade::unZoom()
{
  display_->unZoom();
}
//________________________________________________________________________________
/*void DialogFrameUpgrade::modifyGraphicAttributes()
  {
  // readOption avec nom du fichier apres valeurs changees a la main
  std::cout <<"do it yourself in the root input window"<<std::endl;
  std::cout <<"Edit your option file "<<std::endl;
  std::cout <<"modify the clusAttributes, trackAttributes or simpleTrackAttributes "<<std::endl;
  std::cout <<"type :dm->readOptions(opt.c_str();"<<std::endl;
  } 
*/
//______________________________________________________________________________________
void DialogFrameUpgrade::modifyAttr()
{ 
  display_->clusPattern_->SetMarkerStyle(thisClusPattern_->GetMarkerStyle());
  display_->clusPattern_->SetMarkerSize(thisClusPattern_->GetMarkerSize());
  display_->clusPattern_->SetMarkerColor(thisClusPattern_->GetMarkerColor());
  display_->trackPatternL_->SetLineColor(trackPattern_->GetLineColor());
  display_->trackPatternM_->SetMarkerStyle(trackPattern_->GetMarkerStyle());
  display_->trackPatternM_->SetMarkerSize(trackPattern_->GetMarkerSize());
  //trackPattern_->SetMarkerColor(display_->trackAttributes_[0]);
  closeCanvasAttr();
  display_->drawWithNewGraphicAttributes();
}
//_________________________________________________________________________________
Bool_t DialogFrameUpgrade::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{ 
  switch (GET_MSG(msg)) {
  case kC_TEXTENTRY:
    switch (GET_SUBMSG(msg)) {
    case kTE_ENTER:
      switch (parm1) {
      case EN :case EN+1: case EN+2: case EN+3: case EN+4: case EN+5:
        {
          //int eventNumber=evMan_->iEvent_;
          float val=threshEntry_[parm1-EN]->GetNumber();
          thresholdS_[parm1-EN]->SetPosition(val,val);
          doModifyPtThreshold(parm1-EN,val);
          break;
        }
      case MAXL:  // print genPart max lines
        evMan_->printGenParticles_ = true;
        printButton_[6]->SetState(kButtonDown);
        doPrintGenParticles();
        break;
      case PARTTYPE:
        doLookForGenParticle(); 
        break;
      default:break;
      }
      break;
    default:break;
    }
    break;
  case kC_HSLIDER:
    switch (GET_SUBMSG(msg)) {
    case kSL_POS:
      switch (parm1) {
      case ENER: case ENER+1: case ENER+2: case ENER+3: case ENER+4: case ENER+5:
        {
          unsigned index=parm1-ENER;
          float val = thresholdS_[index]->GetMinPosition();
          threshEntry_[index]->SetNumber(val);
          fClient->NeedRedraw(threshEntry_[index]);
          break;
        } 
      default:break;
      }
      break;  
    case kSL_RELEASE:
      switch (parm1) {
      case ENER: case ENER+1: case ENER+2: case ENER+3:case ENER+4: case ENER+5:
        {
          float val = thresholdS_[parm1-ENER]->GetMinPosition();
          doModifyPtThreshold(parm1-ENER,(double)val);
          break;
        } 
      default:break;    
      }
      break;
    default:break;       
    }
    break;
  default:break;
  }
  return true;   
}             
