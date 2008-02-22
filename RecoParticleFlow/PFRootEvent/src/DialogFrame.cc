#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "RecoParticleFlow/PFBlockAlgo/interface/PFGeometry.h"
#include "RecoParticleFlow/PFRootEvent/interface/DialogFrame.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayCommon.h"
#include <TTree.h>
#include "TLine.h"
#include "TList.h"
#include <TApplication.h>



DialogFrame::DialogFrame(PFRootEventManager *evman,DisplayManager *dm,const TGWindow *p,UInt_t w,UInt_t h)
                         :TGMainFrame(p, w, h),evMan_(evman),display_(dm)
{
  
  mainFrame_= new TGCompositeFrame(this,200,300);
  //maxEvents_=evMan_->tree_->GetEntries();
  
  createCmdFrame();
  
   
  AddFrame(mainFrame_, new TGLayoutHints(kLHintsLeft | kLHintsExpandY,2,0,2,2));
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
void DialogFrame::createCmdFrame() 
{
 //create object selection buttons
  TGGroupFrame *gr1= new TGGroupFrame(mainFrame_,"Draw Selection",kVerticalFrame); 
  gr1->SetLayoutManager(new TGMatrixLayout(gr1,6,3,5));
  
  selectObject[0] = new TGCheckButton(gr1,"Hits");
  selectObject[0]->SetState(display_->drawHits_ ? kButtonDown :kButtonUp);
  selectObject[0]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=0)");
  selectObject[1] = new TGCheckButton(gr1,"Clusters");
  selectObject[1]->SetState(display_->drawClus_ ? kButtonDown :kButtonUp);
  selectObject[1]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=1)");
  selectObject[2] = new TGCheckButton(gr1,"Tracks");
  selectObject[2]->SetState(display_->drawTracks_ ? kButtonDown :kButtonUp);
  selectObject[2]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=2)");
  selectObject[3] = new TGCheckButton(gr1,"Particles");
  selectObject[3]->SetState(display_->drawParticles_ ? kButtonDown :kButtonUp);
  selectObject[3]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=3)");
  selectObject[4] = new TGCheckButton(gr1,"ClusterLines");
  selectObject[4]->SetState(display_->drawClusterL_ ? kButtonDown :kButtonUp);
  selectObject[4]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=4)");

  // create threshold fields
  TGNumberFormat::ELimit lim = TGNumberFormat::kNELLimitMinMax;  
  for (int i=0;i<4;++i){
    thresholdS[i] = new TGHSlider(gr1,100,kSlider1,ENER+i);
    thresholdS[i]->Associate(this);
    thresholdS[i]->SetRange(0,5);
   
    threshEntry[i] = new TGNumberEntryField(gr1,EN+i,0);
    threshEntry[i]->Associate(this);
    threshEntry[i]->SetLimits(lim,0,5);
    threshEntry[i]->SetFormat((TGNumberFormat::EStyle)0);
  }
  thresholdS[0]->SetPosition((long) display_->hitEnMin_);
  thresholdS[1]->SetPosition((long) display_->clusEnMin_);
  thresholdS[2]->SetPosition((long) display_->trackPtMin_);
  thresholdS[3]->SetPosition((long) display_->particlePtMin_);
  
  int charw= threshEntry[0]->GetCharWidth("O");
  int size=charw*4;
  for (int i=0;i<4;++i) {
    threshEntry[i]->SetIntNumber(thresholdS[i]->GetPosition());
    threshEntry[i]->Resize(size,threshEntry[i]->GetDefaultHeight());
  }
  
  //
  TGLayoutHints *lo=new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 5);
  TGLabel *label=new TGLabel(gr1,"  ");
  gr1->AddFrame(label,lo);
  label=new TGLabel(gr1," En/Pt  Threshold");
  gr1->AddFrame(label,lo);
  label=new TGLabel(gr1," (Gev) ");  
  gr1->AddFrame(label,lo);
  
  for (int i=0;i<4;++i) {
    gr1->AddFrame(selectObject[i],lo);
    gr1->AddFrame(thresholdS[i],lo);
    gr1->AddFrame(threshEntry[i],lo);
  }
  gr1->AddFrame(selectObject[4],lo);   // no thresh for clusterLines
  mainFrame_->AddFrame(gr1,new TGLayoutHints(kLHintsCenterX,2,2,2,2));
  
  // Next/Pevious/exit buttons
  
  TGHorizontalFrame *h1 = new TGHorizontalFrame(mainFrame_,20,30);
  mainFrame_->AddFrame(h1,new TGLayoutHints(kLHintsCenterX,2,2,2,2));
  
  nextButton = new TGTextButton(h1,"Draw Next");
  nextButton->Connect("Clicked()","DialogFrame",this,"doNextEvent()");
  h1->AddFrame(nextButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
  
  previousButton = new TGTextButton(h1,"Draw Previous");
  previousButton->Connect("Clicked()","DialogFrame",this,"doPreviousEvent()");
  h1->AddFrame(previousButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
  
  reProcessButton = new TGTextButton(h1,"Re-Process");
  reProcessButton->Connect("Clicked()","DialogFrame",this,"doReProcessEvent()");
  h1->AddFrame(reProcessButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
    
  exitButton = new TGTextButton(h1,"&Exit","gApplication->Terminate(0)");
  h1->AddFrame(exitButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
}  
  
//________________________________________________________________________________
void DialogFrame::CloseWindow()
{
 //!!!WARNING keep the first letter of the method uppercase.It is an overriden ROOT method  
 gApplication->Terminate(0);
}
//_________________________________________________________________________________
void DialogFrame::doModifyOptions(unsigned objNb)
{
 // hits and clusters are always drawn !
 //int eventNb = evMan_->iEvent_;
   //case 0: selectObject[0]->SetState(kButtonDown); break;
   //case 1: selectObject[1]->SetState(kButtonDown); break;
 switch (objNb) {
   case 0:
     display_->drawHits_ = (selectObject[0]->IsDown()) ?true :false;
     break;
   case 1:
     display_->drawClus_ = (selectObject[1]->IsDown()) ?true :false;
     break; 
   case 2:
     display_->drawTracks_ = (selectObject[2]->IsDown()) ?true :false;
     break;
   case 3: 
     display_->drawParticles_ = (selectObject[3]->IsDown()) ?true :false;
     break;
   case 4:
     display_->drawClusterL_ = (selectObject[4]->IsDown()) ?true :false;
     break;
 }
 display_->displayAll();    
}
//_______________________________________________________________________________
DialogFrame::~DialogFrame()
{
 mainFrame_->Cleanup();
}
//________________________________________________________________________________
void DialogFrame::doModifyPtThreshold(unsigned objNb,long pt)
{
 switch(objNb) {
   case 0: 
     display_->hitEnMin_=(double)pt;break;
     break;
   case 1:
     display_->clusEnMin_=(double)pt;break;
     break;
   case 2:
     display_->trackPtMin_=(double)pt;break;
     break;
   case 3:
     display_->particlePtMin_=(double)pt;break;
   default:break;
 }  
 display_->displayAll();
}
//_________________________________________________________________________________
void DialogFrame::doNextEvent()
{
 display_->displayNext();
 int eventNb = evMan_->getEventIndex();
 //TODOLIST:display new value of eventNb in the futur reserve field
} 
//_________________________________________________________________________________
void DialogFrame::doPreviousEvent()
{
  display_->displayPrevious();
  int eventNb = evMan_->getEventIndex();
  //TODOLIST:display new value of eventNb in the futur reserve field
}
//__________________________________________________________________________________
void DialogFrame::doReProcessEvent()
{
// TODOLIST:evMan_->connect() + nouveau nom de fichier s'il y a lieu ??
 int eventNb = evMan_->getEventIndex();
 display_->display(eventNb);
}


//________________________________________________________________________________
void DialogFrame::updateDisplay()
{
  display_->updateDisplay();
}

//________________________________________________________________________________
void DialogFrame::unZoom()
{
  display_->unZoom();
}
//_________________________________________________________________________________
Bool_t DialogFrame::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{ 
 switch (GET_MSG(msg)) {
   case kC_TEXTENTRY:
     switch (GET_SUBMSG(msg)) {
       case kTE_ENTER:
         switch (parm1) {
	   case EN :case EN+1: case EN+2: case EN+3:
	      {
	       //int eventNb=evMan_->iEvent_;
	       long val=threshEntry[parm1-EN]->GetIntNumber();
	       thresholdS[parm1-EN]->SetPosition(val);
	       doModifyPtThreshold(parm1-EN,val);
               break;
	      }
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
	   case ENER: case ENER+1: case ENER+2: case ENER+3:
	     {
	      unsigned index=parm1-ENER;
	      threshEntry[index]->SetIntNumber(parm2);
	      fClient->NeedRedraw(threshEntry[index]);
	      break;
	     } 
	   default:break;
	 }
	 break;  
       case kSL_RELEASE:
         switch (parm1) {
	   case ENER: case ENER+1: case ENER+2: case ENER+3:
	     {
	      long val = thresholdS[parm1-ENER]->GetPosition();
	      doModifyPtThreshold(parm1-ENER,val);
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
