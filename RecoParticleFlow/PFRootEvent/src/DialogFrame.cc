#include "RecoParticleFlow/PFRootEvent/interface/DialogFrame.h"
#include <TTree.h>
#include <TApplication.h>
#include <iostream>

DialogFrame::DialogFrame(PFRootEventManager *evman,const TGWindow *p,UInt_t w,UInt_t h)
                         :TGMainFrame(p, w, h)
{
  evMan_=evman;
  mainFrame_= new TGCompositeFrame(this,200,300);
  maxEvents_=evMan_->tree_->GetEntries();
  eventNb_=evMan_->iEvent_;
  
  //create object selection buttons
  TGGroupFrame *gr1= new TGGroupFrame(mainFrame_,"Draw Selection",kVerticalFrame); 
  gr1->SetLayoutManager(new TGMatrixLayout(gr1,6,3,5));
  
  selectObject[0] = new TGCheckButton(gr1,"Hits");
  selectObject[0]->SetState(kButtonDown);
  selectObject[0]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=0)");
  selectObject[1] = new TGCheckButton(gr1,"Clusters");
  selectObject[1]->SetState(kButtonDown);
  selectObject[1]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=1)");
  selectObject[2] = new TGCheckButton(gr1,"Tracks");
  selectObject[2]->SetState(evMan_->displayRecTracks_ ?kButtonDown :kButtonUp);
  selectObject[2]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=2)");
  selectObject[3] = new TGCheckButton(gr1,"Particles");
  selectObject[3]->SetState(evMan_->displayTrueParticles_ ?kButtonDown :kButtonUp);
  selectObject[3]->Connect("Clicked()","DialogFrame",this,"doModifyOptions(=3)");
  selectObject[4] = new TGCheckButton(gr1,"ClusterLines");
  selectObject[4]->SetState(evMan_->displayClusterLines_ ?kButtonDown :kButtonUp);
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
  thresholdS[0]->SetPosition((long)(evMan_->displayRecHitsPtMin_));
  thresholdS[1]->SetPosition((long)(evMan_->displayClustersPtMin_));
  thresholdS[2]->SetPosition((long)(evMan_->displayRecTracksPtMin_));
  thresholdS[3]->SetPosition((long)(evMan_->displayTrueParticlesPtMin_));
  
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
  label=new TGLabel(gr1,"Pt Threshold");
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
    
  exitButton = new TGTextButton(h1,"&Exit","gApplication->Terminate(0)");
  h1->AddFrame(exitButton,new TGLayoutHints(kLHintsBottom|kLHintsCenterX,2,2,2,2));
  
  // 
   
  AddFrame(mainFrame_, new TGLayoutHints(kLHintsLeft | kLHintsExpandY,2,0,2,2));
  // Set a name to the main frame
  SetWindowName("PFRootEvent");
  // Map all subwindows of main frame
  MapSubwindows();
  // Initialize the layout algorithm
  Resize(mainFrame_->GetDefaultSize());
  // Map main frame
  MapWindow();
  
  // display first event 
  evMan_->display(eventNb_);
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
 eventNb_=evMan_->iEvent_;
 switch (objNb) {
   case 0: selectObject[0]->SetState(kButtonDown); break;
   case 1: selectObject[1]->SetState(kButtonDown); break;
   case 2:
     evMan_->displayRecTracks_ = (selectObject[2]->IsDown()) ?true :false;
     evMan_->display(eventNb_);
     break;
   case 3: 
     evMan_->displayTrueParticles_ = (selectObject[3]->IsDown()) ?true :false;
     evMan_->display(eventNb_);
     //doDraw();
     break;
   case 4:
     evMan_->displayClusterLines_ = (selectObject[4]->IsDown()) ?true :false;
     evMan_->display(eventNb_);
     //doDraw();
     break;
 }    
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
   case 0: evMan_->displayRecHitsPtMin_=(double)pt;break;
   case 1: evMan_->displayClustersPtMin_=(double)pt;break;
   case 2: evMan_->displayRecTracksPtMin_=(double)pt;break;
   case 3: evMan_->displayTrueParticlesPtMin_=(double)pt;break;
   default:break;
 }  
 eventNb_=evMan_->iEvent_;
 evMan_->display(eventNb_);
}
//_________________________________________________________________________________
void DialogFrame::doNextEvent()
{
 eventNb_=evMan_->iEvent_;
 if (eventNb_<maxEvents_) {
    ++eventNb_;
    evMan_->display(eventNb_);
 }   
} 
//_________________________________________________________________________________
void DialogFrame::doPreviousEvent()
{
 eventNb_=evMan_->iEvent_;
 if (eventNb_>0) {
   --eventNb_;
   evMan_->display(eventNb_);
 }  
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
	       eventNb_=evMan_->iEvent_;
	       long val=threshEntry[parm1-EN]->GetIntNumber();
	       thresholdS[parm1-EN]->SetPosition(val);
	       doModifyPtThreshold(parm1-EN,val);
	       evMan_->display(eventNb_);
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
