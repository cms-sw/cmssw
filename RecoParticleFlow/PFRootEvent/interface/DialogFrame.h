#ifndef PF_DialogFrame_h
#define PF_DialogFrame_h

#include <TGClient.h>
#include <TGButton.h>
#include <TGButtonGroup.h>
//#include <TGTextEntry.h>
#include <TGSlider.h>
//#include <TGTextBuffer.h> 
#include <TGNumberEntry.h>

#include <TGFrame.h>
#include <TGLabel.h>
#include <RQ_OBJECT.h>

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include <string>


class DialogFrame  : public TGMainFrame {
  RQ_OBJECT("DialogFrame")
  
  private:
  
    static const int EN=1;     // id threshold energy field
    static const int ENER=10;  // id threshold energy slider
    
    TGCompositeFrame    *mainFrame_;
    PFRootEventManager  *evMan_;
    TGCheckButton       *selectObject[5];
    TGHSlider           *thresholdS[5];
    TGNumberEntryField  *threshEntry[5];
    
    TGTextButton        *exitButton,*nextButton,*previousButton;
    int                  eventNb_;
    int                  maxEvents_;
    
   public:
     DialogFrame(PFRootEventManager *evman,const TGWindow *p,UInt_t w,UInt_t h);
     virtual ~DialogFrame(); 
     
     void doNextEvent();
     void doPreviousEvent();
     void doModifyOptions(unsigned obj);
     void doModifyPtThreshold(unsigned obj,long val);
     virtual bool ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
     virtual void CloseWindow();
};
#endif 
