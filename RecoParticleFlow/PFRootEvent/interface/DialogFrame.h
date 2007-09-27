#ifndef PF_DialogFrame_h
#define PF_DialogFrame_h

#include <TGClient.h>
#include <TGButton.h>
#include <TGButtonGroup.h>
#include <TGSlider.h>
#include <TGNumberEntry.h>
#include <TGFrame.h>
#include <TGLabel.h>
#include <RQ_OBJECT.h>

#include "RecoParticleFlow/PFRootEvent/interface/PFRootEventManager.h"
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"

#include <string>


class DialogFrame  : public TGMainFrame {
  RQ_OBJECT("DialogFrame")
  
  private:
  
    static const int EN=1;     // id threshold energy field
    static const int ENER=10;  // id threshold energy slider
    
    PFRootEventManager  *evMan_;
    
    DisplayManager      *display_;
    TGCompositeFrame    *mainFrame_;
    TGCompositeFrame    *cmdFrame_;
    

    TGCheckButton       *selectObject[5];
    TGHSlider           *thresholdS[5];
    TGNumberEntryField  *threshEntry[5];
    
    TGTextButton        *exitButton,*nextButton,*previousButton;
    TGTextButton        *reProcessButton;
    //int                  eventNr_;
    //int                  maxEvents_;
    
    
  
   public:
     DialogFrame(PFRootEventManager *evman, DisplayManager *dm,const TGWindow *p,UInt_t w,UInt_t h);
     virtual ~DialogFrame(); 
     
     void doNextEvent();
     void doPreviousEvent();
     void doModifyOptions(unsigned obj);
     void doModifyPtThreshold(unsigned obj,long val);
     void doReProcessEvent();
     //void createCanvas();
     void createCmdFrame();
     //void displayAll();
     //void displayCanvas();
     //void getDisplayOptions();
     //void loadGraphicObjects();
     /// print event display 
     //void printDisplay( const char* directory="" ) const;
     void updateDisplay();
     void unZoom();

     virtual bool ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
     virtual void CloseWindow();
     
     //-------------------------------graphic options variable ---------------------
     //double trackPtMin_;
     //double hitPtMin_;
     //double particlePtMin_;
     //double clusPtMin_;
     
     //bool drawHits_;
     //bool drawTracks_;
     //bool drawClus_;
     //bool drawClusterL_;
    // bool drawParticles_;
};
#endif 
