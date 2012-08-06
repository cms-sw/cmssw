#ifndef PF_DialogFrameUpgrade_h
#define PF_DialogFrameUpgrade_h

#include <TGClient.h>
#include <TGButton.h>
#include <TGButtonGroup.h>
#include <TGDoubleSlider.h>
#include <TGNumberEntry.h>
#include <TGFrame.h>
#include <TGLabel.h>
#include <TButton.h>
#include <TCanvas.h>
#include <TMarker.h>
#include <TGraph.h>
#include <RQ_OBJECT.h>


#include <string>

class PFRootEventManagerUpgrade; 
class DisplayManagerUpgrade; 

class DialogFrameUpgrade  : public TGMainFrame {
  RQ_OBJECT("DialogFrameUpgrade")
  
    private:
  
  static const int EN=1;       // id threshold energy field
  static const int ENER=10;    // id threshold energy slider
  static const int MAXL=20;
  static const int PARTTYPE=30;
  static const int PFBLOCK=40;
   
    
  PFRootEventManagerUpgrade  *evMan_;
    
  DisplayManagerUpgrade      *display_;
  TGCompositeFrame    *mainFrame_;
  TGCompositeFrame    *cmdFrame_;
    

  TGCheckButton       *selectObject_[8];
  TGCheckButton       *printButton_[7];
  TGDoubleHSlider     *thresholdS_[6];
  TGNumberEntryField  *threshEntry_[6];
  TGNumberEntryField  *maxLineEntry_; 
  TGNumberEntryField  *particleTypeEntry_; 
  TGTextButton        *exitButton,*nextButton,*previousButton;
  TGTextButton        *reProcessButton;
  //int                  eventNr_;
  //int                  maxEvents_;
  //TButton             *Modify_;
  //TButton             *Cancel_;
  TCanvas             *attrView_;
  TMarker             *thisClusPattern_;
  TGraph              *trackPattern_;
  TGraph              *simplePartPattern_;   
    
  
 public:
  DialogFrameUpgrade(PFRootEventManagerUpgrade *evman, DisplayManagerUpgrade *dm,const TGWindow *p,UInt_t w,UInt_t h);
  virtual ~DialogFrameUpgrade(); 
  
  void closeCanvasAttr();   
  void createCmdFrame();
  void createCanvasAttr();
  void doLookForGenParticle();
  void doNextEvent();
  void doPreviousEvent();
  void doModifyOptions(unsigned obj);
  void doModifyPtThreshold(unsigned obj,double val);
  void isPFBlockVisible();
  void areBremVisible();
  void doPrint();
  void doPrintGenParticles();
  void doReProcessEvent();
  void selectPrintOption(int opt);
  //void modifyGraphicAttributes();
  void modifyAttr();
  void updateDisplay();
  void unZoom();

  virtual bool ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
  virtual void CloseWindow();
     
  ClassDef(DialogFrameUpgrade,1)
};
#endif 
