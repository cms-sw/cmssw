#ifndef GPF_Base_h
#define GPF_Base_h

/*! \file interface/GPFBase.h
     Base class for graphic representation 
     of objects 
*/ 
 
#include "RecoParticleFlow/PFRootEvent/interface/DisplayManager.h"


class GPFBase {

  public:
     DisplayManager  *display_;
    int viewId_;
    int origId_;
    
  public:
    GPFBase(DisplayManager *display,int viewType,int ident);
    virtual ~GPFBase() {;}
    int getView()                 { return viewId_;}
    int getOrigin()               { return origId_;}

    virtual void draw() {;}
    //virtual drawGPFBlocks(){;}
    virtual void setColor(int){;}
    virtual void setInitialColor() {;}
    virtual double getEnergy()  { return -1;}
    virtual double getPt()      {return -1;} 
};
#endif
  
