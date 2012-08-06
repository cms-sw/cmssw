#ifndef GPF_BaseUpgrade_h
#define GPF_BaseUpgrade_h

/*! \file interface/GPFBase.h
  Base class for graphic representation 
  of objects 
*/ 
#include <TAttLine.h>
#include <TAttMarker.h>
 

class DisplayManagerUpgrade;

class GPFBaseUpgrade {

 protected:
  DisplayManagerUpgrade  *display_;
  int              viewId_;
  int              origId_;
  TAttMarker      *markerAttr_;
  TAttLine        *lineAttr_;
  int              color_;
  
 public:
  GPFBaseUpgrade(DisplayManagerUpgrade *display,int viewType,int ident,TAttMarker *attm,TAttLine *attl);
  GPFBaseUpgrade(DisplayManagerUpgrade *display,int viewType,int ident,TAttMarker *attm);
  GPFBaseUpgrade(DisplayManagerUpgrade *display,int viewType,int ident, int color);
  
  virtual ~GPFBaseUpgrade() {;}
  int getView()                 { return viewId_;}
  int getOrigin()               { return origId_;}

  virtual void draw() {;}
  virtual void setColor() = 0;
  virtual void setColor(int newcol ) = 0; 
  virtual void setInitialColor() = 0;
  virtual void setNewStyle() = 0;
  virtual void setNewSize() = 0;
  virtual double getEnergy()  { return -1;}
  virtual double getPt()      { return -1;} 
};
#endif
  
