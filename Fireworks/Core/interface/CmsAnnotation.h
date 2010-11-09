#ifndef Fireworks_Core_CmsAnnotation_h
#define Fireworks_Core_CmsAnnotation_h

#include "TGLOverlay.h"

class TGLViewer;
class FWConfiguration;

class CmsAnnotation : public TGLOverlayElement {
private:
   enum EDrag { kMove, kResize, kNone};

public:
   CmsAnnotation(TGLViewerBase *parent, Float_t posx, Float_t posy);
   virtual ~CmsAnnotation();

   // ---------- member, functions -------------------------
   
   //configuration management interface
   virtual void addTo(FWConfiguration&) const;
   virtual void setFrom(const FWConfiguration&);
   
   virtual void   Render(TGLRnrCtx& rnrCtx);

   virtual Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec,
                         Event_t* event);
   virtual void   MouseLeave();

   Float_t getSize() const { return fSize; }
   void    setSize(Float_t x) { fSize = x; }
   
   bool    getVisible() const;
   void    setVisible(bool x);
   
   bool    getAllowDestroy() const { return fAllowDestroy; }
   void    setAllowDestroy(bool x) { fAllowDestroy = x; }

private:
   CmsAnnotation(const CmsAnnotation&); // stop default
   const CmsAnnotation& operator=(const CmsAnnotation&); // stop default
   
   Float_t           fPosX;           // x position [0, 1]
   Float_t           fPosY;           // y position [0, 1]

   Int_t             fMouseX, fMouseY; //! last mouse position
   EDrag             fDrag;

   TGLViewer        *fParent;
   
   Float_t           fSize;   //! relative size to viewport width
   Float_t           fSizeDrag;

   bool              fActive;
   bool              fAllowDestroy;
};

#endif
