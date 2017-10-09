#ifndef Fireworks_Core_FWFWDetailViewGL_h
#define Fireworks_Core_FWFWDetailViewGL_h

#include "TEveViewer.h"
#include "Fireworks/Core/interface/FWDetailView.h"


class TCanvas;
class TGCompositeFrame;
class TEveViewer;
class TEveScene;
class TEveWindowSlot;

template <typename T> class FWDetailViewGL : public FWDetailView<T> {
public:
   FWDetailViewGL ();
   virtual ~FWDetailViewGL();
  
  virtual void init(TEveWindowSlot*);
  TGLViewer* viewerGL() const { return m_eveViewer->GetGLViewer();}

   virtual void setBackgroundColor(Color_t);
  
protected:
  TCanvas          *m_infoCanvas;
  TGCompositeFrame *m_guiFrame;
  
  TEveViewer       *m_eveViewer;
  TEveScene        *m_eveScene;
};

#include "Fireworks/Core/src/FWDetailViewGL.icc"

#endif

