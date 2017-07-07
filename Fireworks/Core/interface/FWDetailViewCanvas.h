#ifndef Fireworks_Core_FWDetailViewCanvas_h
#define Fireworks_Core_FWDetailViewCanvas_h

class TCanvas;
class TGCompositeFrame;

#include "Fireworks/Core/interface/FWDetailView.h"

template <typename T> class FWDetailViewCanvas : public FWDetailView<T> {
public:
  FWDetailViewCanvas ();
  virtual ~FWDetailViewCanvas();
  
  virtual void init(TEveWindowSlot*);
  
protected:
  TCanvas*          m_infoCanvas;
  TGCompositeFrame* m_guiFrame;
  TCanvas*          m_viewCanvas;
};

#include "Fireworks/Core/src/FWDetailViewCanvas.icc"
#endif

