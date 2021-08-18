#ifndef Fireworks_Core_FWDetailViewCanvas_h
#define Fireworks_Core_FWDetailViewCanvas_h

class TCanvas;
class TGCompositeFrame;
class TEveWindowSlot;

#include "Fireworks/Core/interface/FWDetailView.h"

template <typename T>
class FWDetailViewCanvas : public FWDetailView<T> {
public:
  FWDetailViewCanvas();
  ~FWDetailViewCanvas() override;

  void init(TEveWindowSlot*) override;

protected:
  TCanvas* m_infoCanvas;
  TGCompositeFrame* m_guiFrame;
  TCanvas* m_viewCanvas;
};

#include "Fireworks/Core/interface/FWDetailViewCanvas.icc"
#endif
