#ifndef Fireworks_Core_FWDetailViewBase_h
#define Fireworks_Core_FWDetailViewBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewBase
//
/**\class FWDetailViewBase FWDetailViewBase.h Fireworks/Core/interface/FWDetailViewBase.h

   Description: Base class for detailed views

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Jan  9 13:35:52 EST 2009
// $Id: FWDetailViewBase.h,v 1.6 2009/06/22 14:32:25 amraktad Exp
// system include files

#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

#include "Rtypes.h"

class TEveWindow;
class TEveWindowSlot;
class FWModelId;
class FWEventItem;

namespace fireworks {
  class Context;
}

class FWDetailViewBase {
public:
  virtual ~FWDetailViewBase();

  void build(const FWModelId&);

  virtual void init(TEveWindowSlot*) = 0;
  virtual void setBackgroundColor(Color_t col) {}

  //canvas utilities
  static void drawCanvasDot(Float_t x, Float_t y, Float_t r, Color_t);
  static void drawCanvasBox(Double_t* pos, Color_t fillCol, Int_t fillType = 0, bool bg = kTRUE);

  const FWEventItem* item() { return m_item; }
  void setItem(const FWEventItem* x) { m_item = x; }
  const fireworks::Context& context() const;

protected:
  FWDetailViewBase(const std::type_info&);

private:
  FWDetailViewBase(const FWDetailViewBase&) = delete;                   // stop default
  const FWDetailViewBase& operator=(const FWDetailViewBase&) = delete;  // stop default

  virtual void build(const FWModelId&, const void*) = 0;

  const FWEventItem* m_item;
  FWSimpleProxyHelper m_helper;
};

#endif
