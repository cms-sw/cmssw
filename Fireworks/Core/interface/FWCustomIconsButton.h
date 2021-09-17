#ifndef Fireworks_Core_FWCustomIconsButton_h
#define Fireworks_Core_FWCustomIconsButton_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWCustomIconsButton
//
/**\class FWCustomIconsButton FWCustomIconsButton.h Fireworks/Core/interface/FWCustomIconsButton.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Oct 23 13:05:30 EDT 2008
//

// system include files
#include "TGButton.h"

// user include files

// forward declarations
class TGPicture;

class FWCustomIconsButton : public TGButton {
public:
  FWCustomIconsButton(const TGWindow* iParent,
                      const TGPicture* iUpIcon,
                      const TGPicture* iDownIcon,
                      const TGPicture* iDisableIcon,
                      const TGPicture* iBelowMouseIcon = nullptr,
                      Int_t id = -1,
                      GContext_t norm = TGButton::GetDefaultGC()(),
                      UInt_t option = 0);

  ~FWCustomIconsButton() override;

  bool HandleCrossing(Event_t*) override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void swapIcons(const TGPicture*& iUpIcon, const TGPicture*& iDownIcon, const TGPicture*& iDisabledIcon);

  void setIcons(const TGPicture* iUpIcon,
                const TGPicture* iDownIcon,
                const TGPicture* iDisabledIcon,
                const TGPicture* ibelowMouseIcon = nullptr);

  const TGPicture* upIcon() const { return m_upIcon; }
  const TGPicture* downIcon() const { return m_downIcon; }
  const TGPicture* disabledIcon() const { return m_disabledIcon; }
  const TGPicture* bellowMouseIcon() const { return m_belowMouseIcon; }

  FWCustomIconsButton(const FWCustomIconsButton&) = delete;  // stop default

  const FWCustomIconsButton& operator=(const FWCustomIconsButton&) = delete;  // stop default

protected:
  void DoRedraw() override;

private:
  // ---------- member data --------------------------------
  const TGPicture* m_upIcon;
  const TGPicture* m_downIcon;
  const TGPicture* m_disabledIcon;
  const TGPicture* m_belowMouseIcon;

  bool m_inside;
};

#endif
