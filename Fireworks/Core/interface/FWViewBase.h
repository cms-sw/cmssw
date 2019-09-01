#ifndef Fireworks_Core_FWViewBase_h
#define Fireworks_Core_FWViewBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewBase
//
/**\class FWViewBase FWViewBase.h Fireworks/Core/interface/FWViewBase.h

   Description: Base class for all View instances

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 14:43:25 EST 2008
//

// system include files
#include <string>
#include <sigc++/signal.h>

// user include files
#include "Fireworks/Core/interface/FWConfigurableParameterizable.h"
#include "Fireworks/Core/interface/FWViewType.h"

#include "Rtypes.h"

// forward declarations
class TGFrame;
class FWViewContextMenuHandlerBase;
class ViewerParameterGUI;

class FWViewBase : public FWConfigurableParameterizable {
public:
  FWViewBase(FWViewType::EType, unsigned int iVersion = 1);

  // ---------- const member functions ---------------------
  const std::string& typeName() const;
  FWViewType::EType typeId() const { return m_type.id(); }

  virtual void saveImageTo(const std::string& iName) const = 0;
  void promptForSaveImageTo(TGFrame*) const;

  virtual FWViewContextMenuHandlerBase* contextMenuHandler() const;

  virtual void populateController(ViewerParameterGUI&) const {}

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void destroy();

  sigc::signal<void, const FWViewBase*> beingDestroyed_;
  sigc::signal<void, Int_t, Int_t> openSelectedModelContextMenu_;

protected:
  ~FWViewBase() override;
  FWViewType m_type;

private:
  FWViewBase(const FWViewBase&) = delete;  // stop default

  const FWViewBase& operator=(const FWViewBase&) = delete;  // stop default

  // ---------- member data --------------------------------
};

#endif
