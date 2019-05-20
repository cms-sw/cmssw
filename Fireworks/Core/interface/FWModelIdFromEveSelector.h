#ifndef Fireworks_Core_FWModelIdFromEveSelector_h
#define Fireworks_Core_FWModelIdFromEveSelector_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelIdFromEveSelector
//
/**\class FWModelIdFromEveSelector FWModelIdFromEveSelector.h Fireworks/Core/interface/FWModelIdFromEveSelector.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Oct 28 11:44:11 CET 2009
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWFromEveSelectorBase.h"
#include "Fireworks/Core/interface/FWModelId.h"

// forward declarations

class FWModelIdFromEveSelector : public FWFromEveSelectorBase {
public:
  FWModelIdFromEveSelector(const FWModelId& iId = FWModelId()) : m_id(iId) {}
  //virtual ~FWModelIdFromEveSelector();

  operator FWModelId() const { return m_id; }
  // ---------- const member functions ---------------------
  const FWModelId& id() const { return m_id; }

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void doSelect() override;
  void doUnselect() override;

private:
  //FWModelIdFromEveSelector(const FWModelIdFromEveSelector&); // stop default

  //const FWModelIdFromEveSelector& operator=(const FWModelIdFromEveSelector&); // stop default

  // ---------- member data --------------------------------
  FWModelId m_id;
};

#endif
