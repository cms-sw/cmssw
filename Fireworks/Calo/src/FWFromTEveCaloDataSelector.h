#ifndef Fireworks_Calo_FWFromTEveCaloDataSelector_h
#define Fireworks_Calo_FWFromTEveCaloDataSelector_h
// -*- C++ -*-
//
// Package:     Calo
// Class  :     FWFromTEveCaloDataSelector
//
/**\class FWFromTEveCaloDataSelector FWFromTEveCaloDataSelector.h Fireworks/Calo/interface/FWFromTEveCaloDataSelector.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Oct 23 14:44:32 CDT 2009
//

// system include files
#include "TEveCaloData.h"

// user include files
#include "Fireworks/Core/interface/FWFromEveSelectorBase.h"
#include "Fireworks/Calo/src/FWFromSliceSelector.h"

// forward declarations
class FWEventItem;
class FWModelChangeManager;

//==============================================================================

class FWFromTEveCaloDataSelector : public FWFromEveSelectorBase {
public:
  FWFromTEveCaloDataSelector(TEveCaloData*);
  ~FWFromTEveCaloDataSelector() override;

  // ---------- const member functions ---------------------

  // ---------- static member functions --------------------

  // ---------- member functions ---------------------------
  void doSelect() override;
  void doUnselect() override;

  void addSliceSelector(int iSlice, FWFromSliceSelector*);
  void resetSliceSelector(int iSlice);

private:
  FWFromTEveCaloDataSelector(const FWFromTEveCaloDataSelector&) = delete;  // stop default

  const FWFromTEveCaloDataSelector& operator=(const FWFromTEveCaloDataSelector&) = delete;  // stop default

  // ---------- member data --------------------------------
  std::vector<FWFromSliceSelector*> m_sliceSelectors;
  TEveCaloData* m_data;  // cached
  FWModelChangeManager* m_changeManager;
};

#endif
