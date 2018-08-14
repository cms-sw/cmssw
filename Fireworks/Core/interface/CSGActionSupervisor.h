#ifndef Fireworks_Core_CSGActionSupervisor_h
#define Fireworks_Core_CSGActionSupervisor_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGActionSupervisor
//
/**\class CSGActionSupervisor CSGActionSupervisor.h Fireworks/Core/interface/CSGActionSupervisor.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  
//         Created: Aug 2009

#include <vector>
#include "Rtypes.h"

class CSGAction;
class TGPopupMenu;
struct Event_t;

class CSGActionSupervisor {

public:
   CSGActionSupervisor();
   virtual ~CSGActionSupervisor();

   const std::vector<CSGAction*>& getListOfActions() const;
   void addToActionMap(CSGAction *action);

   virtual void defaultAction();

   CSGAction* getAction(const std::string& name);

   virtual void enableActions(bool enable = true);

   Bool_t activateMenuEntry(int entry);
   Bool_t activateToolBarEntry(int entry);
   void resizeMenu(TGPopupMenu *menu);
   virtual void HandleMenu(Int_t id);

   Long_t getToolTipDelay() const;

protected:
   std::vector<CSGAction*> m_actionList;

private:
   CSGActionSupervisor(const CSGActionSupervisor&) = delete; // stop default
   const CSGActionSupervisor& operator=(const CSGActionSupervisor&) = delete; // stop default

   // ---------- member data --------------------------------

   Long_t m_tooltipDelay;

};

#endif
