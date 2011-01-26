#ifndef Fireworks_Core_FWHLTTriggerTableView_h
#define Fireworks_Core_FWHLTTriggerTableView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWHLTTriggerTableView
// 
/**\class FWHLTTriggerTableView FWHLTTriggerTableView.h Fireworks/Core/interface/FWHLTTriggerTableView.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Tue Jan 25 16:02:24 CET 2011
// $Id$
//

#include "Fireworks/Core/interface/FWTriggerTableView.h"
#include "Fireworks/Core/interface/FWStringParameter.h"

class FWTriggerTableViewManager;
class FWTriggerTableViewTableManager;
class FWHLTTriggerTableView : public FWTriggerTableView
{
public:
   FWHLTTriggerTableView( TEveWindowSlot*);
   virtual ~FWHLTTriggerTableView() {}

protected:
   virtual void fillTable(fwlite::Event* event);

private:
   fwlite::Event*                  m_event;
   acceptmap_t                     m_averageAccept;
   FWStringParameter               m_regex;
   FWStringParameter               m_process;

   void fillAverageAcceptFractions();
};
#endif
