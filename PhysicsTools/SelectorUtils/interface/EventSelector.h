#ifndef PhysicsTools_Utilities_interface_EventSelector_h
#define PhysicsTools_Utilities_interface_EventSelector_h

/**
  \class    EventSelector EventSelector.h "CommonTools/Utils/interface/EventSelector.h"
  \brief    A selector of events. 

  This is a placeholder. 

  \author Salvatore Rappoccio
  \version  $Id: EventSelector.h,v 1.2 2010/04/30 14:30:07 srappocc Exp $
*/


#include "PhysicsTools/SelectorUtils/interface/Selector.h"
#include <fstream>
#include <functional>


typedef Selector<edm::EventBase> EventSelector;

#endif
