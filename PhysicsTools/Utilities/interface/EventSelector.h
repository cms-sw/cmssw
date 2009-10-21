#ifndef PhysicsTools_Utilities_interface_EventSelector_h
#define PhysicsTools_Utilities_interface_EventSelector_h

/**
  \class    EventSelector EventSelector.h "CommonTools/Utils/interface/EventSelector.h"
  \brief    A selector of events. 

  This is a placeholder. 

  \author Salvatore Rappoccio
  \version  $Id: EventSelector.h,v 1.2 2009/09/19 03:30:25 srappocc Exp $
*/


#include "PhysicsTools/Utilities/interface/Selector.h"
#include "DataFormats/Common/interface/EventBase.h"
#include <fstream>
#include <functional>


typedef Selector<edm::EventBase> EventSelector;

#endif
