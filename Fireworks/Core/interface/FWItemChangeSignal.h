#ifndef Fireworks_Core_FWItemChangeSignal_h
#define Fireworks_Core_FWItemChangeSignal_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWItemChangeSignal
//
/**\class FWItemChangeSignal FWItemChangeSignal.h Fireworks/Core/interface/FWItemChangeSignal.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Jan 21 19:30:17 EST 2008
// $Id: FWItemChangeSignal.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <set>
#include <vector>
#include "sigc++/signal.h"

// user include files

// forward declarations
class FWEventItem;
typedef sigc::signal<void,const FWEventItem* > FWItemChangeSignal;
#endif
