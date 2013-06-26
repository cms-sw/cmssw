#ifndef Fireworks_Core_FWModelChangeSignal_h
#define Fireworks_Core_FWModelChangeSignal_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWModelChangeSignal
//
/**\class FWModelChangeSignal FWModelChangeSignal.h Fireworks/Core/interface/FWModelChangeSignal.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Mon Jan 21 19:30:17 EST 2008
// $Id: FWModelChangeSignal.h,v 1.3 2009/01/23 21:35:41 amraktad Exp $
//

// system include files
#include <set>
#include <vector>
#include "sigc++/signal.h"

// user include files

// forward declarations
class FWModelId;
typedef std::set<FWModelId> FWModelIds;
typedef sigc::signal<void,const FWModelIds& > FWModelChangeSignal;
#endif
