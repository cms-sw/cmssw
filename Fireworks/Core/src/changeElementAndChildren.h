#ifndef Fireworks_Core_changeElementAndChildren_h
#define Fireworks_Core_changeElementAndChildren_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     changeElementAndChildren
//
/**\class changeElementAndChildren changeElementAndChildren.h Fireworks/Core/interface/changeElementAndChildren.h

 Description: function to change an element and all its children to match the ModelInfo

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Mar 11 21:41:13 CDT 2008
// $Id: changeElementAndChildren.h,v 1.1 2008/03/12 02:53:25 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWEventItem.h"

// forward declarations
class TEveElement;
void
changeElementAndChildren(TEveElement* iElement,
                         const FWEventItem::ModelInfo& iInfo);
#endif
