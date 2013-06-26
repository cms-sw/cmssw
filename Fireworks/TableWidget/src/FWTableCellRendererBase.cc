// -*- C++ -*-
//
// Package:     TableWidget
// Class  :     FWTableCellRendererBase
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Feb  2 16:40:26 EST 2009
// $Id: FWTableCellRendererBase.cc,v 1.1 2009/02/03 20:33:04 chrjones Exp $
//

// system include files

// user include files
#include "Fireworks/TableWidget/interface/FWTableCellRendererBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWTableCellRendererBase::FWTableCellRendererBase()
{
}

// FWTableCellRendererBase::FWTableCellRendererBase(const FWTableCellRendererBase& rhs)
// {
//    // do actual copying here;
// }

FWTableCellRendererBase::~FWTableCellRendererBase()
{
}

//
// assignment operators
//
// const FWTableCellRendererBase& FWTableCellRendererBase::operator=(const FWTableCellRendererBase& rhs)
// {
//   //An exception safe implementation is
//   FWTableCellRendererBase temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWTableCellRendererBase::buttonEvent(Event_t* /*iClickEvent*/, int /*iRelClickX*/, int /*iRelClickY*/)
{}

//
// const member functions
//

//
// static member functions
//
