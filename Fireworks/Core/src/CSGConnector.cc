// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGConnector
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu May 29 20:58:15 CDT 2008
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/src/CSGConnector.h"
#include "Fireworks/Core/interface/CSGAction.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"

ClassImp(CSGConnector)

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
//CSGConnector::CSGConnector()
//{
//}

// CSGConnector::CSGConnector(const CSGConnector& rhs)
// {
//    // do actual copying here;
// }

//CSGConnector::~CSGConnector()
//{
//}

//
// assignment operators
//
// const CSGConnector& CSGConnector::operator=(const CSGConnector& rhs)
// {
//   //An exception safe implementation is
//   CSGConnector temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void CSGConnector::handleTextButton() {
   m_frame->activateTextButton(m_action->getTextButton());
}

void CSGConnector::handlePictureButton() {
   m_frame->activatePictureButton(m_action->getPictureButton());
}

void CSGConnector::handleMenu(Int_t entry) {
   m_frame->activateMenuEntry(entry);
}

void CSGConnector::handleToolBar(Int_t entry) {
   m_frame->activateToolBarEntry(entry);
}

//
// const member functions
//

//
// static member functions
//
