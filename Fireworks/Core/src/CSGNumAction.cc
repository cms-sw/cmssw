// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGNumAction
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Thu Jun 26 12:49:09 EDT 2008
// $Id: CSGNumAction.cc,v 1.1 2008/06/29 13:18:33 chrjones Exp $
//

// system include files
#include <TGNumberEntry.h>
#include <TQObject.h>

// user include files
#include "Fireworks/Core/interface/CSGNumAction.h"
//#include "Fireworks/Core/src/CSGConnector.h"
#include "Fireworks/Core/interface/CmsShowMainFrame.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
CSGNumAction::CSGNumAction(CmsShowMainFrame *frame, const char *name) :
  m_frame(frame),
  m_name(name)
{
  m_enabled = kTRUE;
  m_numberEntry = 0;
  //  m_connector = new CSGConnector(this, m_frame);
}

// CSGNumAction::CSGNumAction(const CSGNumAction& rhs)
// {
//    // do actual copying here;
// }

CSGNumAction::~CSGNumAction()
{
}

//
// assignment operators
//
// const CSGNumAction& CSGNumAction::operator=(const CSGNumAction& rhs)
// {
//   //An exception safe implementation is
//   CSGNumAction temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void
CSGNumAction::createNumberEntry(TGCompositeFrame* p, TGLayoutHints* l)//, Int_t id, Double_t val, TGNumberFormat::EStyle style, TGNumberFormat::EAttribute attr, TGNumberFormat::ELimit limits, Double_t min, Double_t max)
{
  if (m_numberEntry != 0) {
    delete m_numberEntry;
  }
  m_numberEntry = new TGNumberEntryField(p);//id, val, style, attr, limits, min, max);
  p->AddFrame(m_numberEntry, l);
  TQObject::Connect(m_numberEntry, "ReturnPressed()", "CSGNumAction", this, "activate()");
}

void
CSGNumAction::enable() {
  if (m_numberEntry != 0) m_numberEntry->SetEnabled();
}

void
CSGNumAction::disable() {
  if (m_numberEntry != 0) m_numberEntry->SetEnabled(kFALSE);
}

void
CSGNumAction::activate() {
  activated.emit(m_numberEntry->GetNumber());
}

void
CSGNumAction::setNumber(Double_t num) {
  m_numberEntry->SetNumber(num);
}
//
// const member functions
//
const std::string&
CSGNumAction::getName() const {
  return m_name;
}

TGNumberEntryField*
CSGNumAction::getNumberEntry() const {
  return m_numberEntry;
}

Double_t
CSGNumAction::getNumber() const {
  return m_numberEntry->GetNumber();
}

Bool_t
CSGNumAction::isEnabled() const {
  return m_enabled;
}

//
// static member functions
//
