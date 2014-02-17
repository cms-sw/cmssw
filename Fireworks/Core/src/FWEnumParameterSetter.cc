// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEnumParameterSetter
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  matevz
//         Created:  Fri Apr 30 15:17:33 CEST 2010
// $Id: FWEnumParameterSetter.cc,v 1.7 2012/09/21 09:26:26 eulisse Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWEnumParameterSetter.h"
#include "TGComboBox.h"
#include "TGLabel.h"
#include <cassert>

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWEnumParameterSetter::FWEnumParameterSetter() :
   m_param(0),
   m_widget(0)
{}

// FWEnumParameterSetter::FWEnumParameterSetter(const FWEnumParameterSetter& rhs)
// {
//    // do actual copying here;
// }

FWEnumParameterSetter::~FWEnumParameterSetter()
{}

//
// assignment operators
//
// const FWEnumParameterSetter& FWEnumParameterSetter::operator=(const FWEnumParameterSetter& rhs)
// {
//   //An exception safe implementation is
//   FWEnumParameterSetter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
FWEnumParameterSetter::attach(FWParameterBase* iParam)
{
   m_param = dynamic_cast<FWEnumParameter*>(iParam);
   assert(0!=m_param);
}

TGFrame*
FWEnumParameterSetter::build(TGFrame* iParent, bool labelBack)
{
   TGCompositeFrame *frame = new TGHorizontalFrame(iParent);

   m_widget = new TGComboBox(frame);
   std::map<Long_t, std::string>::const_iterator me = m_param->entryMap().begin();
   UInt_t max_len = 0;
   while (me != m_param->entryMap().end())
   {
      m_widget->AddEntry(me->second.c_str(), static_cast<Int_t>(me->first));
      if (me->second.length() > max_len) max_len = me->second.length();
      ++me;
   }
   m_widget->Resize(8*max_len + 20, 20);
   m_widget->Select(static_cast<Int_t>(m_param->value()), kFALSE);

   m_widget->Connect("Selected(Int_t)", "FWEnumParameterSetter", this, "doUpdate(Int_t)");

   // label
   TGLabel* label = new TGLabel(frame, m_param->name().c_str());
   if (labelBack)
   {
      frame->AddFrame(m_widget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,6,2,2));
      frame->AddFrame(label, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2, 4, 0, 0));
   }
   else
   {
      frame->AddFrame(label, new TGLayoutHints(kLHintsLeft|kLHintsCenterY) );
      frame->AddFrame(m_widget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,8,2,2));
   }
   return frame;
}

void
FWEnumParameterSetter::doUpdate(Int_t id)
{
   assert(0!=m_param);
   assert(0!=m_widget);
   m_param->set((Long_t) id);
   update();
}

void
FWEnumParameterSetter::setEnabled(bool x)
{
   m_widget->SetEnabled(x);
}

//
// const member functions
//

//
// static member functions
//
