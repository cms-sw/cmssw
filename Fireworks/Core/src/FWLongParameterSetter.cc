// -*- C++ -*-
//
// Package:     Core
// Class  :     FWLongParameterSetter
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Mon Mar 10 11:22:32 CDT 2008
//

// system include files
#include "TGLabel.h"
#include "TGNumberEntry.h"

#include <cassert>
#include <iostream>

// user include files
#include "Fireworks/Core/src/FWLongParameterSetter.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWLongParameterSetter::FWLongParameterSetter() :
   m_param(nullptr),
   m_widget(nullptr)
{
}

// FWLongParameterSetter::FWLongParameterSetter(const FWLongParameterSetter& rhs)
// {
//    // do actual copying here;
// }

FWLongParameterSetter::~FWLongParameterSetter()
{
}

//
// assignment operators
//
// const FWLongParameterSetter& FWLongParameterSetter::operator=(const FWLongParameterSetter& rhs)
// {
//   //An exception safe implementation is
//   FWLongParameterSetter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
FWLongParameterSetter::attach(FWParameterBase* iParam)
{
   m_param = dynamic_cast<FWLongParameter*>(iParam);
   assert(nullptr!=m_param);
}

TGFrame*
FWLongParameterSetter::build(TGFrame* iParent, bool labelBack)
{
   TGCompositeFrame* frame = new TGHorizontalFrame(iParent);

   // number entry widget
   TGNumberFormat::ELimit limits = m_param->min()==m_param->max() ?
      TGNumberFormat::kNELNoLimits :
      ( m_param->min() > m_param->max() ? TGNumberFormat::kNELLimitMin : TGNumberFormat::kNELLimitMinMax);
   double min = 0;
   double max = 1;
   if (m_param->min()!=m_param->max())
   {
      min=m_param->min();
      max=m_param->max();
   }
   m_widget = new TGNumberEntry
      (frame, m_param->value(),
       5,                         // number of digits
       0,                         // widget ID
       TGNumberFormat::kNESInteger, // style
       TGNumberFormat::kNEAAnyNumber, // input value filter
       limits,                    // specify limits
       min,                       // min value
       max);                      // max value

   m_widget->Connect("ValueSet(Long_t)", "FWLongParameterSetter", this, "doUpdate(Long_t)");

   // label
   TGLabel* label = new TGLabel(frame, m_param->name().c_str());
   if (labelBack)
   {
      frame->AddFrame(m_widget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,6,2,2));
      frame->AddFrame(label,    new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,2,0,0));
   }
   else
   {
      frame->AddFrame(label,    new TGLayoutHints(kLHintsLeft|kLHintsCenterY));
      frame->AddFrame(m_widget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,8,2,2));
   }

   return frame;
}

void
FWLongParameterSetter::doUpdate(Long_t)
{
   //std::cout <<"doUpdate called"<<std::endl;

   // Idiotic TGNumberEntry arrow buttons can send several events and if
   // individual event processing takes longer it can happen that the widget
   // gets detroyed in the meantime. So, process all events from arrows as
   // soon as possible.
   static bool in_update = false;
   if (in_update)
      return;
   in_update = true;
   gClient->ProcessEventsFor((TGWindow*)gTQSender);
   in_update = false;
      
   assert(nullptr!=m_param);
   assert(nullptr!=m_widget);
   //std::cout <<m_widget->GetNumberEntry()->GetNumber()<<std::endl;
   m_param->set(m_widget->GetNumberEntry()->GetIntNumber());
   update();
}

//
// const member functions
//

//
// static member functions
//
