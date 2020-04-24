// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDoubleParameterSetter
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
#include "Fireworks/Core/src/FWDoubleParameterSetter.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWDoubleParameterSetter::FWDoubleParameterSetter() :
   m_param(nullptr),
   m_widget(nullptr)
{
}

// FWDoubleParameterSetter::FWDoubleParameterSetter(const FWDoubleParameterSetter& rhs)
// {
//    // do actual copying here;
// }

FWDoubleParameterSetter::~FWDoubleParameterSetter()
{
}

//
// assignment operators
//
// const FWDoubleParameterSetter& FWDoubleParameterSetter::operator=(const FWDoubleParameterSetter& rhs)
// {
//   //An exception safe implementation is
//   FWDoubleParameterSetter temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void
FWDoubleParameterSetter::attach(FWParameterBase* iParam)
{
   m_param = dynamic_cast<FWDoubleParameter*>(iParam);
   assert(nullptr!=m_param);
}

TGFrame*
FWDoubleParameterSetter::build(TGFrame* iParent, bool)
{
   TGCompositeFrame* frame = new TGHorizontalFrame(iParent);

   // number entry widget
   TGNumberFormat::ELimit limits = m_param->min()==m_param->max() ?
                                   TGNumberFormat::kNELNoLimits :
                                   ( m_param->min() > m_param->max() ? TGNumberFormat::kNELLimitMin : TGNumberFormat::kNELLimitMinMax);
   double min = 0;
   double max = 1;
   if(m_param->min()!=m_param->max()) {
      min=m_param->min();
      max=m_param->max();
   }
   m_widget = new TGNumberEntry
           (frame, m_param->value(),
           5,                         // number of digits
           0,                         // widget ID
           TGNumberFormat::kNESReal,  // style
           TGNumberFormat::kNEAAnyNumber, // input value filter
           limits,                    // specify limits
           min,                       // min value
           max);                      // max value

   frame->AddFrame(m_widget, new TGLayoutHints(kLHintsLeft|kLHintsCenterY, 2,8,2,2));
   m_widget->Connect("ValueSet(Long_t)", "FWDoubleParameterSetter", this, "doUpdate(Long_t)");

   // label
   frame->AddFrame(new TGLabel(frame,m_param->name().c_str()),
                   new TGLayoutHints(kLHintsLeft|kLHintsCenterY) );
   return frame;
}

void
FWDoubleParameterSetter::doUpdate(Long_t)
{
   //std::cout <<"doUpdate called"<<std::endl;
   assert(nullptr!=m_param);
   assert(nullptr!=m_widget);
   //std::cout <<m_widget->GetNumberEntry()->GetNumber()<<std::endl;
   m_param->set(m_widget->GetNumberEntry()->GetNumber());
   update();
}

void
FWDoubleParameterSetter::setEnabled(bool x)
{
   m_widget->SetState(x);
}
//
// const member functions
//

//
// static member functions
//
