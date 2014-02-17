// -*- C++ -*-
//
// Package:     Core
// Class  :     FWViewEnergyScaleEditor
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel 
//         Created:  Fri Sep 24 18:52:19 CEST 2010
// $Id: FWViewEnergyScaleEditor.cc,v 1.3 2012/08/01 00:41:36 amraktad Exp $
//

// system include files

// user include files
#include "TGButton.h"
#include "TGLabel.h"
#include "Fireworks/Core/interface/FWViewEnergyScaleEditor.h"
#include "Fireworks/Core/interface/FWEveView.h"
#include "Fireworks/Core/interface/FWViewEnergyScale.h"
#include "Fireworks/Core/interface/FWParameterSetterBase.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWViewEnergyScaleEditor::FWViewEnergyScaleEditor(FWViewEnergyScale* s, TGCompositeFrame* w, bool addAutoScaleControll):
   TGVerticalFrame(w),
   m_scale(s),
   m_enabled(true)
{ 
   addParam(&m_scale->m_plotEt);
   addParam(&m_scale->m_scaleMode);
   addParam(&m_scale->m_fixedValToHeight, "FixedScaleMode");
   if (addAutoScaleControll)
      addParam(&m_scale->m_maxTowerHeight, "AutomaticScaleMode");
}


FWViewEnergyScaleEditor::~FWViewEnergyScaleEditor()
{
}


//
// member functions
//
void
FWViewEnergyScaleEditor::setEnabled(bool x)
{
   m_enabled =x;
   typedef  std::vector<boost::shared_ptr<FWParameterSetterBase> > sList;
   for (sList::iterator i = m_setters.begin(); i!=m_setters.end(); ++i)
   {
      (*i)->setEnabled(m_enabled);
   }
}

void
FWViewEnergyScaleEditor::addParam(FWParameterBase* param, const char* title)
{
   int leftPad = 0;
   if (title)
   {
      leftPad = 10;
      AddFrame(new TGLabel(this, title), new TGLayoutHints(kLHintsLeft, leftPad, 0, 0, 0));   
      leftPad *= 2;
   }
   
   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor(param) );
   ptr->attach((FWParameterBase*)param, this);
   TGFrame* pframe = ptr->build(this);
   AddFrame(pframe, new TGLayoutHints(kLHintsLeft, leftPad, 0, 0, 0));
   m_setters.push_back(ptr);
}
