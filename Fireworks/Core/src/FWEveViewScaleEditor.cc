// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveViewScaleEditor
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel 
//         Created:  Fri Sep 24 18:52:19 CEST 2010
// $Id: FWEveViewScaleEditor.cc,v 1.3 2010/09/27 10:46:11 amraktad Exp $
//

// system include files

// user include files
#include "TGButton.h"
#include "TGLabel.h"
#include "Fireworks/Core/interface/FWEveViewScaleEditor.h"
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
FWEveViewScaleEditor::FWEveViewScaleEditor(TGCompositeFrame* w, FWViewEnergyScale* s):
   TGVerticalFrame(w),
   m_scale(s)
{ 
   m_globalScalesBtn = new TGCheckButton(this,"UseGlobalScales");
   AddFrame(m_globalScalesBtn, new TGLayoutHints(kLHintsLeft, 2, 0, 0, 0));
   m_globalScalesBtn->SetState(m_scale->m_useGlobalScales.value() ? kButtonDown : kButtonUp);
   m_globalScalesBtn->Connect("Clicked()","FWEveViewScaleEditor",this,"useGlobalScales()");
   
   addParam(&m_scale->m_plotEt);
   addParam(&m_scale->m_scaleMode);
   addParam(&m_scale->m_fixedValToHeight, "FixedMode");
   
   if (FWViewType::isLego(m_scale->getView()->typeId()) == false)
      addParam(&m_scale->m_maxTowerHeight, "AutomaticMode");
   
   typedef  std::vector<boost::shared_ptr<FWParameterSetterBase> > sList;
   for (sList::iterator i = m_setters.begin(); i!=m_setters.end(); ++i)
   {
      (*i)->setEnabled(!m_globalScalesBtn->IsOn());
   }
   
}


FWEveViewScaleEditor::~FWEveViewScaleEditor()
{
}

//
// member functions
//


void
FWEveViewScaleEditor::useGlobalScales()
{
   typedef  std::vector<boost::shared_ptr<FWParameterSetterBase> > sList;
   for (sList::iterator i = m_setters.begin(); i!=m_setters.end(); ++i)
   {
      (*i)->setEnabled(!m_globalScalesBtn->IsOn());
   }
   
   m_scale->m_useGlobalScales.set(m_globalScalesBtn->IsOn());
}

void
FWEveViewScaleEditor::addParam(const FWParameterBase* param, const char* title)
{
   int leftPad = 0;
   if (title)
   {
      leftPad = 10;
      AddFrame(new TGLabel(this, title), new TGLayoutHints(kLHintsLeft, leftPad, 0, 0, 0));   
      leftPad *= 2;
   }
   
   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor((FWParameterBase*)param) );
   ptr->attach((FWParameterBase*)param, this);
   TGFrame* pframe = ptr->build(this);
   ptr->setEnabled(!m_globalScalesBtn->IsOn());
   AddFrame(pframe, new TGLayoutHints(kLHintsLeft, leftPad, 0, 0, 0));
   m_setters.push_back(ptr);
}
