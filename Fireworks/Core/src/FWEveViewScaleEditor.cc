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
// $Id$
//

// system include files

// user include files
#include "TGButton.h"
#include "Fireworks/Core/interface/FWEveViewScaleEditor.h"
#include "Fireworks/Core/interface/FWEveView.h"
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
FWEveViewScaleEditor::FWEveViewScaleEditor(TGCompositeFrame* w, FWEveView* v):
   TGVerticalFrame(w),
   m_view(v)
{ 
   m_globalScalesBtn = new TGCheckButton(this,"UseGlobalScales");
   AddFrame(m_globalScalesBtn);
   m_globalScalesBtn->Connect("Clicked()","FWEveViewScaleEditor",this,"useGlobalScales()");
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
   m_view->setUseGlobalEnergyScales(m_globalScalesBtn->IsOn());

   typedef  std::vector<boost::shared_ptr<FWParameterSetterBase> > sList;
   for (sList::iterator i = m_setters.begin(); i!=m_setters.end(); ++i)
   {
      (*i)->setEnabled(!m_globalScalesBtn->IsOn());
   }
}

void
FWEveViewScaleEditor::addParam(const FWParameterBase* param)
{
   boost::shared_ptr<FWParameterSetterBase> ptr( FWParameterSetterBase::makeSetterFor((FWParameterBase*)param) );
   ptr->attach((FWParameterBase*)param, this);
   TGFrame* pframe = ptr->build(this);
   ptr->setEnabled(!m_globalScalesBtn->IsOn());
   AddFrame(pframe, new TGLayoutHints(kLHintsExpandX));
   m_setters.push_back(ptr);
}

//
// const member functions
//

//
// static member functions
//
