// -*- C++ -*-
//
// Package:     cmsShow36
// Class  :     FWISpyView
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Alja Mrak-Tadel 
//         Created:  Wed Apr  7 14:40:31 CEST 2010
//

// system include files
#include <boost/bind.hpp>

// user include files
#include "TGLViewer.h"
#include "Fireworks/Core/interface/CmsShowViewPopup.h"
#include "Fireworks/Core/interface/FWISpyView.h"
#include "Fireworks/Core/interface/Context.h"
#include "TEveBoxSet.h"
#include "TEveScene.h"
#include "TEveManager.h"


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWISpyView::FWISpyView(TEveWindowSlot* slot, FWViewType::EType typeId, unsigned int version):
    FW3DViewBase(slot, typeId, version)
{
}


// FWISpyView::FWISpyView(const FWISpyView& rhs)
// {
//    // do actual copying here;
// }

FWISpyView::~FWISpyView()
{
}


void FWISpyView::setContext(const fireworks::Context& x)
{
    FW3DViewBase::setContext(x);
}



void 
FWISpyView::populateController(ViewerParameterGUI& gui) const
{
   FW3DViewBase::populateController(gui);
}
