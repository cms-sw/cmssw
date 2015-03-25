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
#include "Fireworks/Core/interface/FWGeometry.h"
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
    FW3DViewBase(slot, typeId, version),
    m_ecalBarrel(0),
    m_showEcalBarrel(this, "Show Ecal Barrel", true )
{
    m_ecalBarrel = new TEveBoxSet("ecalBarrel"); 
    m_ecalBarrel->UseSingleColor();
    m_ecalBarrel->SetMainColor(kAzure+10);
    m_ecalBarrel->SetMainTransparency(98);
    geoScene()->AddElement(m_ecalBarrel);
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
    m_showEcalBarrel.changed_.connect(boost::bind(&FWISpyView::showEcalBarrel, this,_1));

    showEcalBarrel(m_showEcalBarrel.value());
   
}

void  FWISpyView::showEcalBarrel(bool x) {
    if (x &&  m_ecalBarrel->GetPlex()->Size() == 0) {
        const FWGeometry* geom = context().getGeom();
        std::vector<unsigned int> ids = geom->getMatchedIds(FWGeometry::Detector::Ecal, FWGeometry::SubDetector::PixelBarrel);
        m_ecalBarrel->Reset(TEveBoxSet::kBT_FreeBox, true, ids.size() );
        for (std::vector<unsigned int>::iterator it = ids.begin(); it != ids.end(); ++it) {
            const float* cor = context().getGeom()->getCorners(*it);
            m_ecalBarrel->AddBox(cor);
        }
        m_ecalBarrel->RefitPlex();
    }

    if (m_ecalBarrel->GetRnrSelf() != x) {
        m_ecalBarrel->SetRnrSelf(x);
        gEve->Redraw3D();
    }
}


void 
FWISpyView::populateController(ViewerParameterGUI& gui) const
{
   FW3DViewBase::populateController(gui);

   gui.requestTab("Detector").separator().
       addParam(&m_showEcalBarrel);
}
