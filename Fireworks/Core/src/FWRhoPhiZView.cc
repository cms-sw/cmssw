// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRhoPhiZView
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue Feb 19 10:33:25 EST 2008
// $Id: FWRhoPhiZView.cc,v 1.6 2008/03/19 15:19:11 chrjones Exp $
//

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"
#include "TGLEmbeddedViewer.h"
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TClass.h"
#include "TEveElement.h"
#include "TEveProjectionBases.h"
#include "TEvePolygonSetProjected.h"
#include "TEveProjections.h"

// user include files
#include "Fireworks/Core/interface/FWRhoPhiZView.h"


//
// constants, enums and typedefs
//
static TEveElement* doReplication(TEveProjectionManager* iMgr, TEveElement* iFrom, TEveElement* iParent) {
   static const TEveException eh("FWRhoPhiZView::doReplication ");
   TEveElement  *new_re = 0;
   TEveProjected   *new_pr = 0;
   TEveProjected *pble   = dynamic_cast<TEveProjected*>(iFrom);
   //std::cout << typeid(*iFrom).name() <<std::endl;
   if (pble)
   {
      new_re = (TEveElement*) TClass::GetClass( typeid(*iFrom) )->New();
      assert(0!=new_re);
      new_pr = dynamic_cast<TEveProjected*>(new_re);
      assert(0!=new_pr);
      new_pr->SetProjection(iMgr, pble->GetProjectable());
      new_pr->SetDepth(iMgr->GetCurrentDepth());

      new_re->SetMainTransparency(iFrom->GetMainTransparency());
      new_re->SetMainColor(iFrom->GetMainColor());
      if ( TEvePolygonSetProjected* poly = dynamic_cast<TEvePolygonSetProjected*>(new_re) )
         poly->SetLineColor(dynamic_cast<TEvePolygonSetProjected*>(iFrom)->GetLineColor());
      
   }
   else
   {
      new_re = new TEveElementList;
   }
   new_re->SetElementNameTitle(iFrom->GetElementName(),
                               iFrom->GetElementTitle());
   new_re->SetRnrSelf     (iFrom->GetRnrSelf());
   new_re->SetRnrChildren(iFrom->GetRnrChildren());
   iParent->AddElement(new_re);
   
   for (TEveElement::List_i i=iFrom->BeginChildren(); i!=iFrom->EndChildren(); ++i)
      doReplication(iMgr,*i, new_re);
   return new_re;
}

//
// static data member definitions
//

//
// constructors and destructor
//
FWRhoPhiZView::FWRhoPhiZView(TGFrame* iParent,const std::string& iName, const TEveProjection::EPType_e& iProjType) :
m_typeName(iName),
m_distortion(this,"distortion",5.,0.,20.)
{
   m_projMgr = new TEveProjectionManager;
   m_projMgr->SetProjection(iProjType);
   //m_projMgr->GetProjection()->SetFixedRadius(700);
   m_projMgr->GetProjection()->SetDistortion(m_distortion.value()*1e-3);
   m_projMgr->GetProjection()->SetFixR(200);
   m_projMgr->GetProjection()->SetFixZ(300);
   m_projMgr->GetProjection()->SetPastFixRFac(-0.3);
   m_projMgr->GetProjection()->SetPastFixZFac(-0.3);
   gEve->AddToListTree(m_projMgr,kTRUE);
   
   //m_distortion.changed_.connect(boost::bind(&TEveProjection::SetDistortion, m_projMgr->GetProjection(),
     //                                        boost::bind(toFloat,_1)));
   m_distortion.changed_.connect(boost::bind(&FWRhoPhiZView::doDistortion,this,_1));
   
   m_pad = new TEvePad;
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iParent, m_pad);
   m_embeddedViewer=ev;
   TEveViewer* nv = new TEveViewer(iName.c_str());
   nv->SetGLViewer(ev);
   nv->IncDenyDestroy();
   ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   TEveScene* ns = gEve->SpawnNewScene(iName.c_str());
   m_scene = ns;
   nv->AddScene(ns);
   m_viewer=nv;
   //this is needed so if a TEveElement changes this view will be informed
   gEve->AddElement(nv, gEve->GetViewers());
   
   gEve->AddElement(m_projMgr,ns);
   //ev->ResetCurrentCamera();
   
}

// FWRhoPhiZView::FWRhoPhiZView(const FWRhoPhiZView& rhs)
// {
//    // do actual copying here;
// }

FWRhoPhiZView::~FWRhoPhiZView()
{
}

//
// assignment operators
//
// const FWRhoPhiZView& FWRhoPhiZView::operator=(const FWRhoPhiZView& rhs)
// {
//   //An exception safe implementation is
//   FWRhoPhiZView temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWRhoPhiZView::doDistortion(double iAmount)
{
   //Following code used in TEveProjectionManagerEditor
   m_projMgr->GetProjection()->SetDistortion(iAmount*0.001);
   m_projMgr->UpdateName();
   m_projMgr->ProjectChildren();
}


void 
FWRhoPhiZView::resetCamera()
{
   //this is needed to get EVE to transfer the TEveElements to GL so the camera reset will work
   m_scene->Repaint();
   m_viewer->Redraw(kTRUE);
   //gEve->Redraw3D(kTRUE);

   m_embeddedViewer->ResetCurrentCamera();
}

void 
FWRhoPhiZView::destroyElements()
{
   m_projMgr->DestroyElements();
   std::for_each(m_geom.begin(),m_geom.end(),
                 boost::bind(&TEveProjectionManager::AddElement,
                             m_projMgr,
                             _1));
}
void 
FWRhoPhiZView::replicateGeomElement(TEveElement* iChild)
{
   m_geom.push_back(doReplication(m_projMgr,iChild,m_projMgr));
   m_geom.back()->IncDenyDestroy();
   m_projMgr->AssertBBox();
   m_projMgr->ProjectChildrenRecurse(m_geom.back());
}

//returns the new element created from this import
TEveElement* 
FWRhoPhiZView::importElements(TEveElement* iChildren, float iLayer)
{
   float oldLayer = m_projMgr->GetCurrentDepth();
   m_projMgr->SetCurrentDepth(iLayer);
   //make sure current depth is reset even if an exception is thrown
   boost::shared_ptr<TEveProjectionManager> sentry(m_projMgr,
                                                   boost::bind(&TEveProjectionManager::SetCurrentDepth,
                                                               _1,oldLayer));
   m_projMgr->ImportElements(iChildren);
   TEveElement::List_i it = m_projMgr->BeginChildren();
   std::advance(it,
                m_projMgr->GetNChildren() -1 );
   return *it;
}

//
// const member functions
//
TGFrame* 
FWRhoPhiZView::frame() const
{
   return m_embeddedViewer->GetFrame();
}

const std::string& 
FWRhoPhiZView::typeName() const
{
   return m_typeName;
}

//
// static member functions
//

