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
// $Id: FWRhoPhiZView.cc,v 1.33 2009/01/23 21:35:44 amraktad Exp $
//

#define private public
#include "TGLOrthoCamera.h"
#undef private

#include <stdexcept>

// system include files
#include <algorithm>
#include <boost/bind.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/conversion/converter.hpp>
#include <iostream>
#include <sstream>
#include "TEveProjectionManager.h"
#include "TEveScene.h"
#include "TGLViewer.h"
//EVIL, but only way I can avoid a double delete of TGLEmbeddedViewer::fFrame
#define private public
#include "TGLEmbeddedViewer.h"
#undef private
#include "TEveViewer.h"
#include "TEveManager.h"
#include "TClass.h"
#include "TEveElement.h"
#include "TEveProjectionBases.h"
#include "TEvePolygonSetProjected.h"
#include "TEveProjections.h"
#include "TEveCalo.h"
#include "TEveProjectionAxes.h"
#include "TEveScalableStraightLineSet.h"
#include "TH2F.h"

// user include files
#include "Fireworks/Core/interface/FWRhoPhiZView.h"
#include "Fireworks/Core/interface/FWRhoPhiZViewManager.h"
#include "Fireworks/Core/interface/FWConfiguration.h"
#include "Fireworks/Core/interface/TEveElementIter.h"

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
   m_projType(iProjType),
   m_typeName(iName),
   m_caloScale(1),
   m_axes(),
   m_caloDistortion(this,"Calo compression",1.0,0.01,10.),
   m_muonDistortion(this,"Muon compression",0.2,0.01,10.),
   m_showProjectionAxes(this,"Show projection axes", false),
   m_compressMuon(this,"Compress detectors",false),
   m_caloFixedScale(this,"Calo scale",2.,0.1,100.),
   m_caloAutoScale(this,"Calo auto scale",false),
   m_showHF(0),
   m_showEndcaps(0),
//m_minEcalEnergy(this,"ECAL energy threshold (GeV)",0.,0.,100.),
//m_minHcalEnergy(this,"HCAL energy threshold (GeV)",0.,0.,100.),
   m_cameraZoom(0),
   m_cameraMatrix(0)
{
   m_projMgr.reset(new TEveProjectionManager);
   m_projMgr->SetProjection(iProjType);
   //m_projMgr->GetProjection()->SetFixedRadius(700);
   /*
      m_projMgr->GetProjection()->SetDistortion(m_distortion.value()*1e-3);
      m_projMgr->GetProjection()->SetFixR(200);
      m_projMgr->GetProjection()->SetFixZ(300);
      m_projMgr->GetProjection()->SetPastFixRFac(0.0);
      m_projMgr->GetProjection()->SetPastFixZFac(0.0);
    */

   //m_minEcalEnergy.changed_.connect(  boost::bind(&FWRhoPhiZView::updateCaloThresholdParameters, this) );
   //m_minHcalEnergy.changed_.connect(  boost::bind(&FWRhoPhiZView::updateCaloThresholdParameters, this) );
   if ( iProjType == TEveProjection::kPT_RPhi ) {
      // compression
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 130, 1.0);
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 300, 0.2);
      // projection specific parameters
      m_showEndcaps = new FWBoolParameter(this,"Show calo endcaps", true);
      m_showEndcaps->changed_.connect(  boost::bind(&FWRhoPhiZView::updateCaloParameters, this) );
      m_showHF = new FWBoolParameter(this,"Show HF", true);
      m_showHF->changed_.connect(  boost::bind(&FWRhoPhiZView::updateCaloParameters, this) );
   } else {
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 130, 1.0);
      m_projMgr->GetProjection()->AddPreScaleEntry(1, 310, 1.0);
      m_projMgr->GetProjection()->AddPreScaleEntry(0, 370, 0.2);
      m_projMgr->GetProjection()->AddPreScaleEntry(1, 580, 0.2);
   }

   gEve->AddToListTree(m_projMgr.get(),kTRUE);

   //m_distortion.changed_.connect(boost::bind(&TEveProjection::SetDistortion, m_projMgr->GetProjection(),
   //                                        boost::bind(toFloat,_1)));
   m_caloDistortion.changed_.connect(boost::bind(&FWRhoPhiZView::doDistortion,this));
   m_muonDistortion.changed_.connect(boost::bind(&FWRhoPhiZView::doDistortion,this));
   m_compressMuon.changed_.connect(boost::bind(&FWRhoPhiZView::doCompression,this,_1));
   m_caloFixedScale.changed_.connect( boost::bind(&FWRhoPhiZView::updateScaleParameters, this) );
   m_caloAutoScale.changed_.connect(  boost::bind(&FWRhoPhiZView::updateScaleParameters, this) );

   m_pad = new TEvePad;
   TGLEmbeddedViewer* ev = new TGLEmbeddedViewer(iParent, m_pad, 0);
   m_embeddedViewer=ev;
   TEveViewer* nv = new TEveViewer(iName.c_str());
   nv->SetGLViewer(ev,ev->GetFrame());
   ev->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
   if ( TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( &(ev->CurrentCamera()) ) ) {
      m_cameraZoom = &(camera->fZoom);
      m_cameraMatrix = const_cast<TGLMatrix*>(&(camera->GetCamTrans()));
      camera->SetZoomMax(1e6);
   }

   TEveScene* ns = gEve->SpawnNewScene(iName.c_str());
   m_scene.reset(ns);
   nv->AddScene(ns);
   m_viewer.reset(nv);
   //this is needed so if a TEveElement changes this view will be informed
   gEve->AddElement(nv, gEve->GetViewers());

   m_axes.reset(new TEveProjectionAxes(m_projMgr.get()));
   ns->AddElement(m_axes.get());
   gEve->AddToListTree(m_axes.get(), kTRUE);

   gEve->AddElement(m_projMgr.get(),ns);
   //ev->ResetCurrentCamera();
   m_showProjectionAxes.changed_.connect(boost::bind(&FWRhoPhiZView::showProjectionAxes,this));
}

// FWRhoPhiZView::FWRhoPhiZView(const FWRhoPhiZView& rhs)
// {
//    // do actual copying here;
// }

FWRhoPhiZView::~FWRhoPhiZView()
{
   //NOTE: have to do this EVIL activity to avoid double deletion. The fFrame inside glviewer
   // was added to a CompositeFrame which will delete it.  However, TGLEmbeddedViewer will also
   // delete fFrame in its destructor
   m_axes.destroyElement();
   m_projMgr.destroyElement();
   m_scene.destroyElement();

   TGLEmbeddedViewer* glviewer = dynamic_cast<TGLEmbeddedViewer*>(m_viewer->GetGLViewer());
   glviewer->fFrame=0;
   delete glviewer;
   m_viewer.destroyElement();
   //delete m_viewer;
   //delete m_projMgr;
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
FWRhoPhiZView::doDistortion()
{
   if ( m_projType == TEveProjection::kPT_RPhi ) {
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,1,m_caloDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,2,m_muonDistortion.value());
   } else {
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,1,m_caloDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(0,2,m_muonDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(1,1,m_caloDistortion.value());
      m_projMgr->GetProjection()->ChangePreScaleEntry(1,2,m_muonDistortion.value());
   }
   m_projMgr->UpdateName();
   m_projMgr->ProjectChildren();
   //NOTE: Looks like I have to kick the scene since it doesn't know the project changed?
   m_embeddedViewer->UpdateScene();
}

void
FWRhoPhiZView::doCompression(bool flag)
{
   m_projMgr->GetProjection()->SetUsePreScale(flag);
   m_projMgr->UpdateName();
   m_projMgr->ProjectChildren();
   //NOTE: Looks like I have to kick the scene since it doesn't know the project changed?
   m_embeddedViewer->UpdateScene();

}

/*
   void
   FWRhoPhiZView::doZoom(double iValue)
   {
   if ( TGLOrthoCamera* camera = dynamic_cast<TGLOrthoCamera*>( & (m_viewer->GetGLViewer()->CurrentCamera()) ) ) {
      // camera->SetZoom( iValue );
      camera->fZoom = iValue;
      m_viewer->GetGLViewer()->RequestDraw();
   }
   }
 */

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
                             m_projMgr.get(),
                             _1));
}
void
FWRhoPhiZView::replicateGeomElement(TEveElement* iChild)
{
   m_geom.push_back(doReplication(m_projMgr.get(),iChild,m_projMgr.get()));
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
   boost::shared_ptr<TEveProjectionManager> sentry(m_projMgr.get(),
                                                   boost::bind(&TEveProjectionManager::SetCurrentDepth,
                                                               _1,oldLayer));
   m_projMgr->ImportElements(iChildren);
   TEveElement* lastChild = m_projMgr->LastChild();
   updateCalo( lastChild, true );
   updateCaloLines( lastChild );

   return lastChild;
}

void
FWRhoPhiZView::updateCaloThresholds(TEveElement* iParent)
{
   /*
      TEveElementIter child(iParent);
      while ( TEveElement* element = child.current() )
      {
        if ( TEveCalo2D* calo2d = dynamic_cast<TEveCalo2D*>(element) ) {
           setMinEnergy(calo2d, m_minEcalEnergy.value(), "ecal");
           setMinEnergy(calo2d, m_minHcalEnergy.value(), "hcal");
        }
        child.next();
      }
    */
}

void
FWRhoPhiZView::updateCalo(TEveElement* iParent, bool dataChanged)
{
   TEveElementIter child(iParent);
   while ( TEveElement* element = child.current() )
   {
      if ( TEveCalo2D* calo2d = dynamic_cast<TEveCalo2D*>(element) ) {
         calo2d->SetValueIsColor(kFALSE);
         calo2d->SetMaxTowerH( 150 );
         calo2d->SetMaxValAbs( 150/m_caloFixedScale.value() );
         calo2d->SetScaleAbs( !m_caloAutoScale.value() );
         if ( dataChanged ) calo2d->GetData()->DataChanged();
         double eta_range = 5.191;
         if ( m_showHF && !m_showHF->value() ) eta_range = 3.0;
         if ( m_showEndcaps && !m_showEndcaps->value() ) eta_range = 1.479;
         calo2d->SetEta(-eta_range,eta_range);
         calo2d->ElementChanged();
         m_caloScale = calo2d->GetValToHeight();
         if ( m_axes ) {
            if ( m_showProjectionAxes.value() )
               m_axes->SetRnrState(kTRUE);
            else
               m_axes->SetRnrState(kFALSE);
         }
      }
      child.next();
   }
}

void
FWRhoPhiZView::updateCaloLines(TEveElement* iParent)
{
   TEveElementIter child(iParent);
   while ( TEveElement* element = child.current() )
   {
      if ( TEveStraightLineSetProjected* projected = dynamic_cast<TEveStraightLineSetProjected*>(element) )
         if ( TEveScalableStraightLineSet* line = dynamic_cast<TEveScalableStraightLineSet*>(projected->GetProjectable()) )
         {
            line->SetScale( m_caloScale );
            line->ElementChanged();
            projected->UpdateProjection();
            projected->ElementChanged();
         }
      child.next();
   }
}


void
FWRhoPhiZView::setFrom(const FWConfiguration& iFrom)
{
   // take care of parameters
   FWConfigurableParameterizable::setFrom(iFrom);

   // retrieve camera parameters
   // zoom
   assert(m_cameraZoom);
   std::string zoomName("cameraZoom"); zoomName += typeName();
   assert( 0!=iFrom.valueForKey(zoomName) );
   std::istringstream s(iFrom.valueForKey(zoomName)->value());
   s>>(*m_cameraZoom);

   // transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream os;
      os << i;
      const FWConfiguration* value = iFrom.valueForKey( matrixName + os.str() + typeName() );
      assert( value );
      std::istringstream s(value->value());
      s>>((*m_cameraMatrix)[i]);
   }
   m_viewer->GetGLViewer()->RequestDraw();
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

void
FWRhoPhiZView::addTo(FWConfiguration& iTo) const
{
   // take care of parameters
   FWConfigurableParameterizable::addTo(iTo);

   // store camera parameters
   // zoom
   assert(m_cameraZoom);
   std::ostringstream s;
   s<<(*m_cameraZoom);
   std::string name("cameraZoom");
   iTo.addKeyValue(name+typeName(),FWConfiguration(s.str()));

   // transformation matrix
   assert(m_cameraMatrix);
   std::string matrixName("cameraMatrix");
   for ( unsigned int i = 0; i < 16; ++i ){
      std::ostringstream osIndex;
      osIndex << i;
      std::ostringstream osValue;
      osValue << (*m_cameraMatrix)[i];
      iTo.addKeyValue(matrixName+osIndex.str()+typeName(),FWConfiguration(osValue.str()));
   }
}

void
FWRhoPhiZView::saveImageTo(const std::string& iName) const
{
   bool succeeded = m_viewer->GetGLViewer()->SavePicture(iName.c_str());
   if(!succeeded) {
      throw std::runtime_error("Unable to save picture to file");
   }
}

void
FWRhoPhiZView::updateScaleParameters()
{
   updateCalo(m_projMgr.get());
   updateCaloLines(m_projMgr.get());
   //NOTE: Looks like I have to kick the scene since it doesn't know the project changed?
   m_embeddedViewer->UpdateScene();
}

void
FWRhoPhiZView::updateCaloParameters()
{
   updateCalo(m_projMgr.get());
   //NOTE: Looks like I have to kick the scene since it doesn't know the project changed?
   m_embeddedViewer->UpdateScene();
}

void
FWRhoPhiZView::updateCaloThresholdParameters()
{
   //updateCaloThresholds(m_projMgr);
}


void
FWRhoPhiZView::setMinEnergy( TEveCalo2D* calo, double value, std::string name )
{
   if ( !calo->GetData() ) return;
   for ( int i = 0; i < calo->GetData()->GetNSlices(); ++i ) {
      std::string histName(calo->GetData()->RefSliceInfo(i).fHist->GetName());
      if ( histName.find(name,0) != std::string::npos  )
      {
         calo->GetData()->RefSliceInfo(i).fThreshold = value;
         calo->ElementChanged();
         calo->DataChanged();
         break;
      }
   }
}

void FWRhoPhiZView::showProjectionAxes( )
{
   if ( !m_axes ) return; // just in case
   if ( m_showProjectionAxes.value() )
      m_axes->SetRnrState(kTRUE);
   else
      m_axes->SetRnrState(kFALSE);
   gEve->Redraw3D();
}

//
// static member functions
//

