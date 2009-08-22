#ifndef Fireworks_Core_FWDetailViewBase_h
#define Fireworks_Core_FWDetailViewBase_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewBase
//
/**\class FWDetailViewBase FWDetailViewBase.h Fireworks/Core/interface/FWDetailViewBase.h

   Description: Base class for detailed views

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Jan  9 13:35:52 EST 2009
// $Id: FWDetailViewBase.h,v 1.6 2009/06/22 14:32:25 amraktad Exp
// system include files


#include "Fireworks/Core/interface/FWSimpleProxyHelper.h"

class TCanvas;
class TGCompositeFrame;
class TCanvas;
class TGPack;

class TGLViewer;
class TEveScene ;
class TEveWindowSlot;
class TEveWindow;
class TGVerticalFrame;

class FWModelId;

class FWDetailViewBase
{
public:
   virtual ~FWDetailViewBase ();

   void  build (const FWModelId&, TEveWindowSlot*);
   TEveWindow*  getEveWindow() { return m_eveWindow; }

protected:
   FWDetailViewBase(const std::type_info&);

   void makePackCanvas(TEveWindowSlot *&slot, TGVerticalFrame *&guiFrame, TCanvas *&viewCanvas);
   void makePackViewer(TEveWindowSlot *&slot, TGVerticalFrame *&guiFrame, TGLViewer *&viewer, TEveScene *&scene);

   TEveWindow           *m_eveWindow;

private:
   FWDetailViewBase(const FWDetailViewBase&); // stop default
   const FWDetailViewBase& operator=(const FWDetailViewBase&); // stop default

   virtual void build(const FWModelId&, const void*, TEveWindowSlot* slot) = 0;

   FWSimpleProxyHelper m_helper;
};

#endif
