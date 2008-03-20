#ifndef Fireworks_Core_FWEveLegoView_h
#define Fireworks_Core_FWEveLegoView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWEveLegoView
// 
/**\class FWEveLegoView FWEveLegoView.h Fireworks/Core/interface/FWEveLegoView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FWEveLegoView.h,v 1.1.2.2 2008/03/17 02:19:58 dmytro Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"

// forward declarations
class TGFrame;
class TGLEmbeddedViewer;
class TEveCaloLego;
class TEveCaloDataHist;
class TEvePad;
class TEveViewer;
class TEveScene;
class TEveElementList;

class FWEveLegoView : public FWViewBase
{

   public:
      FWEveLegoView(TGFrame*, TEveElementList*);
      virtual ~FWEveLegoView();

      // ---------- const member functions ---------------------
      TGFrame* frame() const;
      const std::string& typeName() const;
     
      // ---------- static member functions --------------------
      static const std::string& staticTypeName();
   
      // ---------- member functions ---------------------------
      void draw(TEveCaloDataHist* data);
   
   private:
      FWEveLegoView(const FWEveLegoView&); // stop default

      const FWEveLegoView& operator=(const FWEveLegoView&); // stop default

      void doMinThreshold(double);
   
      // ---------- member data --------------------------------
      TEvePad* m_pad;
      TEveViewer* m_viewer;
      TGLEmbeddedViewer* m_embeddedViewer;
      TEveScene* m_scene;
      TEveCaloLego* m_lego;
      
      // FWLongParameter m_range;
      FWDoubleParameter m_range;
};


#endif
