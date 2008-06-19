#ifndef Fireworks_Core_FWGlimpseView_h
#define Fireworks_Core_FWGlimpseView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGlimpseView
// 
/**\class FWGlimpseView FWGlimpseView.h Fireworks/Core/interface/FWGlimpseView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FWGlimpseView.h,v 1.5 2008/06/11 13:57:32 dmytro Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "TEveCaloData.h"

// forward declarations
class TGFrame;
class TGLEmbeddedViewer;
class TEvePad;
class TEveViewer;
class TEveScene;
class TEveElementList;
class TGLMatrix;

class FWGlimpseView : public FWViewBase
{

   public:
      FWGlimpseView(TGFrame*, TEveElementList*);
      virtual ~FWGlimpseView();

      // ---------- const member functions ---------------------
      TGFrame* frame() const;
      const std::string& typeName() const;
      virtual void addTo(FWConfiguration&) const;

      virtual void saveImageTo(const std::string& iName) const;

      // ---------- static member functions --------------------
      static const std::string& staticTypeName();
   
      // ---------- member functions ---------------------------
      void draw(TEveCaloDataHist* data);
      virtual void setFrom(const FWConfiguration&);
   
   private:
      FWGlimpseView(const FWGlimpseView&); // stop default

      const FWGlimpseView& operator=(const FWGlimpseView&); // stop default

      // ---------- member data --------------------------------
      TEvePad* m_pad;
      TEveViewer* m_viewer;
      TGLEmbeddedViewer* m_embeddedViewer;
      TEveScene* m_scene;
      
      TGLMatrix* m_cameraMatrix;
      TGLMatrix* m_cameraMatrixBase;
};


#endif
