#ifndef Fireworks_Core_FW3DView_h
#define Fireworks_Core_FW3DView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FW3DView
//
/**\class FW3DView FW3DView.h Fireworks/Core/interface/FW3DView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Feb 21 11:22:37 EST 2008
// $Id: FW3DView.h,v 1.8 2008/11/06 22:05:23 amraktad Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "Fireworks/Core/interface/FWViewBase.h"
#include "Fireworks/Core/interface/FWLongParameter.h"
#include "Fireworks/Core/interface/FWDoubleParameter.h"
#include "Fireworks/Core/interface/FWBoolParameter.h"
#include "TEveCaloData.h"

// forward declarations
class TGFrame;
class TGLEmbeddedViewer;
class TEvePad;
class TEveViewer;
class TEveScene;
class TEveElementList;
class TEveGeoShape;
class TGLMatrix;
class FW3DViewManager;
class FW3DView : public FWViewBase
{

   public:
      FW3DView(TGFrame*, TEveElementList*);
      virtual ~FW3DView();

      // ---------- const member functions ---------------------
      TGFrame* frame() const;
      const std::string& typeName() const;
      virtual void addTo(FWConfiguration&) const;

      virtual void saveImageTo(const std::string& iName) const;

      // ---------- static member functions --------------------
      static const std::string& staticTypeName();

      // ---------- member functions ---------------------------
      virtual void setFrom(const FWConfiguration&);
      TEveScene* getScene() { return m_scene; }

   private:
      FW3DView(const FW3DView&); // stop default

      const FW3DView& operator=(const FW3DView&); // stop default

      // ---------- member data --------------------------------
      TEvePad* m_pad;
      TEveViewer* m_viewer;
      TGLEmbeddedViewer* m_embeddedViewer;
      TEveScene* m_scene;

      TGLMatrix* m_cameraMatrix;
      TGLMatrix* m_cameraMatrixBase;
      Double_t*  m_cameraFOV;
};


#endif
