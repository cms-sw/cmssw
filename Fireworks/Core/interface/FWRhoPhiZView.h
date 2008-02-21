#ifndef Fireworks_Core_FWRhoPhiZView_h
#define Fireworks_Core_FWRhoPhiZView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWRhoPhiZView
// 
/**\class FWRhoPhiZView FWRhoPhiZView.h Fireworks/Core/interface/FWRhoPhiZView.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue Feb 19 10:33:21 EST 2008
// $Id$
//

// system include files
#include <string>

// user include files
#include "TGLViewer.h"
#include "TEveProjections.h"

// forward declarations
class TEvePad;
class TEveViewer;
class TGLEmbeddedViewer;
class TEveProjectionManager;
class TGFrame;

class FWRhoPhiZView
{

   public:
      FWRhoPhiZView(TGFrame* iParent,
                    const std::string& iTypeName,
                    const TEveProjection::EPType_e& iProjType);
      virtual ~FWRhoPhiZView();

      // ---------- const member functions ---------------------
      TGFrame* frame() const;
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void destroyElements();
      void replicateGeomElement(TEveElement*);

      //returns the new element created from this import
      TEveElement* importElements(TEveElement*);
   private:
      FWRhoPhiZView(const FWRhoPhiZView&); // stop default

      const FWRhoPhiZView& operator=(const FWRhoPhiZView&); // stop default

      // ---------- member data --------------------------------
      TEvePad* m_pad;
      TEveViewer* m_viewer;
      TGLEmbeddedViewer* m_embeddedViewer;
      TEveProjectionManager* m_projMgr;
      std::vector<TEveElement*> m_geom;


};


#endif
