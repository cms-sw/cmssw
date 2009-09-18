#ifndef Fireworks_Core_FWDetailViewManager_h
#define Fireworks_Core_FWDetailViewManager_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWDetailViewManager
//
/**\class FWDetailViewManager FWDetailViewManager.h Fireworks/Core/interface/FWDetailViewManager.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Wed Mar  5 09:13:43 EST 2008
// $Id: FWDetailViewManager.h,v 1.16 2009/07/15 14:10:27 amraktad Exp $
//

// system include files
#include <map>
#include <string>
// user include files

// forward declarations
class TGLEmbeddedViewer;
class TGPack;
class TEveScene;
class TEveViewer;
class TGMainFrame;
class TCanvas;
class TRootEmbeddedCanvas;
class FWDetailViewBase;
class FWModelId;
class TGWindow;

class FWDetailViewManager
{
public:
   FWDetailViewManager(const TGWindow*);
   virtual ~FWDetailViewManager();

   // ---------- const member functions ---------------------
   bool haveDetailViewFor(const FWModelId&) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void openDetailViewFor(const FWModelId& );
   void saveImage() const;

private:
   FWDetailViewManager(const FWDetailViewManager&);    // stop default

   const FWDetailViewManager& operator=(const FWDetailViewManager&);    // stop default

   std::string findViewerFor(const std::string&) const;
   void createDetailViewFrame();

protected:
   // ---------- member data --------------------------------
  const TGWindow        *m_parentWindow;

   FWDetailViewBase     *m_detailView;

   Bool_t                m_modeGL;

   TGMainFrame          *m_mainFrame;
   TGPack               *m_pack;

   TRootEmbeddedCanvas  *m_textCanvas;
   TRootEmbeddedCanvas  *m_viewCanvas;
   TEveScene            *m_sceneGL;
   TGLEmbeddedViewer    *m_viewerGL;



   std::map<std::string, FWDetailViewBase *>  m_detailViews;
   mutable std::map<std::string, std::string> m_typeToViewers;
};


#endif
