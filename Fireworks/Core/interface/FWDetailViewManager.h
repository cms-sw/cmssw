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
// $Id: FWDetailViewManager.h,v 1.13 2009/03/31 10:45:25 amraktad Exp $
//

// system include files
#include <map>
#include <string>
// user include files

// forward declarations
class TGLEmbeddedViewer;
class TEveScene;
class TEveViewer;
class TGMainFrame;
class TCanvas;

class FWDetailViewBase;
class FWModelId;

class FWDetailViewManager
{
public:
   FWDetailViewManager();
   virtual ~FWDetailViewManager();

   // ---------- const member functions ---------------------
   bool haveDetailViewFor(const FWModelId&) const;

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void openDetailViewFor(const FWModelId& );
   void close_button ();

private:
   FWDetailViewManager(const FWDetailViewManager&);    // stop default

   const FWDetailViewManager& operator=(const FWDetailViewManager&);    // stop default

   std::string findViewerFor(const std::string&) const;
  void createDetailViewFrame();

protected:
   // ---------- member data --------------------------------
   FWDetailViewBase *m_detailView;

   TEveScene          *m_scene;
   TGLEmbeddedViewer   *m_viewer;

   TGMainFrame      *m_frame;
   TCanvas              *m_latexCanvas; 

   std::map<std::string, FWDetailViewBase *>  m_detailViews;
   mutable std::map<std::string, std::string>    m_typeToViewers;
};


#endif
