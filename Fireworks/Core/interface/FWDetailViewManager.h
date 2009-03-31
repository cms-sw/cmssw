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
// $Id: FWDetailViewManager.h,v 1.12 2009/03/29 14:13:38 amraktad Exp $
//

// system include files
#include <map>
#include <string>
// user include files

// forward declarations
class FWModelId;
class TEveScene;
class TEveViewer;
class TGMainFrame;
class TLatex;
class FWDetailViewBase;

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
   void destroyed_wm ();
   void close_button ();

private:
   FWDetailViewManager(const FWDetailViewManager&);    // stop default

   const FWDetailViewManager& operator=(const FWDetailViewManager&);    // stop default

   std::string findViewerFor(const std::string&) const;
protected:
   // ---------- member data --------------------------------
   TEveScene          *m_scene;
   TEveViewer         *m_viewer;
   TGMainFrame        *m_frame;
   TLatex             *m_latex;

   std::map<std::string, FWDetailViewBase *>  m_viewers;
   mutable std::map<std::string, std::string> m_typeToViewers;
};


#endif
