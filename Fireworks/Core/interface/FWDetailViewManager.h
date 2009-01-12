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
// $Id: FWDetailViewManager.h,v 1.7 2009/01/09 20:58:49 chrjones Exp $
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
class TGTextView;
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
      void close_wm ();
      void close_button ();

   private:
      FWDetailViewManager(const FWDetailViewManager&); // stop default

      const FWDetailViewManager& operator=(const FWDetailViewManager&); // stop default

protected:
     // ---------- member data --------------------------------
     TEveScene 		*ns;
     TEveViewer		*nv;
     TGMainFrame	*frame;
     TGTextView		*text_view;

     std::map<std::string, FWDetailViewBase *>	m_viewers;
};


#endif
