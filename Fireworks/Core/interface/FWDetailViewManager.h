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
// $Id: FWDetailViewManager.h,v 1.3 2008/03/07 01:05:13 jmuelmen Exp $
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
class FWDetailView;

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
      void registerDetailView (const std::string &, FWDetailView *);
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

     std::map<std::string, FWDetailView *>	m_viewers;
};


#endif
