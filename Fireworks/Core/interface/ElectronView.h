#ifndef Fireworks_ElectronView_h
#define Fireworks_ElectronView_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     ElectronSCViewManager
//
/**\class ElectronSCViewManager ElectronSCViewManager.h Fireworks/Core/interface/ElectronSCViewManager.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:
//         Created:  Sun Jan  6 22:01:21 EST 2008
// $Id: ElectronView.h,v 1.2 2008/03/06 22:48:31 jmuelmen Exp $
//

// system include files
#include <string>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>

class TEveScene;
class TEveViewer;

class ElectronView {

public:
     ElectronView();
     virtual ~ElectronView();

     // ---------- const member functions ---------------------

     // ---------- static member functions --------------------

     // ---------- member functions ---------------------------
     virtual void event();
//      void addElements ();
     void close_wm ();
     void close_button ();

private:
     ElectronView(const ElectronView&); // stop default
     const ElectronView& operator=(const ElectronView&); // stop default

protected:
     // ---------- member data --------------------------------
     TEveScene 		*ns;
     TEveViewer		*nv;
     TGMainFrame	*frame;
};


#endif
