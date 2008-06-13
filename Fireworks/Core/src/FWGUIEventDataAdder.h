#ifndef Fireworks_Core_FWGUIEventDataAdder_h
#define Fireworks_Core_FWGUIEventDataAdder_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUIEventDataAdder
// 
/**\class FWGUIEventDataAdder FWGUIEventDataAdder.h Fireworks/Core/interface/FWGUIEventDataAdder.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 13 09:12:36 EDT 2008
// $Id$
//

// system include files
#include <RQ_OBJECT.h>

// user include files

// forward declarations
class FWEventItemsManager;
class TGMainFrame;
class TGTextEntry;
class TGTextButton;

class FWGUIEventDataAdder {
   RQ_OBJECT("FWGUIEventDataAdder")
public:
   FWGUIEventDataAdder(UInt_t w,UInt_t, FWEventItemsManager*);
   virtual ~FWGUIEventDataAdder();
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void addNewItem();
   void show();
   
   void windowIsClosing();
private:
   FWGUIEventDataAdder(const FWGUIEventDataAdder&); // stop default
   void createWindow();
   
   const FWGUIEventDataAdder& operator=(const FWGUIEventDataAdder&); // stop default
   
   // ---------- member data --------------------------------
   FWEventItemsManager* m_manager;
   
   TGMainFrame* m_frame;
   TGTextEntry* m_name;
   TGTextEntry* m_purpose;

   TGTextEntry* m_type;
   TGTextEntry* m_moduleLabel;
   TGTextEntry* m_productInstanceLabel;
   TGTextEntry* m_processName;
   TGTextButton* m_apply;
   
};


#endif
