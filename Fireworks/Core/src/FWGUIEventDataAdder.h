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
// $Id: FWGUIEventDataAdder.h,v 1.1 2008/06/13 23:38:18 chrjones Exp $
//

// system include files
#include <set>
#include <vector>
#include <string>
#include <RQ_OBJECT.h>

// user include files

// forward declarations
class FWEventItemsManager;
class TGTransientFrame;
class TGTextEntry;
class TGTextButton;
class TFile;
namespace fwlite {
   class Event;
}
class LightTableWidget;
class DataAdderTableManager;

class FWGUIEventDataAdder {
   RQ_OBJECT("FWGUIEventDataAdder")
public:
   FWGUIEventDataAdder(UInt_t w,UInt_t, 
                       FWEventItemsManager*, TGFrame*,
                       const fwlite::Event*,
                       const TFile*,
                       const std::set<std::pair<std::string,std::string> >& iTypeAndPurpose);
   virtual ~FWGUIEventDataAdder();
   
   struct Data {
      std::string purpose_;
      std::string type_;
      std::string moduleLabel_;
      std::string productInstanceLabel_;
      std::string processName_;
   };
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void addNewItem();
   void show();
   
   void windowIsClosing();
   void update(const TFile*, const fwlite::Event*);
private:
   FWGUIEventDataAdder(const FWGUIEventDataAdder&); // stop default
   void createWindow();
   
   void fillData(const TFile*);
   void newIndexSelected(int);
   const FWGUIEventDataAdder& operator=(const FWGUIEventDataAdder&); // stop default
   
   // ---------- member data --------------------------------
   FWEventItemsManager* m_manager;
   const fwlite::Event* m_presentEvent;
   
   TGFrame* m_parentFrame;
   
   TGTransientFrame* m_frame;
   TGTextEntry* m_name;
   
   TGTextEntry* m_purpose;

   TGTextEntry* m_type;
   TGTextEntry* m_moduleLabel;
   TGTextEntry* m_productInstanceLabel;
   TGTextEntry* m_processName;

   DataAdderTableManager* m_tableManager;
   LightTableWidget* m_tableWidget;
   TGTextButton* m_apply;
   
   std::set<std::pair<std::string,std::string> > m_typeAndPurpose;
   std::vector<Data> m_useableData;
};


#endif
