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
// $Id: FWGUIEventDataAdder.h,v 1.10 2009/09/23 20:28:17 chrjones Exp $
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
class TGCheckButton;
class TFile;
class FWTypeToRepresentations;
namespace fwlite {
   class Event;
}
class FWTableWidget;
class DataAdderTableManager;

class FWGUIEventDataAdder {
   RQ_OBJECT("FWGUIEventDataAdder")
public:
   FWGUIEventDataAdder(UInt_t w,UInt_t,
                       FWEventItemsManager*, TGFrame*,
                       const fwlite::Event*,
                       const TFile*,
                       const FWTypeToRepresentations& iTypeAndReps);
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
   
   void rowClicked(Int_t iRow,Int_t iButton,Int_t iKeyMod,Int_t,Int_t);
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
   TGCheckButton* m_doNotUseProcessName;

   std::string m_purpose;

   std::string m_type;
   std::string m_moduleLabel;
   std::string m_productInstanceLabel;
   std::string m_processName;

   std::vector<std::string> m_processNamesInFile;
   
   DataAdderTableManager* m_tableManager;
   FWTableWidget* m_tableWidget;
   TGTextButton* m_apply;

   FWTypeToRepresentations* m_typeAndReps;
   std::vector<Data> m_useableData;
};


#endif
