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
// $Id: FWGUIEventDataAdder.h,v 1.16 2010/12/02 09:22:26 matevz Exp $
//

// system include files
#include <set>
#include <vector>
#include <string>
#include <RQ_OBJECT.h>

// user include files

// forward declarations
class FWEventItemsManager;
class FWJobMetadataManager;
class TGTransientFrame;
class TGTextEntry;
class TGTextButton;
class TGCheckButton;
class TFile;
class FWTableWidget;
class DataAdderTableManager;

class FWGUIEventDataAdder
{
   RQ_OBJECT("FWGUIEventDataAdder")

public:
   FWGUIEventDataAdder(UInt_t w,UInt_t,
                       FWEventItemsManager*, TGFrame*,
                       FWJobMetadataManager *);
   virtual ~FWGUIEventDataAdder();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void addNewItem();
   void addNewItemAndClose();
   void show();

   void windowIsClosing();
   void updateFilterString(const char *str);   
   void rowClicked(Int_t iRow,Int_t iButton,Int_t iKeyMod,Int_t,Int_t);

   void metadataUpdatedSlot(void);

   void resetNameEntry();

private:
   FWGUIEventDataAdder(const FWGUIEventDataAdder&); // stop default
   void createWindow();
 
   void newIndexSelected(int);
   const FWGUIEventDataAdder& operator=(const FWGUIEventDataAdder&); // stop default

   // ---------- member data --------------------------------
   FWEventItemsManager*     m_manager;
   FWJobMetadataManager*    m_metadataManager;

   TGFrame*          m_parentFrame;
   TGTransientFrame* m_frame;
   TGTextEntry*      m_name;
   TGCheckButton*    m_doNotUseProcessName;
   TGTextButton*     m_apply;
   TGTextButton*     m_applyAndClose;
   TGTextEntry*      m_search;

   std::string m_purpose;

   std::string m_type;
   std::string m_moduleLabel;
   std::string m_productInstanceLabel;
   std::string m_processName;

   std::string m_expr; // this the search term for use in searchData()
   
   DataAdderTableManager*   m_tableManager;
   FWTableWidget*           m_tableWidget;
};


#endif
