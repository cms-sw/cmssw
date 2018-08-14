#ifndef Fireworks_Core_FWGUIValidatingTextEntry_h
#define Fireworks_Core_FWGUIValidatingTextEntry_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWGUIValidatingTextEntry
//
/**\class FWGUIValidatingTextEntry FWGUIValidatingTextEntry.h Fireworks/Core/interface/FWGUIValidatingTextEntry.h

   Description: <one line class summary>

   Usage:
    <usage>

 */
//
// Original Author:  Chris Jones
//         Created:  Fri Aug 22 18:13:29 EDT 2008
//

// system include files
#include <vector>
#include <string>
#ifndef __CINT__
#include <memory>
#endif
// user include files
#include "TGTextEntry.h"

// forward declarations
class FWValidatorBase;
class TGComboBoxPopup;
class TGListBox;

class FWGUIValidatingTextEntry : public TGTextEntry {

public:
   FWGUIValidatingTextEntry(const TGWindow *parent = nullptr, const char *text = nullptr, Int_t id = -1);

   ~FWGUIValidatingTextEntry() override;

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setValidator(FWValidatorBase*);
   void showOptions();
   void hideOptions();

   TGListBox* getListBox() const { return m_list; }
   void setMaxListBoxHeight(UInt_t x) { m_listHeight = x; }

   Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2) override;

   void keyPressedInPopup(TGFrame*, UInt_t keysym, UInt_t mask);
 
   ClassDefOverride(FWGUIValidatingTextEntry, 0);

private:
   FWGUIValidatingTextEntry(const FWGUIValidatingTextEntry&); // stop default

   const FWGUIValidatingTextEntry& operator=(const FWGUIValidatingTextEntry&); // stop default
   void insertTextOption(const std::string&);

   // ---------- member data --------------------------------
   TGComboBoxPopup* m_popup;
   TGListBox*       m_list;
   FWValidatorBase* m_validator;

   UInt_t           m_listHeight;
#ifndef __CINT__
   std::vector<std::pair<std::shared_ptr<std::string>, std::string> > m_options;
#endif
};


#endif
