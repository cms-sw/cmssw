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
// $Id: FWGUIValidatingTextEntry.h,v 1.5 2009/10/02 17:55:27 dmytro Exp $
//

// system include files
#include <vector>
#include <string>
#include <boost/shared_ptr.hpp>

// user include files
#include "TGTextEntry.h"

// forward declarations
class FWValidatorBase;
class TGComboBoxPopup;
class TGListBox;

class FWGUIValidatingTextEntry : public TGTextEntry {

public:
   FWGUIValidatingTextEntry(const TGWindow *parent = 0, const char *text = 0, Int_t id = -1);

   virtual ~FWGUIValidatingTextEntry();

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------
   void setValidator(FWValidatorBase*);
   void showOptions();
   void hideOptions();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   void keyPressedInPopup(TGFrame*, UInt_t keysym, UInt_t mask);
private:
   FWGUIValidatingTextEntry(const FWGUIValidatingTextEntry&); // stop default

   const FWGUIValidatingTextEntry& operator=(const FWGUIValidatingTextEntry&); // stop default
   void insertTextOption(const std::string&);

   // ---------- member data --------------------------------
   TGComboBoxPopup* m_popup;
   TGListBox* m_list;

   FWValidatorBase* m_validator;
   std::vector<std::pair<boost::shared_ptr<std::string>, std::string> > m_options;

   ClassDef(FWGUIValidatingTextEntry, 1);
};


#endif
