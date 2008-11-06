#ifndef Fireworks_Core_CSGNumAction_h
#define Fireworks_Core_CSGNumAction_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     CSGNumAction
//
/**\class CSGNumAction CSGNumAction.h Fireworks/Core/interface/CSGNumAction.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:
//         Created:  Thu Jun 26 12:49:02 EDT 2008
// $Id: CSGNumAction.h,v 1.1 2008/06/29 13:18:33 chrjones Exp $
//

// system include files
#include <string>
#include <sigc++/sigc++.h>
#include <TGNumberEntry.h>

// user include files

// forward declarations
class CmsShowMainFrame;
class CSGAction;
//class CSGConnector;

class CSGNumAction
{

   public:
      CSGNumAction(CmsShowMainFrame *frame, const char *name);
      virtual ~CSGNumAction();

      // ---------- const member functions ---------------------
      const std::string& getName() const;
      TGNumberEntryField* getNumberEntry() const;
      Double_t getNumber() const;
      Bool_t isEnabled() const;

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void createNumberEntry(TGCompositeFrame* p, TGLayoutHints* l = 0);//, Int_t id = -1, Double_t val = 0, TGNumberFormat::EStyle style = kNESReal, TGNumberFormat::EAttribute attr = kNEAAnyNumber, TGNumberFormat::ELimit limits = kNELNoLimits, Double_t min = 0, Double_t max = 1);
      void enable();
      void disable();
      void setNumber(Double_t num);

      void activate();
      sigc::signal<void, double> activated;

   private:
      CSGNumAction(const CSGNumAction&); // stop default

      const CSGNumAction& operator=(const CSGNumAction&); // stop default

      // ---------- member data --------------------------------
      CmsShowMainFrame *m_frame;
      std::string m_name;
      TGNumberEntryField* m_numberEntry;
      //      CSGConnector* m_connector;
      Bool_t m_enabled;

};


#endif
