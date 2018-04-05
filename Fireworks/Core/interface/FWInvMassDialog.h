#ifndef Fireworks_Core_FWInvMassDialog_h
#define Fireworks_Core_FWInvMassDialog_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     FWInvMassDialog
// 
/**\class FWInvMassDialog FWInvMassDialog.h Fireworks/Core/interface/FWInvMassDialog.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Matevz Tadel
//         Created:  Mon Nov 22 11:05:41 CET 2010
//

// system include files

// user include files
#include "TGFrame.h"

// forward declarations
class FWSelectionManager;

class TGTextView;
class TGTextButton;

class FWInvMassDialog : public TGMainFrame
{

public:
   FWInvMassDialog(FWSelectionManager* sm);
   ~FWInvMassDialog() override;

   void CloseWindow() override;

   // ---------- const member functions ---------------------

   // ---------- static member functions --------------------

   // ---------- member functions ---------------------------

   void Calculate();

protected:
   void beginUpdate();
   void addLine(const TString& line);
   void endUpdate();

private:
   FWInvMassDialog(const FWInvMassDialog&); // stop default

   const FWInvMassDialog& operator=(const FWInvMassDialog&); // stop default

   // ---------- member data --------------------------------

   FWSelectionManager *m_selectionMgr;

   TGTextView   *m_text;
   TGTextButton *m_button;

   bool          m_firstLine;

   ClassDefOverride(FWInvMassDialog, 0);
};


#endif
