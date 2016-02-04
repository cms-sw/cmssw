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
// $Id: FWInvMassDialog.h,v 1.1 2010/11/22 21:35:07 matevz Exp $
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
   virtual ~FWInvMassDialog();

   virtual void CloseWindow();

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

   ClassDef(FWInvMassDialog, 0);
};


#endif
