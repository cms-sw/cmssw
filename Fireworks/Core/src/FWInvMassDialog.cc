// -*- C++ -*-
//
// Package:     Core
// Class  :     FWInvMassDialog
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Matevz Tadel
//         Created:  Mon Nov 22 11:05:57 CET 2010
// $Id$
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWInvMassDialog.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "TClass.h"

#include "TGTextView.h"
#include "TGButton.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

FWInvMassDialog::FWInvMassDialog(FWSelectionManager* sm) :
   TGMainFrame(gClient->GetRoot(), 420, 220),
   m_selectionMgr(sm),
   m_text(0),
   m_button(0)
{
   SetWindowName("Invariant Mass Dialog");
   SetCleanup(kDeepCleanup);

   m_text = new TGTextView(this);
   AddFrame(m_text, new TGLayoutHints(kLHintsNormal | kLHintsExpandX | kLHintsExpandY, 1, 1, 1, 1));

   m_button = new TGTextButton(this, "Calculate");
   AddFrame(m_button, new TGLayoutHints(kLHintsNormal | kLHintsExpandX, 1, 1, 1, 1));

   m_button->Connect("Clicked()","FWInvMassDialog", this, "Calculate()");

   Layout();
   MapSubwindows();
}

// FWInvMassDialog::FWInvMassDialog(const FWInvMassDialog& rhs)
// {
//    // do actual copying here;
// }

FWInvMassDialog::~FWInvMassDialog()
{
}

void FWInvMassDialog::CloseWindow()
{
   UnmapWindow();
}

//
// assignment operators
//
// const FWInvMassDialog& FWInvMassDialog::operator=(const FWInvMassDialog& rhs)
// {
//   //An exception safe implementation is
//   FWInvMassDialog temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void FWInvMassDialog::beginUpdate()
{
   m_text->Clear();
   m_firstLine = true;
}

void FWInvMassDialog::addLine(const TString& line)
{
   TGText *txt = m_text->GetText();

   if (m_firstLine)
   {
      txt->InsText(TGLongPosition(0, 0), line);
      m_firstLine = false;
   }
   else
   {
      txt->InsText(TGLongPosition(0, txt->RowCount()), line);
   }
}

void FWInvMassDialog::endUpdate()
{
   m_text->Update();
}

void FWInvMassDialog::Calculate()
{
   const std::set<FWModelId>& sted = m_selectionMgr->selected();

   beginUpdate();

   addLine(TString::Format(" %d items in selection", (int) sted.size()));
   addLine("");
   addLine("---------------------------------------+--------------");
   addLine("      pT           pz           mass   | Collection");
   addLine("---------------------------------------+--------------");

   TClass *rc_class = TClass::GetClass(typeid(reco::Candidate));

   math::XYZTLorentzVector sum;

   for (std::set<FWModelId>::const_iterator i = sted.begin(); i != sted.end(); ++i)
   {
      TString line;

      TClass *model_class = const_cast<TClass*>(i->item()->modelType());
      void   *model_data  = const_cast<void*>  (i->item()->modelData(i->index()));

      reco::Candidate *rc = reinterpret_cast<reco::Candidate*>
         (model_class->DynamicCast(rc_class, model_data));

      if (rc != 0)
      {
         const math::XYZTLorentzVector &v = rc->p4();
         line = TString::Format("  %10.3f, %+11.3f, %10.3f", v.pt(), v.pz(), v.mass());
         sum += v;
      }
      else
      {
         line = TString::Format("  ------ not a reco::Candidate ------");
      }
      line += TString::Format("  | %s[%d]", i->item()->name().c_str(), i->index());

      addLine(line);
   }

   addLine("-------------------------------------------------------");
   addLine(TString::Format("  %10.3f, %+11.3f, %10.3f", sum.pt(), sum.pz(), sum.mass()));

   endUpdate();
}

//
// const member functions
//

//
// static member functions
//
