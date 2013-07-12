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
// $Id: FWInvMassDialog.cc,v 1.3 2010/12/06 16:06:44 matevz Exp $
//

// system include files

// user include files
#include "Fireworks/Core/interface/FWInvMassDialog.h"
#include "Fireworks/Core/interface/FWSelectionManager.h"
#include "Fireworks/Core/interface/FWEventItem.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TClass.h"
#include "TMath.h"

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
   TGMainFrame(gClient->GetRoot(), 470, 240),
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
   addLine("--------------------------------------------------+--------------");
   addLine("       px          py          pz          pT     | Collection");
   addLine("--------------------------------------------------+--------------");

   TClass *rc_class  = TClass::GetClass(typeid(reco::Candidate));
   TClass *rtb_class = TClass::GetClass(typeid(reco::TrackBase));

   math::XYZVector sum;
   double          sum_len = 0;
   double          sum_len_xy = 0;

   math::XYZVector first, second; int n = 0;

   for (std::set<FWModelId>::const_iterator i = sted.begin(); i != sted.end(); ++i, ++n)
   {
      TString line;

      TClass *model_class = const_cast<TClass*>(i->item()->modelType());
      void   *model_data  = const_cast<void*>  (i->item()->modelData(i->index()));

      math::XYZVector v;
      bool            ok_p = false;

      reco::Candidate *rc = reinterpret_cast<reco::Candidate*>
         (model_class->DynamicCast(rc_class, model_data));

      if (rc != 0)
      {
         ok_p = true;
         v.SetXYZ(rc->px(), rc->py(), rc->pz());
      }
      else
      {
         reco::TrackBase *rtb = reinterpret_cast<reco::TrackBase*>
            (model_class->DynamicCast(rtb_class, model_data));

         if (rtb != 0)
         {
            ok_p = true;
            v.SetXYZ(rtb->px(), rtb->py(), rtb->pz());
         }
      }

      if (ok_p)
      {
         sum        += v;
         sum_len    += TMath::Sqrt(v.mag2());
         sum_len_xy += TMath::Sqrt(v.perp2());

         line = TString::Format("  %+10.3f  %+10.3f  %+10.3f  %10.3f", v.x(), v.y(), v.z(), TMath::Sqrt(v.perp2()));

      }
      else
      {
         line = TString::Format("  -------- not a Candidate or TrackBase --------");
      }
      line += TString::Format("  | %s[%d]", i->item()->name().c_str(), i->index());

      addLine(line);

      if (n == 0) first = v; else if (n == 1) second = v;
   }

   addLine("--------------------------------------------------+--------------");
   addLine(TString::Format("  %+10.3f  %+10.3f  %+10.3f  %10.3f  | Sum", sum.x(), sum.y(), sum.z(), TMath::Sqrt(sum.perp2())));
   addLine("");
   addLine(TString::Format("m  = %10.3f", TMath::Sqrt(TMath::Max(0.0, sum_len    * sum_len    - sum.mag2()))));
   addLine(TString::Format("mT = %10.3f", TMath::Sqrt(TMath::Max(0.0, sum_len_xy * sum_len_xy - sum.perp2()))));
   addLine(TString::Format("HT = %10.3f", sum_len_xy));
 
   if (n == 2) {
      addLine(TString::Format("deltaPhi  = %+6.4f", deltaPhi(first.Phi(), second.Phi())));
      addLine(TString::Format("deltaEta  = %+6.4f", first.Eta()- second.Eta()));
      addLine(TString::Format("deltaR    = % 6.4f", deltaR(first.Eta(), first.Phi(), second.Eta(), second.Phi())));
   }

   endUpdate();
}

//
// const member functions
//

//
// static member functions
//
