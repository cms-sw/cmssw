#include "Fireworks/Core/src/FW3DViewDistanceMeasureTool.h"
#include "TGFrame.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TG3DLine.h"

namespace {
const char * lbpformat="(%.2f, %.2f, %.2f)";
}

FW3DViewDistanceMeasureTool::FW3DViewDistanceMeasureTool():
  m_action(kNone), m_bp1(0), m_bp2(0),
  m_lp1(0), m_lp2(0), m_ldist(0)
{
}

void FW3DViewDistanceMeasureTool::Print() const
{
    printf("==============================================\n");
    m_pnt1.Dump();
    m_pnt2.Dump();
    TGLVector3 a =  m_pnt1 - m_pnt2;
    printf("Distance:\n%f \n", a.Mag());
}

void FW3DViewDistanceMeasureTool::resetAction()
{
  // printf("Reset ACTION !!!! \n");
  m_action = kNone;
  m_bp1->SetState(kButtonUp);
  m_lp1->SetText( Form(lbpformat, m_pnt1.X(), m_pnt1.Y(), m_pnt1.Z()));

  m_bp2->SetState(kButtonUp);
  m_lp2->SetText( Form(lbpformat, m_pnt2.X(), m_pnt2.Y(), m_pnt2.Z()));

  TGLVector3 d = m_pnt2 - m_pnt1;
  m_ldist->SetText(Form("%.2f", d.Mag()));

  {
     TGCompositeFrame* p = (TGCompositeFrame*)(m_ldist->GetParent());
     p->Resize(p->GetDefaultSize());
  }
  {
  TGCompositeFrame* p = (TGCompositeFrame*)(m_ldist->GetParent()->GetParent());
  p->Layout();
  }
} 

void FW3DViewDistanceMeasureTool::setActionPnt1()
{
  // printf("Action ! 1111 \n");
   m_action = kPnt1;
   m_bp1->SetState(kButtonEngaged);
   m_bp2->SetState(kButtonUp);
}

void  FW3DViewDistanceMeasureTool::setActionPnt2()
{  
  // printf("Action ! 222 \n");
   m_action = kPnt2;
   m_bp2->SetState(kButtonEngaged);
   m_bp1->SetState(kButtonUp);
}

TGLVector3& FW3DViewDistanceMeasureTool::refCurrentVertex()
{
  //assert (m_action != kNone);
  if (m_action == kNone) {
    printf("ERROR refCurrentVertex m_action == kNone \n");
    return m_pnt1 ;
  }
   if (m_action == kPnt1)
      return m_pnt1;
   else
      return m_pnt2; 

}

TGCompositeFrame* FW3DViewDistanceMeasureTool::buildGUI(TGCompositeFrame* p)
{
   TGVerticalFrame* vf = new TGVerticalFrame(p);

   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(vf);
      TGLabel* lb = new TGLabel(hf, "Distance: ");
      hf->AddFrame(lb);
      m_ldist = new TGLabel(hf, " --- ");
      hf->AddFrame(m_ldist);
      vf->AddFrame(hf);
   }
   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(vf);

      m_bp1 = new TGTextButton(hf, "Pick Point1");
      m_bp1->Connect("Clicked()", "FW3DViewDistanceMeasureTool", this, "setActionPnt1()");
      m_bp1->SetToolTipText("Click on the butto to pick the first point in viewer.");
      hf->AddFrame( m_bp1, new TGLayoutHints(kLHintsNormal, 0, 5, 4, 4));

      m_lp1 = new TGLabel(hf, Form(lbpformat, m_pnt1.X(), m_pnt1.Y(), m_pnt1.Z()));
      hf->AddFrame(m_lp1, new TGLayoutHints(kLHintsNormal, 0, 1, 4, 4));

      vf->AddFrame(hf);
   }

   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(vf);
   
      m_bp2 = new TGTextButton(hf, "Pick Point2");
      m_bp2->Connect("Clicked()", "FW3DViewDistanceMeasureTool", this, "setActionPnt2()");
      m_bp2->SetToolTipText("Click on the butto to pick the secoond point in viewer.");
      hf->AddFrame( m_bp2, new TGLayoutHints(kLHintsExpandX, 0, 5, 4, 4));

      m_lp2 = new TGLabel(hf, Form(lbpformat, m_pnt2.X(), m_pnt2.Y(), m_pnt2.Z()));
      hf->AddFrame(m_lp2, new TGLayoutHints(kLHintsNormal, 0, 1, 4, 4));

      vf->AddFrame(hf);
   }

   {
      TGHorizontalFrame* hf = new TGHorizontalFrame(vf);
      TGTextButton* b = new TGTextButton(hf, "Print distance to terminal");
      b->Connect("Clicked()", "FW3DViewDistanceMeasureTool", this, "Print()");
      hf->AddFrame(b, new TGLayoutHints(kLHintsNormal, 0, 5, 4, 4));
      vf->AddFrame(hf);
   }
   return vf;

}
