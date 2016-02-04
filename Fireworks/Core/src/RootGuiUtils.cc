// @(#)root/eve:$Id: RootGuiUtils.cc,v 1.1 2011/02/21 18:50:54 matevz Exp $
// Author: Matevz Tadel 2011

#include "Fireworks/Core/interface/RootGuiUtils.h"

#include "TGFrame.h"
#include "TGLabel.h"
#include "TGWidget.h"

namespace fireworks_root_gui
{

TGHorizontalFrame* makeHorizontalFrame(TGCompositeFrame* p)
{
   // Make standard horizontal frame.

   TGHorizontalFrame* f = new TGHorizontalFrame(p);
   p->AddFrame(f, new TGLayoutHints(kLHintsNormal|kLHintsExpandX));
   return f;
}

TGLabel* makeLabel(TGCompositeFrame* p, const char* txt, int width,
                   int lo, int ro, int to, int bo)
{
   // Make standard label.

   TGLabel *l = new TGLabel(p, txt);
   p->AddFrame(l, new TGLayoutHints(kLHintsNormal, lo,ro,to,bo));
   l->SetTextJustify(kTextRight);
   l->SetWidth(width);
   l->ChangeOptions(l->GetOptions() | kFixedWidth);
   return l;
}

}
