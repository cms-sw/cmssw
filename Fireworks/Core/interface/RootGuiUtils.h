// @(#)root/eve:$Id: SKEL-base.h 23035 2008-04-08 09:17:02Z matevz $
// Author: Matevz Tadel 2011

#ifndef Fireworks_Core_RootGuiUtils
#define Fireworks_Core_RootGuiUtils

class TGCompositeFrame;
class TGHorizontalFrame;

class TGLabel;

namespace fireworks_root_gui
{

TGHorizontalFrame* makeHorizontalFrame(TGCompositeFrame* p=0);
TGLabel*           makeLabel(TGCompositeFrame* p, const char* txt, int width,
                             int lo=0, int ro=0, int to=2, int bo=0);

}

#endif
