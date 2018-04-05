// Author: Matevz Tadel 2011

#ifndef Fireworks_Core_RootGuiUtils
#define Fireworks_Core_RootGuiUtils

class TGCompositeFrame;
class TGHorizontalFrame;

class TGLabel;

namespace fireworks_root_gui
{

TGHorizontalFrame* makeHorizontalFrame(TGCompositeFrame* p=nullptr);
TGLabel*           makeLabel(TGCompositeFrame* p, const char* txt, int width,
                             int lo=0, int ro=0, int to=2, int bo=0);

}

#endif
