#include "IOPool/TFileAdaptor/src/TFileAdaptorUI.h"
#include <iostream>

TFileAdaptorUI::TFileAdaptorUI (void)
{ 
  TFileAdaptorParams param;
  param.mode = "default";
  me.reset(new TFileAdaptor(param));
}

void
TFileAdaptorUI::stats (void) const
{
  me->stats(std::cout);
  std::cout << std::endl;
}

void
TFileAdaptorUI::statsXML (void) const
{
  me->statsXML(std::cout);
  std::cout << std::endl;
}
