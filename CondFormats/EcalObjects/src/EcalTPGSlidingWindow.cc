#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"


EcalTPGSlidingWindow::EcalTPGSlidingWindow()
{ }

EcalTPGSlidingWindow::~EcalTPGSlidingWindow()
{ }

void EcalTPGSlidingWindow::setValue(const uint32_t & id, const uint32_t & value)
{
  map_[id] = value ;
}
