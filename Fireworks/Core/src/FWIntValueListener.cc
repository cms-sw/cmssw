#include "Fireworks/Core/interface/FWIntValueListener.h"

void FWIntValueListener::setValueImp(Int_t val)
{
   valueChanged_.emit(val);
}
