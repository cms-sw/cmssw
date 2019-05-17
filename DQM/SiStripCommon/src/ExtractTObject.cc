#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "TNamed.h"
#include "TH1.h"
#include "TH1C.h"
#include "TH1S.h"
#include "TH1I.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TH2.h"
#include "TH2C.h"
#include "TH2S.h"
#include "TH2I.h"
#include "TH2F.h"
#include "TH2D.h"
#include "TProfile.h"

// -----------------------------------------------------------------------------
//
template <class T>
T* ExtractTObject<T>::extract(MonitorElement* me) {
  return me ? dynamic_cast<T*>(me->getRootObject()) : nullptr;
}
// -----------------------------------------------------------------------------
//
template class ExtractTObject<TH1>;
template class ExtractTObject<TH1C>;
template class ExtractTObject<TH1S>;
template class ExtractTObject<TH1I>;
template class ExtractTObject<TH1F>;
template class ExtractTObject<TH1D>;
template class ExtractTObject<TH2>;
template class ExtractTObject<TH2C>;
template class ExtractTObject<TH2S>;
template class ExtractTObject<TH2I>;
template class ExtractTObject<TH2F>;
template class ExtractTObject<TH2D>;
template class ExtractTObject<TProfile>;
