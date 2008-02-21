#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/MonitorElementT.h"
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
template<class T> 
T* ExtractTObject<T>::extract( MonitorElement* me ) {
  if ( me ) {
    MonitorElementT<TNamed>* tnamed = dynamic_cast< MonitorElementT<TNamed>* >( me );
    if ( tnamed ) {
      T* histo = ExtractTObject::extract( tnamed->operator->() );
      if ( histo ) { 
	return histo; 
      } else { return 0; }
    } else { return 0; }
  } else { return 0; }
}

// -----------------------------------------------------------------------------  
//
template<class T> 
T* ExtractTObject<T>::extract( TNamed* tnamed ) {
  if ( tnamed ) {
    T* histo = dynamic_cast<T*>( tnamed );
    if ( histo ) { 
      return histo; 
    } else { return 0; }
  } else { return 0; }
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

