#include <DataFormats/DTDigi/interface/DTDigi.h>
#include <DataFormats/DTDigi/interface/DTDigiCollection.h>
#include <DataFormats/DTDigi/interface/DTLocalTrigger.h>
#include <DataFormats/DTDigi/interface/DTLocalTriggerCollection.h>
#include <DataFormats/DTDigi/interface/DTControlData.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/Common/interface/Wrapper.h>
#include <vector>
#include <map>

namespace DataFormats_DTDigi {
  struct dictionary {

  DTDigi d;
  std::vector<DTDigi>  vv;
  std::vector<std::vector<DTDigi> >  v1; 
  DTDigiCollection dd;

  DTLocalTrigger t;
  std::vector<DTLocalTrigger>  ww;
  std::vector<std::vector<DTLocalTrigger> >  w1; 
  DTLocalTriggerCollection tt;
    
  DTDDUData u;
  std::vector<DTDDUData> vu;
  DTDDUFirstStatusWord fsw;
  std::vector<DTDDUFirstStatusWord> vfsw;
  DTDDUSecondStatusWord ssw;
  std::vector<DTDDUSecondStatusWord> vssw;
  
  DTROBHeaderWord rbhw;
  std::vector<DTROBHeaderWord> vrbhw;
  std::pair<int,DTROBHeaderWord> pirbhw;
  std::vector<std::pair<int,DTROBHeaderWord> > vpirbhw;
  
  DTROBTrailerWord rbtw;
  std::vector<DTROBTrailerWord> vrbtw;
  
  DTROS25Data r;
  std::vector<DTROS25Data> vr;
  std::vector<std::vector<DTROS25Data> > vvr;
  DTROSWordType rwt;
  std::vector<DTROSWordType> vrwt;
  DTROSHeaderWord rhw;
  std::vector<DTROSHeaderWord> vrhw;
  DTROSTrailerWord rtw;
  std::vector<DTROSTrailerWord> vrtw;
  DTROSErrorWord rew;
  std::vector<DTROSErrorWord> vrew;
  DTROSDebugWord rdw;
  std::vector<DTROSDebugWord> vrdw;
  
  DTTDCHeaderWord thw;
  std::vector<DTTDCHeaderWord> vthw;
  std::pair<int,DTTDCHeaderWord> pithw;
  std::vector<std::pair<int,DTTDCHeaderWord> > vpithw;
  
  DTTDCTrailerWord ttw;
  std::vector<DTTDCTrailerWord> vttw;
  std::pair<int,DTTDCTrailerWord> pittw;
  std::vector<std::pair<int,DTTDCTrailerWord> > vpittw;
  
  DTTDCMeasurementWord tmw;
  std::vector<DTTDCMeasurementWord> vtmw;
  std::pair<int,DTTDCMeasurementWord> pitmw;
  std::vector<std::pair<int,DTTDCMeasurementWord> > vpitmw;
  
  DTTDCErrorWord tew;
  std::vector<DTTDCErrorWord> vtew;
  std::pair<int,DTTDCErrorWord> pitew;
  std::vector<std::pair<int,DTTDCErrorWord> > vpitew;
  
  DTLocalTriggerHeaderWord lthw;
  std::vector<DTLocalTriggerHeaderWord> vlthw;
  std::pair<DTLocalTriggerHeaderWord,int> plthiw;
  std::vector<std::pair<DTLocalTriggerHeaderWord,int> > vplthiw;
  
  DTLocalTriggerTrailerWord lttw;
  std::vector<DTLocalTriggerTrailerWord> vlttw;
  std::pair<DTLocalTriggerTrailerWord,int> plttiw;
  std::vector<std::pair<DTLocalTriggerTrailerWord,int> > vplttiw;
  
  DTLocalTriggerDataWord ltdw;
  std::vector<DTLocalTriggerDataWord> vltdw;
  std::pair<DTLocalTriggerDataWord,int> pltdiw;
  std::vector<std::pair<DTLocalTriggerDataWord,int> > vpltdiw;
  
  DTLocalTriggerSectorCollectorHeaderWord ltschw;
  
  DTLocalTriggerSectorCollectorSubHeaderWord ltsctw;
  
  edm::Wrapper<DTDigiCollection> dw;
  edm::Wrapper<DTLocalTriggerCollection> tw;
  edm::Wrapper<std::vector<DTDDUData> > uw;
  edm::Wrapper< std::vector<std::vector<DTROS25Data> > > rw;
  
  };
}
