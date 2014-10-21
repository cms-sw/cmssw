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
    
  std::vector<DTDDUData> vu;
  std::vector<DTDDUFirstStatusWord> vfsw;
  std::vector<DTDDUSecondStatusWord> vssw;
  
  std::vector<DTROBHeaderWord> vrbhw;
  std::pair<int,DTROBHeaderWord> pirbhw;
  std::vector<std::pair<int,DTROBHeaderWord> > vpirbhw;
  
  std::vector<DTROBTrailerWord> vrbtw;
  
  std::vector<DTROS25Data> vr;
  std::vector<std::vector<DTROS25Data> > vvr;
  std::vector<DTROSWordType> vrwt;
  std::vector<DTROSHeaderWord> vrhw;
  std::vector<DTROSTrailerWord> vrtw;
  std::vector<DTROSErrorWord> vrew;
  std::vector<DTROSDebugWord> vrdw;
  
  std::vector<DTTDCHeaderWord> vthw;
  std::pair<int,DTTDCHeaderWord> pithw;
  std::vector<std::pair<int,DTTDCHeaderWord> > vpithw;
  
  std::vector<DTTDCTrailerWord> vttw;
  std::pair<int,DTTDCTrailerWord> pittw;
  std::vector<std::pair<int,DTTDCTrailerWord> > vpittw;
  
  std::vector<DTTDCMeasurementWord> vtmw;
  std::pair<int,DTTDCMeasurementWord> pitmw;
  std::vector<std::pair<int,DTTDCMeasurementWord> > vpitmw;
  
  std::vector<DTTDCErrorWord> vtew;
  std::pair<int,DTTDCErrorWord> pitew;
  std::vector<std::pair<int,DTTDCErrorWord> > vpitew;
  
  std::vector<DTLocalTriggerHeaderWord> vlthw;
  std::pair<DTLocalTriggerHeaderWord,int> plthiw;
  std::vector<std::pair<DTLocalTriggerHeaderWord,int> > vplthiw;
  
  std::vector<DTLocalTriggerTrailerWord> vlttw;
  std::pair<DTLocalTriggerTrailerWord,int> plttiw;
  std::vector<std::pair<DTLocalTriggerTrailerWord,int> > vplttiw;
  
  std::vector<DTLocalTriggerDataWord> vltdw;
  std::pair<DTLocalTriggerDataWord,int> pltdiw;
  std::vector<std::pair<DTLocalTriggerDataWord,int> > vpltdiw;
  
  edm::Wrapper<DTDigiCollection> dw;
  edm::Wrapper<DTLocalTriggerCollection> tw;
  edm::Wrapper<std::vector<DTDDUData> > uw;
  edm::Wrapper< std::vector<std::vector<DTROS25Data> > > rw;
  
  };
}
