#include "EventFilter/HcalRawToDigi/interface/HcalTTPUnpacker.h"

bool HcalTTPUnpacker::unpack(const HcalHTRData& data, HcalTTPDigi& digi) {
  int id=data.getSubmodule();
  int samples=data.getNDD();
  int presamples=data.getNPS();
  int algo=data.getFirmwareFlavor()&0x1F;

  digi=HcalTTPDigi(id,samples,presamples,algo);

  if (samples==0) {
    const unsigned short* rd=data.getRawData();
    int len=data.getRawLength();
    for (int i=0; i<len; i++)
      printf("%3d %04X\n",i,rd[i]);
  }

  const unsigned short* daq_first, *daq_last, *tp_first, *tp_last;
  data.dataPointers(&daq_first,&daq_last,&tp_first,&tp_last);

  for (int i=0; i<samples; i++) {
    const uint16_t* work=(daq_first+6*i);
    if (work>daq_last) break;
    uint32_t algo=((*(work+4))>>8)&0xFF;
    algo|=((*(work+5))&0xFFF)<<8;    
    uint8_t trig=((*(work+5))>>12)&0xF;
    
    digi.setSample(i-presamples,work,algo,trig);
  }
  
  return true;
}
