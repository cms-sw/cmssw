#ifndef DQMOFFLINE_TRIGGER_EGHLTERRCODES
#define DQMOFFLINE_TRIGGER_EGHLTERRCODES

//defines our error codes

namespace egHLT{
  
  namespace errCodes {
    
    enum ErrCodes { 
      NoErr =0,
      TrigEvent=1,
      OffEle=2,
      OffPho=3,
      OffJet=4,
      Geom=5,
      EBRecHits=6,
      EERecHits=7,
      IsolTrks=8,
      HBHERecHits=9,
      HFRecHits=10,
      PhoID=11,
      EleEcalIsol=12,
      EleHcalD1Isol=13,
      EleHcalD2Isol=14,
      EleTrkIsol=15
    };
    
  }

}

#endif
