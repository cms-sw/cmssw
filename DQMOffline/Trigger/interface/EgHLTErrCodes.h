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
      BeamSpot=11,
      MagField=12,
      CaloTowers=13,
      OffVertex=14
    };
    
  }

}

#endif
