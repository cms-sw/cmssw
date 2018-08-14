#include "EventFilter/L1TRawToDigi/interface/OmtfLinkMappingCsc.h"

namespace omtf {

MapEleIndex2CscDet mapEleIndex2CscDet() {

  MapEleIndex2CscDet result;
  for (unsigned int fed=1380; fed<=1381; fed++) {
    //Endcap label. 1=forward (+Z); 2=backward (-Z)
    unsigned int endcap = (fed==1380) ? 2 : 1;
    for (unsigned int amc=1;    amc<=6; amc++) {
      for (unsigned int link=0; link <=34; link++) {
        unsigned int stat=0;
        unsigned int ring=0;
        unsigned int cham=0;
        switch (link) {
          case ( 0) : { stat=1; ring=2; cham=3; break;} //  (0,  9, 2, 3 ), --channel_0  OV1A_4 chamber_ME1/2/3  layer_9 input 2, 3
          case ( 1) : { stat=1; ring=2; cham=4; break;} //  (1,  9, 4, 5 ), --channel_1  OV1A_5 chamber_ME1/2/4  layer_9 input 4, 5
          case ( 2) : { stat=1; ring=2; cham=5; break;} //  (2,  9, 6, 7 ), --channel_2  OV1A_6 chamber_ME1/2/5  layer_9 input 6, 7
          case ( 3) : { stat=1; ring=3; cham=3; break;} //  (3,  6, 2, 3 ), --channel_3  OV1A_7 chamber_ME1/3/3  layer_6 input 2, 3
          case ( 4) : { stat=1; ring=3; cham=4; break;} //  (4,  6, 4, 5 ), --channel_4  OV1A_8 chamber_ME1/3/4  layer_6 input 4, 5
          case ( 5) : { stat=1; ring=3; cham=5; break;} //  (5,  6, 6, 7 ), --channel_5  OV1A_9 chamber_ME1/3/5  layer_6 input 6, 7
          case ( 6) : { stat=1; ring=2; cham=6; break;} //  (6,  9, 8, 9 ), --channel_6  OV1B_4 chamber_ME1/2/6  layer_9 input 8, 9
          case ( 7) : { stat=1; ring=2; cham=7; break;} //  (7,  9, 10,11), --channel_7  OV1B_5 chamber_ME1/2/7  layer_9 input 10,11
          case ( 8) : { stat=1; ring=2; cham=8; break;} //  (8,  9, 12,13), --channel_8  OV1B_6 chamber_ME1/2/8  layer_9 input 12,13
          case ( 9) : { stat=1; ring=3; cham=6; break;} //  (9,  6, 8, 9 ), --channel_9  OV1B_7 chamber_ME1/3/6  layer_6 input 8, 9
          case (10) : { stat=1; ring=3; cham=7; break;} //  (10, 6, 10,11), --channel_10 OV1B_8 chamber_ME1/3/7  layer_6 input 10,11
          case (11) : { stat=1; ring=3; cham=8; break;} //  (11, 6, 12,13), --channel_11 OV1B_9 chamber_ME1/3/8  layer_6 input 12,13
          case (12) : { stat=2; ring=2; cham=3; break;} //  (12, 7, 2, 3 ), --channel_0  OV2_4  chamber_ME2/2/3  layer_7 input 2, 3
          case (13) : { stat=2; ring=2; cham=4; break;} //  (13, 7, 4, 5 ), --channel_1  OV2_5  chamber_ME2/2/4  layer_7 input 4, 5
          case (14) : { stat=2; ring=2; cham=5; break;} //  (14, 7, 6, 7 ), --channel_2  OV2_6  chamber_ME2/2/5  layer_7 input 6, 7
          case (15) : { stat=2; ring=2; cham=6; break;} //  (15, 7, 8, 9 ), --channel_3  OV2_7  chamber_ME2/2/6  layer_7 input 8, 9
          case (16) : { stat=2; ring=2; cham=7; break;} //  (16, 7, 10,11), --channel_4  OV2_8  chamber_ME2/2/7  layer_7 input 10,11
          case (17) : { stat=2; ring=2; cham=8; break;} //  (17, 7, 12,13), --channel_5  OV2_9  chamber_ME2/2/8  layer_7 input 12,13
          case (18) : { stat=3; ring=2; cham=3; break;} //  (18, 8, 2, 3 ), --channel_6  OV3_4  chamber_ME3/2/3  layer_8 input 2, 3
          case (19) : { stat=3; ring=2; cham=4; break;} //  (19, 8, 4, 5 ), --channel_7  OV3_5  chamber_ME3/2/4  layer_8 input 4, 5
          case (20) : { stat=3; ring=2; cham=5; break;} //  (20, 8, 6, 7 ), --channel_8  OV3_6  chamber_ME3/2/5  layer_8 input 6, 7
          case (21) : { stat=3; ring=2; cham=6; break;} //  (21, 8, 8, 9 ), --channel_9  OV3_7  chamber_ME3/2/6  layer_8 input 8, 9
          case (22) : { stat=3; ring=2; cham=7; break;} //  (22, 8, 10,11), --channel_10 OV3_8  chamber_ME3/2/7  layer_8 input 10,11
          case (23) : { stat=3; ring=2; cham=8; break;} //  (23, 8, 12,13), --channel_11 OV3_9  chamber_ME3/2/8  layer_8 input 12,13
          case (24) : { stat=4; ring=2; cham=3; break;} //--(24,  ,      ), --channel_3  OV4_4  chamber_ME4/2/3  layer   input
          case (25) : { stat=4; ring=2; cham=4; break;} //--(25,  ,      ), --channel_4  OV4_5  chamber_ME4/2/4  layer   input
          case (26) : { stat=4; ring=2; cham=5; break;} //--(26,  ,      ), --channel_5  OV4_6  chamber_ME4/2/5  layer   input
          case (27) : { stat=4; ring=2; cham=6; break;} //--(27,  ,      ), --channel_7  OV4_7  chamber_ME4/2/6  layer   input
          case (28) : { stat=4; ring=2; cham=7; break;} //--(28,  ,      ), --channel_8  OV4_8  chamber_ME4/2/7  layer   input
          case (29) : { stat=4; ring=2; cham=8; break;} //--(29,  ,      ), --channel_9  OV4_9  chamber_ME4/2/8  layer   input
          case (30) : { stat=1; ring=2; cham=2; break;} //  (30, 9, 0, 1 ), --channel_0  OV1B_6 chamber_ME1/2/2  layer_9 input 0, 1
          case (31) : { stat=1; ring=3; cham=2; break;} //  (31, 6, 0, 1 ), --channel_1  OV1B_9 chamber_ME1/3/2  layer_6 input 0, 1
          case (32) : { stat=2; ring=2; cham=2; break;} //  (32, 7, 0, 1 ), --channel_2  OV2_9  chamber_ME2/2/2  layer_7 input 0, 1
          case (33) : { stat=3; ring=2; cham=2; break;} //  (33, 8, 0, 1 ), --channel_3  ON3_9  chamber_ME3/2/2  layer_8 input 0, 1
          case (34) : { stat=4; ring=2; cham=2; break;} //--(34,  ,      ), --channel_4  ON4_9  chamber_ME4/2/2  layer   input
          default   : { stat=0; ring=0; cham=0; break;}
        }
        if (ring !=0) {
          int chamber = cham+(amc-1)*6;
          if (chamber > 36) chamber -= 36;
          CSCDetId cscDetId(endcap, stat, ring, chamber,0);
          EleIndex omtfEle(fed, amc, link);
          result[omtfEle]=cscDetId;
        }
      }
    }
  }
  return result;
}

MapCscDet2EleIndex mapCscDet2EleIndex() {

  MapCscDet2EleIndex result;
  MapEleIndex2CscDet omtf2cscs = mapEleIndex2CscDet();

  for (const auto & omtf2csc : omtf2cscs) {
    uint32_t rawId = omtf2csc.second;
    auto it = result.find(rawId);
    if (result.end() == it) {
      result[rawId]=std::make_pair(omtf2csc.first,EleIndex());
    } else {
      it->second.second = omtf2csc.first;
    }
  }

  return result;
}

}
