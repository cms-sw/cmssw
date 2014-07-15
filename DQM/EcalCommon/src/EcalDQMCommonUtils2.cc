#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEBGeom.h"
#include "CalibCalorimetry/EcalLaserAnalyzer/interface/MEEEGeom.h"

namespace ecaldqm
{
  EcalPnDiodeDetId
  pnForCrystal(DetId const& _id, char _ab)
  {
    bool pnA(_ab == 'a' || _ab == 'A');

    if(!isCrystalId(_id)) return EcalPnDiodeDetId(0);

    if(_id.subdetId() == EcalBarrel){
      EBDetId ebid(_id);
      int lmmod(MEEBGeom::lmmod(ebid.ieta(), ebid.iphi()));

      switch(dccId(_id)){
      case 10:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 1) : EcalPnDiodeDetId(EcalBarrel, 10, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 2) : EcalPnDiodeDetId(EcalBarrel, 10, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 2) : EcalPnDiodeDetId(EcalBarrel, 10, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 3) : EcalPnDiodeDetId(EcalBarrel, 10, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 3) : EcalPnDiodeDetId(EcalBarrel, 10, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 4) : EcalPnDiodeDetId(EcalBarrel, 10, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 4) : EcalPnDiodeDetId(EcalBarrel, 10, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 5) : EcalPnDiodeDetId(EcalBarrel, 10, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 10, 5) : EcalPnDiodeDetId(EcalBarrel, 10, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 11:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 1) : EcalPnDiodeDetId(EcalBarrel, 11, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 2) : EcalPnDiodeDetId(EcalBarrel, 11, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 2) : EcalPnDiodeDetId(EcalBarrel, 11, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 3) : EcalPnDiodeDetId(EcalBarrel, 11, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 3) : EcalPnDiodeDetId(EcalBarrel, 11, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 4) : EcalPnDiodeDetId(EcalBarrel, 11, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 4) : EcalPnDiodeDetId(EcalBarrel, 11, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 5) : EcalPnDiodeDetId(EcalBarrel, 11, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 11, 5) : EcalPnDiodeDetId(EcalBarrel, 11, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 12:
        switch(lmmod){
        case 1: return /*pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 1) :*/ EcalPnDiodeDetId(EcalBarrel, 12, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 2) : EcalPnDiodeDetId(EcalBarrel, 12, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 2) : EcalPnDiodeDetId(EcalBarrel, 12, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 3) : EcalPnDiodeDetId(EcalBarrel, 12, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 3) : EcalPnDiodeDetId(EcalBarrel, 12, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 4) : EcalPnDiodeDetId(EcalBarrel, 12, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 4) : EcalPnDiodeDetId(EcalBarrel, 12, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 5) : EcalPnDiodeDetId(EcalBarrel, 12, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 12, 5) : EcalPnDiodeDetId(EcalBarrel, 12, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 13:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 1) : EcalPnDiodeDetId(EcalBarrel, 13, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 2) : EcalPnDiodeDetId(EcalBarrel, 13, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 2) : EcalPnDiodeDetId(EcalBarrel, 13, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 3) : EcalPnDiodeDetId(EcalBarrel, 13, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 3) : EcalPnDiodeDetId(EcalBarrel, 13, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 4) : EcalPnDiodeDetId(EcalBarrel, 13, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 4) : EcalPnDiodeDetId(EcalBarrel, 13, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 5) : EcalPnDiodeDetId(EcalBarrel, 13, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 13, 5) : EcalPnDiodeDetId(EcalBarrel, 13, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 14:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 1) : EcalPnDiodeDetId(EcalBarrel, 14, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 2) : EcalPnDiodeDetId(EcalBarrel, 14, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 2) : EcalPnDiodeDetId(EcalBarrel, 14, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 3) : EcalPnDiodeDetId(EcalBarrel, 14, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 3) : EcalPnDiodeDetId(EcalBarrel, 14, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 4) : EcalPnDiodeDetId(EcalBarrel, 14, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 4) : EcalPnDiodeDetId(EcalBarrel, 14, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 5) : EcalPnDiodeDetId(EcalBarrel, 14, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 14, 5) : EcalPnDiodeDetId(EcalBarrel, 14, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 15:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 1) : EcalPnDiodeDetId(EcalBarrel, 15, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 2) : EcalPnDiodeDetId(EcalBarrel, 15, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 2) : EcalPnDiodeDetId(EcalBarrel, 15, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 3) : EcalPnDiodeDetId(EcalBarrel, 15, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 3) : EcalPnDiodeDetId(EcalBarrel, 15, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 4) : EcalPnDiodeDetId(EcalBarrel, 15, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 4) : EcalPnDiodeDetId(EcalBarrel, 15, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 5) : EcalPnDiodeDetId(EcalBarrel, 15, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 15, 5) : EcalPnDiodeDetId(EcalBarrel, 15, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 16:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 1) : EcalPnDiodeDetId(EcalBarrel, 16, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 2) : EcalPnDiodeDetId(EcalBarrel, 16, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 2) : EcalPnDiodeDetId(EcalBarrel, 16, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 3) : EcalPnDiodeDetId(EcalBarrel, 16, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 3) : EcalPnDiodeDetId(EcalBarrel, 16, 8);
        case 6: return pnA ? /*EcalPnDiodeDetId(EcalBarrel, 16, 4)*/ EcalPnDiodeDetId(EcalBarrel, 16, 1) : EcalPnDiodeDetId(EcalBarrel, 16, 9);
        case 7: return /*pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 4) :*/ EcalPnDiodeDetId(EcalBarrel, 16, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 5) : EcalPnDiodeDetId(EcalBarrel, 16, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 16, 5) : EcalPnDiodeDetId(EcalBarrel, 16, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 17:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 1) : EcalPnDiodeDetId(EcalBarrel, 17, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 2) : EcalPnDiodeDetId(EcalBarrel, 17, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 2) : EcalPnDiodeDetId(EcalBarrel, 17, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 3) : EcalPnDiodeDetId(EcalBarrel, 17, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 3) : EcalPnDiodeDetId(EcalBarrel, 17, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 4) : EcalPnDiodeDetId(EcalBarrel, 17, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 4) : EcalPnDiodeDetId(EcalBarrel, 17, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 5) : EcalPnDiodeDetId(EcalBarrel, 17, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 17, 5) : EcalPnDiodeDetId(EcalBarrel, 17, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 18:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 1) : EcalPnDiodeDetId(EcalBarrel, 18, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 2) : EcalPnDiodeDetId(EcalBarrel, 18, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 2) : EcalPnDiodeDetId(EcalBarrel, 18, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 3) : EcalPnDiodeDetId(EcalBarrel, 18, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 3) : EcalPnDiodeDetId(EcalBarrel, 18, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 4) : EcalPnDiodeDetId(EcalBarrel, 18, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 4) : EcalPnDiodeDetId(EcalBarrel, 18, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 5) : EcalPnDiodeDetId(EcalBarrel, 18, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 18, 5) : EcalPnDiodeDetId(EcalBarrel, 18, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 19:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 1) : EcalPnDiodeDetId(EcalBarrel, 19, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 2) : EcalPnDiodeDetId(EcalBarrel, 19, 8);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 2) : EcalPnDiodeDetId(EcalBarrel, 19, 8);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 3) : EcalPnDiodeDetId(EcalBarrel, 19, 7);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 3) : EcalPnDiodeDetId(EcalBarrel, 19, 7);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 4) : EcalPnDiodeDetId(EcalBarrel, 19, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 4) : EcalPnDiodeDetId(EcalBarrel, 19, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 5) : EcalPnDiodeDetId(EcalBarrel, 19, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 19, 5) : EcalPnDiodeDetId(EcalBarrel, 19, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 20:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 1) : EcalPnDiodeDetId(EcalBarrel, 20, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 2) : EcalPnDiodeDetId(EcalBarrel, 20, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 2) : EcalPnDiodeDetId(EcalBarrel, 20, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 3) : EcalPnDiodeDetId(EcalBarrel, 20, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 3) : EcalPnDiodeDetId(EcalBarrel, 20, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 4) : EcalPnDiodeDetId(EcalBarrel, 20, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 4) : EcalPnDiodeDetId(EcalBarrel, 20, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 5) : EcalPnDiodeDetId(EcalBarrel, 20, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 20, 5) : EcalPnDiodeDetId(EcalBarrel, 20, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 21:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 1) : EcalPnDiodeDetId(EcalBarrel, 21, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 2) : EcalPnDiodeDetId(EcalBarrel, 21, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 2) : EcalPnDiodeDetId(EcalBarrel, 21, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 3) : EcalPnDiodeDetId(EcalBarrel, 21, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 3) : EcalPnDiodeDetId(EcalBarrel, 21, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 4) : EcalPnDiodeDetId(EcalBarrel, 21, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 4) : EcalPnDiodeDetId(EcalBarrel, 21, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 5) : EcalPnDiodeDetId(EcalBarrel, 21, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 21, 5) : EcalPnDiodeDetId(EcalBarrel, 21, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 22:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 1) : EcalPnDiodeDetId(EcalBarrel, 22, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 2) : EcalPnDiodeDetId(EcalBarrel, 22, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 2) : EcalPnDiodeDetId(EcalBarrel, 22, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 3) : EcalPnDiodeDetId(EcalBarrel, 22, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 3) : EcalPnDiodeDetId(EcalBarrel, 22, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 4) : EcalPnDiodeDetId(EcalBarrel, 22, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 4) : EcalPnDiodeDetId(EcalBarrel, 22, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 5) : EcalPnDiodeDetId(EcalBarrel, 22, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 22, 5) : EcalPnDiodeDetId(EcalBarrel, 22, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 23:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 1) : EcalPnDiodeDetId(EcalBarrel, 23, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 2) : EcalPnDiodeDetId(EcalBarrel, 23, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 2) : EcalPnDiodeDetId(EcalBarrel, 23, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 3) : EcalPnDiodeDetId(EcalBarrel, 23, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 3) : EcalPnDiodeDetId(EcalBarrel, 23, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 4) : EcalPnDiodeDetId(EcalBarrel, 23, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 4) : EcalPnDiodeDetId(EcalBarrel, 23, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 5) : EcalPnDiodeDetId(EcalBarrel, 23, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 23, 5) : EcalPnDiodeDetId(EcalBarrel, 23, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 24:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 1) : EcalPnDiodeDetId(EcalBarrel, 24, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 2) : EcalPnDiodeDetId(EcalBarrel, 24, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 2) : EcalPnDiodeDetId(EcalBarrel, 24, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 3) : EcalPnDiodeDetId(EcalBarrel, 24, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 3) : EcalPnDiodeDetId(EcalBarrel, 24, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 4) : EcalPnDiodeDetId(EcalBarrel, 24, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 4) : EcalPnDiodeDetId(EcalBarrel, 24, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 5) : EcalPnDiodeDetId(EcalBarrel, 24, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 24, 5) : EcalPnDiodeDetId(EcalBarrel, 24, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 25:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 2) : EcalPnDiodeDetId(EcalBarrel, 25, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 1) : EcalPnDiodeDetId(EcalBarrel, 25, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 1) : EcalPnDiodeDetId(EcalBarrel, 25, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 3) : EcalPnDiodeDetId(EcalBarrel, 25, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 3) : EcalPnDiodeDetId(EcalBarrel, 25, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 4) : EcalPnDiodeDetId(EcalBarrel, 25, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 4) : EcalPnDiodeDetId(EcalBarrel, 25, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 5) : EcalPnDiodeDetId(EcalBarrel, 25, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 25, 5) : EcalPnDiodeDetId(EcalBarrel, 25, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 26:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 1) : EcalPnDiodeDetId(EcalBarrel, 26, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 2) : EcalPnDiodeDetId(EcalBarrel, 26, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 2) : EcalPnDiodeDetId(EcalBarrel, 26, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 3) : EcalPnDiodeDetId(EcalBarrel, 26, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 3) : EcalPnDiodeDetId(EcalBarrel, 26, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 4) : EcalPnDiodeDetId(EcalBarrel, 26, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 4) : EcalPnDiodeDetId(EcalBarrel, 26, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 5) : EcalPnDiodeDetId(EcalBarrel, 26, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 26, 5) : EcalPnDiodeDetId(EcalBarrel, 26, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 27:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 1) : EcalPnDiodeDetId(EcalBarrel, 27, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 2) : EcalPnDiodeDetId(EcalBarrel, 27, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 2) : EcalPnDiodeDetId(EcalBarrel, 27, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 3) : EcalPnDiodeDetId(EcalBarrel, 27, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 3) : EcalPnDiodeDetId(EcalBarrel, 27, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 4) : EcalPnDiodeDetId(EcalBarrel, 27, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 4) : EcalPnDiodeDetId(EcalBarrel, 27, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 5) : EcalPnDiodeDetId(EcalBarrel, 27, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 27, 5) : EcalPnDiodeDetId(EcalBarrel, 27, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 28:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 1) : EcalPnDiodeDetId(EcalBarrel, 28, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 2) : EcalPnDiodeDetId(EcalBarrel, 28, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 2) : EcalPnDiodeDetId(EcalBarrel, 28, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 3) : EcalPnDiodeDetId(EcalBarrel, 28, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 3) : EcalPnDiodeDetId(EcalBarrel, 28, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 4) : EcalPnDiodeDetId(EcalBarrel, 28, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 4) : EcalPnDiodeDetId(EcalBarrel, 28, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 5) : EcalPnDiodeDetId(EcalBarrel, 28, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 28, 5) : EcalPnDiodeDetId(EcalBarrel, 28, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 29:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 1) : EcalPnDiodeDetId(EcalBarrel, 29, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 2) : EcalPnDiodeDetId(EcalBarrel, 29, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 2) : EcalPnDiodeDetId(EcalBarrel, 29, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 3) : EcalPnDiodeDetId(EcalBarrel, 29, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 3) : EcalPnDiodeDetId(EcalBarrel, 29, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 4) : EcalPnDiodeDetId(EcalBarrel, 29, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 4) : EcalPnDiodeDetId(EcalBarrel, 29, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 5) : EcalPnDiodeDetId(EcalBarrel, 29, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 29, 5) : EcalPnDiodeDetId(EcalBarrel, 29, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 30:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 1) : EcalPnDiodeDetId(EcalBarrel, 30, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 2) : EcalPnDiodeDetId(EcalBarrel, 30, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 2) : EcalPnDiodeDetId(EcalBarrel, 30, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 3) : EcalPnDiodeDetId(EcalBarrel, 30, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 3) : EcalPnDiodeDetId(EcalBarrel, 30, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 4) : EcalPnDiodeDetId(EcalBarrel, 30, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 4) : EcalPnDiodeDetId(EcalBarrel, 30, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 5) : EcalPnDiodeDetId(EcalBarrel, 30, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 30, 5) : EcalPnDiodeDetId(EcalBarrel, 30, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 31:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 1) : EcalPnDiodeDetId(EcalBarrel, 31, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 2) : EcalPnDiodeDetId(EcalBarrel, 31, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 2) : EcalPnDiodeDetId(EcalBarrel, 31, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 3) : EcalPnDiodeDetId(EcalBarrel, 31, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 3) : EcalPnDiodeDetId(EcalBarrel, 31, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 4) : EcalPnDiodeDetId(EcalBarrel, 31, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 4) : EcalPnDiodeDetId(EcalBarrel, 31, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 5) : EcalPnDiodeDetId(EcalBarrel, 31, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 31, 5) : EcalPnDiodeDetId(EcalBarrel, 31, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 32:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 1) : EcalPnDiodeDetId(EcalBarrel, 32, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 2) : EcalPnDiodeDetId(EcalBarrel, 32, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 2) : EcalPnDiodeDetId(EcalBarrel, 32, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 3) : EcalPnDiodeDetId(EcalBarrel, 32, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 3) : EcalPnDiodeDetId(EcalBarrel, 32, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 5) : EcalPnDiodeDetId(EcalBarrel, 32, 10);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 5) : EcalPnDiodeDetId(EcalBarrel, 32, 10);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 4) : EcalPnDiodeDetId(EcalBarrel, 32, 9);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 32, 4) : EcalPnDiodeDetId(EcalBarrel, 32, 9);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 33:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 1) : EcalPnDiodeDetId(EcalBarrel, 33, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 2) : EcalPnDiodeDetId(EcalBarrel, 33, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 2) : EcalPnDiodeDetId(EcalBarrel, 33, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 3) : EcalPnDiodeDetId(EcalBarrel, 33, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 3) : EcalPnDiodeDetId(EcalBarrel, 33, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 4) : EcalPnDiodeDetId(EcalBarrel, 33, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 4) : EcalPnDiodeDetId(EcalBarrel, 33, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 5) : EcalPnDiodeDetId(EcalBarrel, 33, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 33, 5) : EcalPnDiodeDetId(EcalBarrel, 33, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 34:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 1) : EcalPnDiodeDetId(EcalBarrel, 34, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 2) : EcalPnDiodeDetId(EcalBarrel, 34, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 2) : EcalPnDiodeDetId(EcalBarrel, 34, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 3) : EcalPnDiodeDetId(EcalBarrel, 34, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 3) : EcalPnDiodeDetId(EcalBarrel, 34, 8);
        case 6: return pnA ? /*EcalPnDiodeDetId(EcalBarrel, 34, 4)*/ EcalPnDiodeDetId(EcalBarrel, 34, 1) : EcalPnDiodeDetId(EcalBarrel, 34, 9);
        case 7: return /*pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 4) :*/ EcalPnDiodeDetId(EcalBarrel, 34, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 5) : EcalPnDiodeDetId(EcalBarrel, 34, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 34, 5) : EcalPnDiodeDetId(EcalBarrel, 34, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 35:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 1) : EcalPnDiodeDetId(EcalBarrel, 35, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 2) : EcalPnDiodeDetId(EcalBarrel, 35, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 2) : EcalPnDiodeDetId(EcalBarrel, 35, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 3) : EcalPnDiodeDetId(EcalBarrel, 35, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 3) : EcalPnDiodeDetId(EcalBarrel, 35, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 4) : EcalPnDiodeDetId(EcalBarrel, 35, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 4) : EcalPnDiodeDetId(EcalBarrel, 35, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 5) : EcalPnDiodeDetId(EcalBarrel, 35, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 35, 5) : EcalPnDiodeDetId(EcalBarrel, 35, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 36:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 1) : EcalPnDiodeDetId(EcalBarrel, 36, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 2) : EcalPnDiodeDetId(EcalBarrel, 36, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 2) : EcalPnDiodeDetId(EcalBarrel, 36, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 3) : EcalPnDiodeDetId(EcalBarrel, 36, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 3) : EcalPnDiodeDetId(EcalBarrel, 36, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 4) : EcalPnDiodeDetId(EcalBarrel, 36, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 4) : EcalPnDiodeDetId(EcalBarrel, 36, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 5) : EcalPnDiodeDetId(EcalBarrel, 36, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 36, 5) : EcalPnDiodeDetId(EcalBarrel, 36, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 37:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 1) : EcalPnDiodeDetId(EcalBarrel, 37, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 2) : EcalPnDiodeDetId(EcalBarrel, 37, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 2) : EcalPnDiodeDetId(EcalBarrel, 37, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 3) : EcalPnDiodeDetId(EcalBarrel, 37, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 3) : EcalPnDiodeDetId(EcalBarrel, 37, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 4) : EcalPnDiodeDetId(EcalBarrel, 37, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 4) : EcalPnDiodeDetId(EcalBarrel, 37, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 5) : EcalPnDiodeDetId(EcalBarrel, 37, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 37, 5) : EcalPnDiodeDetId(EcalBarrel, 37, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 38:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 1) : EcalPnDiodeDetId(EcalBarrel, 38, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 2) : EcalPnDiodeDetId(EcalBarrel, 38, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 2) : EcalPnDiodeDetId(EcalBarrel, 38, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 3) : EcalPnDiodeDetId(EcalBarrel, 38, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 3) : EcalPnDiodeDetId(EcalBarrel, 38, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 4) : EcalPnDiodeDetId(EcalBarrel, 38, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 4) : EcalPnDiodeDetId(EcalBarrel, 38, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 5) : EcalPnDiodeDetId(EcalBarrel, 38, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 38, 5) : EcalPnDiodeDetId(EcalBarrel, 38, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 39:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 1) : EcalPnDiodeDetId(EcalBarrel, 39, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 2) : EcalPnDiodeDetId(EcalBarrel, 39, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 2) : EcalPnDiodeDetId(EcalBarrel, 39, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 3) : EcalPnDiodeDetId(EcalBarrel, 39, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 3) : EcalPnDiodeDetId(EcalBarrel, 39, 8);
        case 6: return pnA ? /*EcalPnDiodeDetId(EcalBarrel, 39, 4)*/ EcalPnDiodeDetId(EcalBarrel, 39, 1) : EcalPnDiodeDetId(EcalBarrel, 39, 9);
        case 7: return /*pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 4) :*/ EcalPnDiodeDetId(EcalBarrel, 39, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 5) : EcalPnDiodeDetId(EcalBarrel, 39, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 39, 5) : EcalPnDiodeDetId(EcalBarrel, 39, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 40:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 1) : EcalPnDiodeDetId(EcalBarrel, 40, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 2) : EcalPnDiodeDetId(EcalBarrel, 40, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 2) : EcalPnDiodeDetId(EcalBarrel, 40, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 3) : EcalPnDiodeDetId(EcalBarrel, 40, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 3) : EcalPnDiodeDetId(EcalBarrel, 40, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 4) : EcalPnDiodeDetId(EcalBarrel, 40, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 4) : EcalPnDiodeDetId(EcalBarrel, 40, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 5) : EcalPnDiodeDetId(EcalBarrel, 40, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 40, 5) : EcalPnDiodeDetId(EcalBarrel, 40, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 41:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 1) : EcalPnDiodeDetId(EcalBarrel, 41, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 2) : EcalPnDiodeDetId(EcalBarrel, 41, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 2) : EcalPnDiodeDetId(EcalBarrel, 41, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 3) : EcalPnDiodeDetId(EcalBarrel, 41, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 3) : EcalPnDiodeDetId(EcalBarrel, 41, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 4) : EcalPnDiodeDetId(EcalBarrel, 41, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 4) : EcalPnDiodeDetId(EcalBarrel, 41, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 5) : EcalPnDiodeDetId(EcalBarrel, 41, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 41, 5) : EcalPnDiodeDetId(EcalBarrel, 41, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 42:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 1) : EcalPnDiodeDetId(EcalBarrel, 42, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 2) : EcalPnDiodeDetId(EcalBarrel, 42, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 2) : EcalPnDiodeDetId(EcalBarrel, 42, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 3) : EcalPnDiodeDetId(EcalBarrel, 42, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 3) : EcalPnDiodeDetId(EcalBarrel, 42, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 4) : EcalPnDiodeDetId(EcalBarrel, 42, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 4) : EcalPnDiodeDetId(EcalBarrel, 42, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 5) : EcalPnDiodeDetId(EcalBarrel, 42, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 42, 5) : EcalPnDiodeDetId(EcalBarrel, 42, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 43:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 1) : EcalPnDiodeDetId(EcalBarrel, 43, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 2) : EcalPnDiodeDetId(EcalBarrel, 43, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 2) : EcalPnDiodeDetId(EcalBarrel, 43, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 3) : EcalPnDiodeDetId(EcalBarrel, 43, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 3) : EcalPnDiodeDetId(EcalBarrel, 43, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 4) : EcalPnDiodeDetId(EcalBarrel, 43, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 4) : EcalPnDiodeDetId(EcalBarrel, 43, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 5) : EcalPnDiodeDetId(EcalBarrel, 43, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 43, 5) : EcalPnDiodeDetId(EcalBarrel, 43, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 44:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 1) : EcalPnDiodeDetId(EcalBarrel, 44, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 2) : EcalPnDiodeDetId(EcalBarrel, 44, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 2) : EcalPnDiodeDetId(EcalBarrel, 44, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 3) : EcalPnDiodeDetId(EcalBarrel, 44, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 3) : EcalPnDiodeDetId(EcalBarrel, 44, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 4) : EcalPnDiodeDetId(EcalBarrel, 44, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 4) : EcalPnDiodeDetId(EcalBarrel, 44, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 5) : EcalPnDiodeDetId(EcalBarrel, 44, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 44, 5) : EcalPnDiodeDetId(EcalBarrel, 44, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      case 45:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 1) : EcalPnDiodeDetId(EcalBarrel, 45, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 2) : EcalPnDiodeDetId(EcalBarrel, 45, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 2) : EcalPnDiodeDetId(EcalBarrel, 45, 7);
        case 4: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 3) : EcalPnDiodeDetId(EcalBarrel, 45, 8);
        case 5: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 3) : EcalPnDiodeDetId(EcalBarrel, 45, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 4) : EcalPnDiodeDetId(EcalBarrel, 45, 9);
        case 7: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 4) : EcalPnDiodeDetId(EcalBarrel, 45, 9);
        case 8: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 5) : EcalPnDiodeDetId(EcalBarrel, 45, 10);
        case 9: return pnA ? EcalPnDiodeDetId(EcalBarrel, 45, 5) : EcalPnDiodeDetId(EcalBarrel, 45, 10);
        default: return EcalPnDiodeDetId(0);
        }
        break;
      default: return EcalPnDiodeDetId(0);
      }

    }
    else{
      EcalScDetId scid(EEDetId(_id).sc());
      int ix(scid.ix());
      int iy(scid.iy());
      int dee(MEEEGeom::dee(ix, iy, scid.zside()));
      int lmmod(MEEEGeom::lmmod(ix, iy));

      switch(dee){
      case 1:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 1) : EcalPnDiodeDetId(EcalEndcap, 50, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 2) : EcalPnDiodeDetId(EcalEndcap, 50, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 3) : EcalPnDiodeDetId(EcalEndcap, 50, 8);
        case 4: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 4) : EcalPnDiodeDetId(EcalEndcap, 50, 9);
        case 5: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 3) : EcalPnDiodeDetId(EcalEndcap, 50, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 1) : EcalPnDiodeDetId(EcalEndcap, 50, 6);
        case 7: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 2) : EcalPnDiodeDetId(EcalEndcap, 50, 7);
        case 8: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 4) : EcalPnDiodeDetId(EcalEndcap, 50, 9);
        case 9: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 6) : EcalPnDiodeDetId(EcalEndcap, 50, 1);
        case 10: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 7) : EcalPnDiodeDetId(EcalEndcap, 50, 2);
        case 11: return /*pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 8) :*/ EcalPnDiodeDetId(EcalEndcap, 50, 3);
        case 12: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 9) : EcalPnDiodeDetId(EcalEndcap, 50, 4);
        case 13: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 10) : EcalPnDiodeDetId(EcalEndcap, 50, 5);
        case 14: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 8) : EcalPnDiodeDetId(EcalEndcap, 50, 3);
        case 15: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 6) : EcalPnDiodeDetId(EcalEndcap, 50, 1);
        case 16: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 7) : EcalPnDiodeDetId(EcalEndcap, 50, 2);
        case 17: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 10) : EcalPnDiodeDetId(EcalEndcap, 50, 5);
        case 18: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 5) : EcalPnDiodeDetId(EcalEndcap, 50, 10);
        case 19: return pnA ? EcalPnDiodeDetId(EcalEndcap, 51, 9) : EcalPnDiodeDetId(EcalEndcap, 50, 4);
        default: return EcalPnDiodeDetId(0);
        }
      case 2:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 1) : EcalPnDiodeDetId(EcalEndcap, 46, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 2) : EcalPnDiodeDetId(EcalEndcap, 46, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 3) : EcalPnDiodeDetId(EcalEndcap, 46, 8);
        case 4: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 4) : EcalPnDiodeDetId(EcalEndcap, 46, 9);
        case 5: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 3) : EcalPnDiodeDetId(EcalEndcap, 46, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 1) : EcalPnDiodeDetId(EcalEndcap, 46, 6);
        case 7: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 2) : EcalPnDiodeDetId(EcalEndcap, 46, 7);
        case 8: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 4) : EcalPnDiodeDetId(EcalEndcap, 46, 9);
        case 9: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 6) : EcalPnDiodeDetId(EcalEndcap, 46, 1);
        case 10: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 7) : EcalPnDiodeDetId(EcalEndcap, 46, 2);
        case 11: return /*pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 8) :*/ EcalPnDiodeDetId(EcalEndcap, 46, 3);
        case 12: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 9) : EcalPnDiodeDetId(EcalEndcap, 46, 4);
        case 13: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 10) : EcalPnDiodeDetId(EcalEndcap, 46, 5);
        case 14: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 8) : EcalPnDiodeDetId(EcalEndcap, 46, 3);
        case 15: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 6) : EcalPnDiodeDetId(EcalEndcap, 46, 1);
        case 16: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 7) : EcalPnDiodeDetId(EcalEndcap, 46, 2);
        case 17: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 10) : EcalPnDiodeDetId(EcalEndcap, 46, 5);
        case 18: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 5) : EcalPnDiodeDetId(EcalEndcap, 46, 10);
        case 19: return pnA ? EcalPnDiodeDetId(EcalEndcap, 47, 9) : EcalPnDiodeDetId(EcalEndcap, 46, 4);
        default: return EcalPnDiodeDetId(0);
        }
      case 3:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 1) : EcalPnDiodeDetId(EcalEndcap, 1, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 2) : EcalPnDiodeDetId(EcalEndcap, 1, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 3) : EcalPnDiodeDetId(EcalEndcap, 1, 8);
        case 4: return /*pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 4) :*/ EcalPnDiodeDetId(EcalEndcap, 1, 9);
        case 5: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 3) : EcalPnDiodeDetId(EcalEndcap, 1, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 1) : EcalPnDiodeDetId(EcalEndcap, 1, 6);
        case 7: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 2) : EcalPnDiodeDetId(EcalEndcap, 1, 7);
        case 8: return /*pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 4) :*/ EcalPnDiodeDetId(EcalEndcap, 1, 9);
        case 9: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 6) : EcalPnDiodeDetId(EcalEndcap, 1, 1);
        case 10: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 7) : EcalPnDiodeDetId(EcalEndcap, 1, 2);
        case 11: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 8) : EcalPnDiodeDetId(EcalEndcap, 1, 3);
        case 12: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 9) : EcalPnDiodeDetId(EcalEndcap, 1, 4);
        case 13: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 10) : EcalPnDiodeDetId(EcalEndcap, 1, 5);
        case 14: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 8) : EcalPnDiodeDetId(EcalEndcap, 1, 3);
        case 15: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 6) : EcalPnDiodeDetId(EcalEndcap, 1, 1);
        case 16: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 7) : EcalPnDiodeDetId(EcalEndcap, 1, 2);
        case 17: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 10) : EcalPnDiodeDetId(EcalEndcap, 1, 5);
        case 18: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 5) : EcalPnDiodeDetId(EcalEndcap, 1, 10);
        case 19: return pnA ? EcalPnDiodeDetId(EcalEndcap, 2, 9) : EcalPnDiodeDetId(EcalEndcap, 1, 4);
        default: return EcalPnDiodeDetId(0);
        }
      case 4:
        switch(lmmod){
        case 1: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 1) : EcalPnDiodeDetId(EcalEndcap, 5, 6);
        case 2: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 2) : EcalPnDiodeDetId(EcalEndcap, 5, 7);
        case 3: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 3) : EcalPnDiodeDetId(EcalEndcap, 5, 8);
        case 4: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 4) : EcalPnDiodeDetId(EcalEndcap, 5, 9);
        case 5: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 3) : EcalPnDiodeDetId(EcalEndcap, 5, 8);
        case 6: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 1) : EcalPnDiodeDetId(EcalEndcap, 5, 6);
        case 7: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 2) : EcalPnDiodeDetId(EcalEndcap, 5, 7);
        case 8: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 4) : EcalPnDiodeDetId(EcalEndcap, 5, 9);
        case 9: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 6) : EcalPnDiodeDetId(EcalEndcap, 5, 1);
        case 10: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 7) : EcalPnDiodeDetId(EcalEndcap, 5, 2);
        case 11: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 8) : EcalPnDiodeDetId(EcalEndcap, 5, 3);
        case 12: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 9) : EcalPnDiodeDetId(EcalEndcap, 5, 4);
        case 13: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 10) : EcalPnDiodeDetId(EcalEndcap, 5, 5);
        case 14: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 8) : EcalPnDiodeDetId(EcalEndcap, 5, 3);
        case 15: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 6) : EcalPnDiodeDetId(EcalEndcap, 5, 1);
        case 16: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 7) : EcalPnDiodeDetId(EcalEndcap, 5, 2);
        case 17: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 10) : EcalPnDiodeDetId(EcalEndcap, 5, 5);
        case 18: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 5) : EcalPnDiodeDetId(EcalEndcap, 5, 10);
        case 19: return pnA ? EcalPnDiodeDetId(EcalEndcap, 6, 9) : EcalPnDiodeDetId(EcalEndcap, 5, 4);
        default: return EcalPnDiodeDetId(0);
        }
      default: return EcalPnDiodeDetId(0);
      }

    }
 }

}
