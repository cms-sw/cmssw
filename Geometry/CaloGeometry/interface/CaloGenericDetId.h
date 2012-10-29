#ifndef CALOGEOMETRY_CALOGENERICDETID_H
#define CALOGEOMETRY_CALOGENERICDETID_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

class CaloGenericDetId : public DetId 
{
   public:
//      CaloGenericDetId() {}
      CaloGenericDetId( uint32_t rawid ) : DetId( rawid ) {}

      CaloGenericDetId( const DetId& id ) : DetId ( id ) {}

      CaloGenericDetId( DetId::Detector iDet ,
			int             iSub ,
			uint32_t        iDenseIndex  ) ; // to check valid iDenseIndex

      bool     validDetId() const ;

      uint32_t denseIndex() const ;

      uint32_t sizeForDenseIndexing() const ;

      bool isEcal()      const { return det() == DetId::Ecal ; }
      bool isEB()        const { return isEcal() && subdetId() == EBDetId::Subdet ; }
      bool isEE()        const { return isEcal() && subdetId() == EEDetId::Subdet ; }
      bool isES()        const { return isEcal() && subdetId() == ESDetId::Subdet ; }
      bool isCalo()      const { return det() == DetId::Calo ; }
      bool isZDC()       const { return isCalo() && subdetId() == HcalZDCDetId::SubdetectorId ; }
      bool isCastor()    const { return isCalo() && subdetId() == HcalCastorDetId::SubdetectorId ; }
      bool isCaloTower() const { return isCalo() && subdetId() == CaloTowerDetId::SubdetId ; } 
};

std::ostream& operator<<(std::ostream& s,const CaloGenericDetId& id);


#endif
