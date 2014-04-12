#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

CaloGenericDetId::CaloGenericDetId( DetId::Detector iDet ,
				    int             iSub ,
				    uint32_t        iDin  ) : DetId( iDet, iSub )
{
  if (det() == DetId::Hcal) { 
    std::cerr << "No support for HB/HE/HO/HF in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  } else {

   id_ = ( isEB() ? EBDetId::detIdFromDenseIndex( iDin ).rawId() :
	   ( isEE() ? EEDetId::detIdFromDenseIndex( iDin ).rawId() :
	     ( isES() ? ESDetId::detIdFromDenseIndex( iDin ).rawId() :
		 ( isZDC() ? HcalZDCDetId::detIdFromDenseIndex( iDin ).rawId() :
		   ( isCastor() ? HcalCastorDetId::detIdFromDenseIndex( iDin ).rawId() :
		     ( isCaloTower() ? CaloTowerDetId::detIdFromDenseIndex( iDin ).rawId() : 0 ) ) ) ) ) ); 
  }
}

uint32_t 
CaloGenericDetId::denseIndex() const 
{
  if (det() == DetId::Hcal) { 
    std::cerr << "No support for HB/HE/HO/HF in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  } 

   return ( isEB() ? EBDetId( rawId() ).denseIndex() :
	    ( isEE() ? EEDetId( rawId() ).denseIndex() :
	      ( isES() ? ESDetId( rawId() ).denseIndex() :
		  ( isZDC() ? HcalZDCDetId( rawId() ).denseIndex() :
		    ( isCastor() ? HcalCastorDetId( rawId() ).denseIndex() :
		      ( isCaloTower() ? CaloTowerDetId( rawId() ).denseIndex() : ~0 ) ) ) ) ) ) ;
}

uint32_t 
CaloGenericDetId::sizeForDenseIndexing() const 
{
  if (det() == DetId::Hcal) { 
    std::cerr << "No support for HB/HE/HO/HF in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  } 

   return ( isEB() ? EBDetId::kSizeForDenseIndexing :
	   ( isEE() ? EEDetId::kSizeForDenseIndexing :
	     ( isES() ? ESDetId::kSizeForDenseIndexing :
		 ( isZDC() ? HcalZDCDetId::kSizeForDenseIndexing :
		   ( isCastor() ? HcalCastorDetId::kSizeForDenseIndexing :
		     ( isCaloTower() ? CaloTowerDetId::kSizeForDenseIndexing : 0 ) ) ) ) ) ) ; 
}

bool 
CaloGenericDetId::validDetId() const       
{
   bool returnValue ( false ) ;
   if( isEB() )
   {
      const EBDetId ebid ( rawId() ) ;
      returnValue = EBDetId::validDetId( ebid.ieta(),
					 ebid.iphi() ) ;
   }
   else
   {
      if( isEE() )
      {
	 const EEDetId eeid ( rawId() ) ;
	 returnValue = EEDetId::validDetId( eeid.ix(), 
					    eeid.iy(),
					    eeid.zside() ) ;
      }
      else
      {
	 if( isES() )
	 {
	    const ESDetId esid ( rawId() ) ;
	    returnValue = ESDetId::validDetId( esid.strip(),
					       esid.six(),
					       esid.siy(), 
					       esid.plane(),
					       esid.zside() ) ;
	 }
	 else
	 {
	   if (det() == DetId::Hcal) { 
	     std::cerr << "No support for HB/HE/HO/HF in CaloGenericDetId" << std::endl;
	     throw cms::Exception("No support");

	     returnValue = false;
	   }
	   else
	     {
	       if( isZDC() )
		 {
		   const HcalZDCDetId zdid ( rawId() ) ;
		   returnValue = HcalZDCDetId::validDetId( zdid.section(),
							   zdid.channel()    ) ;
		 }
	       else
	       {
		  if( isCastor() )
		  {
		     const HcalCastorDetId zdid ( rawId() ) ;
		     returnValue = HcalCastorDetId::validDetId( zdid.section(),
								zdid.zside()>0,
								zdid.sector(),
								zdid.module() ) ;
		  }
		  else
		  {
		     if( isCaloTower() )
		     {
			const CaloTowerDetId ctid ( rawId() ) ;
			returnValue = CaloTowerDetId::validDetId( ctid.ieta(),
								  ctid.iphi() ) ;
		     }
		  }
	       }
	    }
	 }
      }
   }
   return returnValue ;
}

std::ostream& operator<<(std::ostream& s, const CaloGenericDetId& id) 
{
  if (id.det() == DetId::Hcal) { 
    std::cerr << "No support for HB/HE/HO/HF in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  } 

   return ( id.isEB() ? s<<EBDetId( id ) :
	    ( id.isEE() ? s<<EEDetId( id ) :
	      ( id.isES() ? s<<ESDetId( id ) :
		( id.isCaloTower() ? s<<CaloTowerDetId( id ) :
		    ( id.isZDC() ? s<<HcalZDCDetId( id ) :
		      s<<"UnknownId="<<std::hex<<id.rawId()<<std::dec ) ) ) ) ) ;
}
