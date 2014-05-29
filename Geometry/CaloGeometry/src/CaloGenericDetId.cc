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
  } 
  else if(isCaloTower()) {
    std::cerr << "No support for CaloTower in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  }
  else {

    id_ = ( isEB() ? EBDetId::detIdFromDenseIndex( iDin ).rawId() :
	    ( isEE() ? EEDetId::detIdFromDenseIndex( iDin ).rawId() :
	      ( isEK() ? EKDetId::detIdFromDenseIndex( iDin ).rawId() :
		( isES() ? ESDetId::detIdFromDenseIndex( iDin ).rawId() :
		  ( isZDC() ? HcalZDCDetId::detIdFromDenseIndex( iDin ).rawId() :
		    ( isCastor() ? HcalCastorDetId::detIdFromDenseIndex( iDin ).rawId() : 0 ) ) ) ) ) ); 
  }
}
  
uint32_t 
CaloGenericDetId::denseIndex() const 
{
  if (det() == DetId::Hcal) { 
    std::cerr << "No support for HB/HE/HO/HF in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  }
  else if(isCaloTower()) {
    std::cerr << "No support for CaloTower in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  }

   return ( isEB() ? EBDetId( rawId() ).denseIndex() :
	    ( isEE() ? EEDetId( rawId() ).denseIndex() :
	      ( isEK() ? EKDetId( rawId() ).denseIndex() :
		( isES() ? ESDetId( rawId() ).denseIndex() :
		  ( isZDC() ? HcalZDCDetId( rawId() ).denseIndex() :
		    ( isCastor() ? HcalCastorDetId( rawId() ).denseIndex() : ~0 ) ) ) ) ) );
}

uint32_t 
CaloGenericDetId::sizeForDenseIndexing() const 
{
  if (det() == DetId::Hcal) { 
    std::cerr << "No support for HB/HE/HO/HF in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  }
  else if(isCaloTower()) {
    std::cerr << "No support for CaloTower in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  }

   return ( isEB() ? EBDetId::kSizeForDenseIndexing :
	   ( isEE() ? EEDetId::kSizeForDenseIndexing :
	     ( isEK() ? EKDetId::kSizeForDenseIndexing :
	       ( isES() ? ESDetId::kSizeForDenseIndexing :
		 ( isZDC() ? HcalZDCDetId::kSizeForDenseIndexing :
		   ( isCastor() ? HcalCastorDetId::kSizeForDenseIndexing : 0 ) ) ) ) ) ); 
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
     if( isEK() )
       {
	 std::cerr << "CaloGenericDetId::validDetId-> not implemented for Shashlik EE" << std::endl;
	 return false;
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
                std::cerr << "No support for CaloTower in CaloGenericDetId" << std::endl;
                throw cms::Exception("No support");
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
  else if(id.isCaloTower()) {
    std::cerr << "No support for CaloTower in CaloGenericDetId" << std::endl;
    throw cms::Exception("No support");
  }

   return ( id.isEB() ? s<<EBDetId( id ) :
	    ( id.isEE() ? s<<EEDetId( id ) :
	     ( id.isEK() ? s<<EKDetId( id ) :
	       ( id.isES() ? s<<ESDetId( id ) :
		 ( id.isZDC() ? s<<HcalZDCDetId( id ) :
		   s<<"UnknownId="<<std::hex<<id.rawId()<<std::dec ) ) ) ) );
}
