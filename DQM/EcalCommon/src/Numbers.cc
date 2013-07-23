// $Id: Numbers.cc,v 1.84 2012/04/27 13:46:04 yiiyama Exp $

/*!
  \file Numbers.cc
  \brief Some "id" conversions
  \author B. Gobbo
  \version $Revision: 1.84 $
  \date $Date: 2012/04/27 13:46:04 $
*/

#include <sstream>
#include <iomanip>
#include <set>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalElectronicsId.h"
#include "DataFormats/EcalDetId/interface/EcalPnDiodeDetId.h"
#include "DataFormats/EcalRawData/interface/EcalDCCHeaderBlock.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DQM/EcalCommon/interface/Numbers.h"

//-------------------------------------------------------------------------

const EcalElectronicsMapping* Numbers::map = 0;
const EcalTrigTowerConstituentsMap* Numbers::mapTT = 0;
const CaloGeometry *Numbers::geometry = 0;

const unsigned Numbers::crystalsTCCArraySize_;
const unsigned Numbers::crystalsDCCArraySize_;

std::vector<DetId> Numbers::crystalsTCC_[crystalsTCCArraySize_];
std::vector<DetId> Numbers::crystalsDCC_[crystalsDCCArraySize_];

bool Numbers::init = false;

//-------------------------------------------------------------------------

void
Numbers::initGeometry( const edm::EventSetup& setup, bool verbose )
{

  if( Numbers::init ) return;

  if ( verbose ) std::cout << "Initializing EcalElectronicsMapping ..." << std::endl;

  Numbers::init = true;

  edm::ESHandle<EcalElectronicsMapping> handle;
  setup.get<EcalMappingRcd>().get(handle);
  Numbers::map = handle.product();

  edm::ESHandle<EcalTrigTowerConstituentsMap> handleTT;
  setup.get<IdealGeometryRecord>().get(handleTT);
  Numbers::mapTT = handleTT.product();

  edm::ESHandle<CaloGeometry> handleGeom;
  setup.get<CaloGeometryRecord>().get(handleGeom);
  Numbers::geometry = handleGeom.product();

  if ( verbose ) std::cout << "done." << std::endl;

}

//-------------------------------------------------------------------------

int
Numbers::iEB( const unsigned ism )
{

  // EB-
  if( ism >=  1 && ism <= 18 ) return( -ism );

  // EB+
  if( ism >= 19 && ism <= 36 ) return( +ism - 18 );

  throw cms::Exception("InvalidParameter") << "Wrong SM id determination: iSM = " << ism;

}

//-------------------------------------------------------------------------

std::string
Numbers::sEB( const unsigned ism  )
{

  int ieb = Numbers::iEB( ism );

  std::ostringstream s;
  s << "EB" << std::setw(3) << std::setfill('0')
    << std::setiosflags( std::ios::showpos )
    << std::setiosflags( std::ios::internal )
    << ieb
    << std::resetiosflags( std::ios::showpos )
    << std::resetiosflags( std::ios::internal );
  return( s.str() );

}

//-------------------------------------------------------------------------

int
Numbers::iEE( const unsigned ism )
{

  // EE-
  if( ism ==  1 ) return( -7 );
  if( ism ==  2 ) return( -8 );
  if( ism ==  3 ) return( -9 );
  if( ism ==  4 ) return( -1 );
  if( ism ==  5 ) return( -2 );
  if( ism ==  6 ) return( -3 );
  if( ism ==  7 ) return( -4 );
  if( ism ==  8 ) return( -5 );
  if( ism ==  9 ) return( -6 );

  // EE+
  if( ism == 10 ) return( +7 );
  if( ism == 11 ) return( +8 );
  if( ism == 12 ) return( +9 );
  if( ism == 13 ) return( +1 );
  if( ism == 14 ) return( +2 );
  if( ism == 15 ) return( +3 );
  if( ism == 16 ) return( +4 );
  if( ism == 17 ) return( +5 );
  if( ism == 18 ) return( +6 );

  throw cms::Exception("InvalidParameter") << "Wrong SM id determination: iSM = " << ism;

}

//-------------------------------------------------------------------------

EcalSubdetector
Numbers::subDet( const EBDetId& id )
{

  return( id.subdet() );

}

//-------------------------------------------------------------------------

EcalSubdetector
Numbers::subDet( const EEDetId& id ) 
{

  return( id.subdet() );

}

//-------------------------------------------------------------------------

EcalSubdetector
Numbers::subDet( const EcalTrigTowerDetId& id )
{

  return( id.subDet() );

}

//-------------------------------------------------------------------------                                                                                                                                                                

EcalSubdetector
Numbers::subDet( const EcalScDetId& id )
{

  return( id.subdet() );

}

//-------------------------------------------------------------------------

EcalSubdetector
Numbers::subDet( const EcalElectronicsId& id ) 
{

  return( id.subdet() );

}

//-------------------------------------------------------------------------

EcalSubdetector
Numbers::subDet( const EcalPnDiodeDetId& id ) 
{

  return( (EcalSubdetector) id.iEcalSubDetectorId() );

}

//-------------------------------------------------------------------------

EcalSubdetector 
Numbers::subDet( const EcalDCCHeaderBlock& id )
{

  int idcc = id.id();

  // EE-
  if ( idcc >=  1 && idcc <=  9 ) return( EcalEndcap );

  // EB-/EB+
  if ( idcc >= 10 && idcc <= 45 ) return( EcalBarrel);

  // EE+
  if ( idcc >= 46 && idcc <= 54 ) return( EcalEndcap );

  throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

}

//-------------------------------------------------------------------------

std::string
Numbers::sEE( const unsigned ism  )
{

  int iee = Numbers::iEE( ism );

  std::ostringstream s;
  s << "EE" << std::setw(3) << std::setfill('0')
    << std::setiosflags( std::ios::showpos )
    << std::setiosflags( std::ios::internal )
    << iee
    << std::resetiosflags( std::ios::showpos )
    << std::resetiosflags( std::ios::internal );
  return( s.str() );

}

//-------------------------------------------------------------------------

// for EB, converts between two schemes. Old scheme [1:9] for EB-, new scheme (used in EBDetId) [1:9] for EB+
unsigned
Numbers::iSM( const unsigned ism, const EcalSubdetector subdet )
{

  if( subdet == EcalBarrel ) {

    if( ism >=  1 && ism <= 18 ) return( ism+18 );

    if( ism >= 19 && ism <= 36 ) return( ism-18 );

    throw cms::Exception("InvalidParameter") << "Wrong SM id: iSM = " << ism;

  } else if( subdet == EcalEndcap ) {

    if( ism >=  1 && ism <=  9 ) return( ism+9 );

    if (ism >= 10 && ism <= 18 ) return( ism-9 );

    throw cms::Exception("InvalidParameter") << "Wrong SM id: iSM = " << ism;

  }

  throw cms::Exception("InvalidParameter") << "Invalid subdetector: subdet = " << subdet;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSM( const EBDetId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  const EcalElectronicsId eid = Numbers::map->getElectronicsId(id);
  int idcc = eid.dccId();

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

}


//-------------------------------------------------------------------------

unsigned
Numbers::iSM( const EEDetId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  const EcalElectronicsId eid = Numbers::map->getElectronicsId(id);
  int idcc = eid.dccId();

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSM( const EcalTrigTowerDetId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  EcalSubdetector subdet = Numbers::subDet( id );

  if( subdet == EcalBarrel ) {

    int idcc = Numbers::map->DCCid(id);

    // EB-/EB+
    if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

    throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

  } else if( subdet ==  EcalEndcap) {

    int idcc = Numbers::map->DCCid(id);

    // EE-
    if( idcc >=  1 && idcc <=  9 ) return( idcc );

    // EE+
    if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

    throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

  }

  throw cms::Exception("InvalidParameter") << "Invalid subdetector: subdet = " << subdet;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSM( const EcalElectronicsId& id )
{

  int idcc = id.dccId();

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSM( const EcalPnDiodeDetId& id )
{

  int idcc = id.iDCCId();

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSM( const EcalScDetId& id )
{

  std::pair<int, int> dccsc = Numbers::map->getDCCandSC( id );

  int idcc = dccsc.first;

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSM( const EcalDCCHeaderBlock& id, const EcalSubdetector subdet )
{

  int idcc = id.id();

  // EE-
  if( idcc >=  1 && idcc <=  9 ) return( idcc );

  // EB-/EB+
  if( idcc >= 10 && idcc <= 45 ) return( idcc - 9 );

  // EE+
  if( idcc >= 46 && idcc <= 54 ) return( idcc - 45 + 9 );

  throw cms::Exception("InvalidParameter") << "Wrong DCC id: dcc = " << idcc;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSC( const EcalScDetId& id )
{

    std::pair<int, int> dccsc = Numbers::map->getDCCandSC( id );

    return static_cast<unsigned>(dccsc.second);

}

//-------------------------------------------------------------------------

unsigned
Numbers::iSC( const unsigned ism, const EcalSubdetector subdet, const unsigned i1, const unsigned i2 )
{

  if( subdet == EcalBarrel ) {

    int iet = 1 + ((i1-1)/5);
    int ipt = 1 + ((i2-1)/5);

    return( (ipt-1) + 4*(iet-1) + 1 );

  } else if( subdet == EcalEndcap ) {

    if( !Numbers::map ) throw( std::runtime_error( "ECAL Geometry not available" ) );

    // use ism only for determination of +/-

    int iz = 0;

    if( ism >=  1 && ism <=  9 ) iz = -1;
    if( ism >= 10 && ism <= 18 ) iz = +1;

    EEDetId id(i1, i2, iz, EEDetId::XYMODE); // throws an exception if invalid

    const EcalElectronicsId eid = Numbers::map->getElectronicsId(id);

    return( static_cast<unsigned>( eid.towerId() ));

  }

  throw cms::Exception("InvalidParameter") << "Invalid subdetector: subdet = " << subdet;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iTT( const unsigned ism, const EcalSubdetector subdet, const unsigned i1, const unsigned i2 )
{

  if( subdet == EcalBarrel ) {

    return( Numbers::iSC(ism, subdet, i1, i2) );

  } else if( subdet == EcalEndcap ) {

    if( !Numbers::mapTT ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

    // use ism only for determination of +/-

    int iz = 0;

    if( ism >=  1 && ism <=  9 ) iz = -1;
    if( ism >= 10 && ism <= 18 ) iz = +1;

    EEDetId id(i1, i2, iz, EEDetId::XYMODE); // throws an exception if invalid

    const EcalTrigTowerDetId towid = Numbers::mapTT->towerOf(id);

    return( static_cast<unsigned>( iTT(towid) ) );

  }

  throw cms::Exception("InvalidParameter") << "Invalid subdetector: subdet = " << subdet;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iTT( const EcalTrigTowerDetId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  EcalSubdetector subdet = Numbers::subDet( id );

  if( subdet == EcalBarrel || subdet ==  EcalEndcap ) return( static_cast<unsigned>( Numbers::map->iTT(id) ) );

  throw cms::Exception("InvalidParameter") << "Invalid subdetector: subdet = " << subdet;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iTCC( const unsigned ism, const EcalSubdetector subdet, const unsigned i1, const unsigned i2 )
{

  if( !Numbers::mapTT ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  if( subdet == EcalBarrel ) {

    EBDetId id(i1, i2, EBDetId::ETAPHIMODE);

    const EcalTrigTowerDetId towid = Numbers::mapTT->towerOf(id);

    return( static_cast<unsigned>( Numbers::map->TCCid(towid) ) );

  } else if( subdet ==  EcalEndcap) {

    int iz = 0;

    if( ism >=  1 && ism <=  9 ) iz = -1;
    if( ism >= 10 && ism <= 18 ) iz = +1;

    EEDetId id(i1, i2, iz, EEDetId::XYMODE);

    const EcalTrigTowerDetId towid = Numbers::mapTT->towerOf(id);

    return( static_cast<unsigned>( Numbers::map->TCCid(towid) ) );

  }

  throw cms::Exception("InvalidSubdetector") << "subdet = " << subdet;

}

//-------------------------------------------------------------------------

unsigned
Numbers::iTCC( const EcalTrigTowerDetId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  EcalSubdetector subdet = Numbers::subDet( id );

  if( subdet == EcalBarrel || subdet == EcalEndcap ) return( static_cast<unsigned>( Numbers::map->TCCid(id) ) );

  throw cms::Exception("InvalidParameter") << "Invalid subdetector: subdet = " << subdet;

}

//-------------------------------------------------------------------------

std::vector<DetId> *
Numbers::crystals( const EcalTrigTowerDetId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  int itcc = Numbers::map->TCCid(id);
  int itt = Numbers::map->iTT(id);

  unsigned index = 100*(itcc-1) + (itt-1);

  if( index >= crystalsTCCArraySize_ ) throw cms::Exception("InvalidParameter") << "TCC index " << index;

  if ( Numbers::crystalsTCC_[index].size() == 0 ) {
    Numbers::crystalsTCC_[index] = Numbers::map->ttConstituents( itcc, itt );
  }

  return &(Numbers::crystalsTCC_[index]);

}

//-------------------------------------------------------------------------

unsigned
Numbers::RtHalf(const EBDetId& id)
{

  int ic = id.ic();
  int ie = (ic-1)/20 + 1;
  int ip = (ic-1)%20 + 1;

  if( ie > 5 && ip < 11 ) return 1;

  return 0;

}

//-------------------------------------------------------------------------

unsigned
Numbers::RtHalf(const EEDetId& id)
{

  int ix = id.ix();

  int ism = Numbers::iSM( id );

  // EE-05
  if ( ism ==  8 && ix > 50 ) return 1;

  // EE+05
  if ( ism == 17 && ix > 50 ) return 1;

  return 0;

}

//-------------------------------------------------------------------------

std::vector<DetId> *
Numbers::crystals( const EcalElectronicsId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  int idcc = id.dccId();
  int isc = id.towerId();

  return Numbers::crystals( idcc, isc );

}

//-------------------------------------------------------------------------

std::vector<DetId> *
Numbers::crystals( unsigned idcc, unsigned isc )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  unsigned index = 100*(idcc-1) + (isc-1);

  if( index > crystalsDCCArraySize_ ) throw cms::Exception("InvalidParameter") << "DCC index " << index;

  if ( Numbers::crystalsDCC_[index].size() == 0 ) {
    Numbers::crystalsDCC_[index] = Numbers::map->dccTowerConstituents(idcc, isc);
  }

  return &(Numbers::crystalsDCC_[index]);

}

//-------------------------------------------------------------------------

const EcalScDetId
Numbers::getEcalScDetId( const EEDetId& id )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  const EcalElectronicsId& eid = Numbers::map->getElectronicsId(id);

  int idcc = eid.dccId();
  int isc = eid.towerId();

  const std::vector<EcalScDetId> ids = Numbers::map->getEcalScDetId( idcc, isc, true );

  return ids.size() > 0 ? ids[0] : EcalScDetId();

}

//-------------------------------------------------------------------------

unsigned
Numbers::indexEB( const unsigned ism, const unsigned ie, const unsigned ip )
{

  unsigned ic = (ip-1) + 20*(ie-1) + 1;

  if( ic == 0 || ic > static_cast<unsigned>( EBDetId::kCrystalsPerSM ) ) throw cms::Exception("InvalidParameter") << "ism=" << ism << " ie=" << ie << " ip=" << ip;

  return ic;

}

//-------------------------------------------------------------------------

unsigned
Numbers::indexEE( const unsigned ism, const unsigned ix, const unsigned iy )
{

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  if( !EEDetId::validDetId(ix, iy, iz) ) throw cms::Exception("InvalidParameter") << "ism=" << ism << " ix=" << ix << " iy=" << iy;

  return( 1000*ix + iy );

}

//-------------------------------------------------------------------------

unsigned
Numbers::icEB( const unsigned ism, const unsigned ie, const unsigned ip )
{

  return Numbers::indexEB( ism, ie, ip );

}

//-------------------------------------------------------------------------

unsigned
Numbers::icEE( const unsigned ism, const unsigned ix, const unsigned iy )
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  EEDetId id(ix, iy, iz, EEDetId::XYMODE);

  const EcalElectronicsId eid = Numbers::map->getElectronicsId(id);

  int vfe = eid.towerId();
  int strip = eid.stripId();
  int channel = eid.xtalId();

  // EE-05 & EE+05
  if( ism == 8 || ism == 17 ) {
    if( vfe > 17 ) vfe = vfe - 7;
  }

  unsigned ic = (vfe-1)*25 + (strip-1)*5 + channel;

  if( ic == 0 || ic > static_cast<unsigned>( EEDetId::kSizeForDenseIndexing ) ) throw cms::Exception("InvalidParameter") << "ic=" << ic;

  return ic;

}

//-------------------------------------------------------------------------

int
Numbers::ix0EE( const unsigned ism )
{

  if( ism == 1 || ism == 15 ) return( -  5 );
  if( ism == 2 || ism == 14 ) return( +  0 );
  if( ism == 3 || ism == 13 ) return( + 10 );
  if( ism == 4 || ism == 12 ) return( + 40 );
  if( ism == 5 || ism == 11 ) return( + 50 );
  if( ism == 6 || ism == 10 ) return( + 55 );
  if( ism == 7 || ism == 18 ) return( + 50 );
  if( ism == 8 || ism == 17 ) return( + 25 );
  if( ism == 9 || ism == 16 ) return( +  0 );

  return( + 0 );

}

//-------------------------------------------------------------------------

int
Numbers::ix0EEm( const unsigned ism )
{

  switch( ism ){
  case 1: return -105;
  case 2: return -100;
  case 3: return -90;
  case 4: return -60;
  case 5: return -50;
  case 6: return -45;
  case 7: return -50;
  case 8: return -75;
  case 9: return -100;
  }

  return ix0EE( ism );
}

int
Numbers::iy0EE( const unsigned ism )
{

  if( ism == 1 || ism == 10 ) return( + 20 );
  if( ism == 2 || ism == 11 ) return( + 45 );
  if( ism == 3 || ism == 12 ) return( + 55 );
  if( ism == 4 || ism == 13 ) return( + 55 );
  if( ism == 5 || ism == 14 ) return( + 45 );
  if( ism == 6 || ism == 15 ) return( + 20 );
  if( ism == 7 || ism == 16 ) return( +  0 );
  if( ism == 8 || ism == 17 ) return( -  5 );
  if( ism == 9 || ism == 18 ) return( +  0 );

  return( + 0 );

}

//-------------------------------------------------------------------------

bool
Numbers::validEE( const unsigned ism, const unsigned ix, const unsigned iy )
{

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  if( EEDetId::validDetId(ix, iy, iz) ) {

    EEDetId id(ix, iy, iz, EEDetId::XYMODE);

    if( Numbers::iSM( id ) == ism ) return true;

  }

  return false;

}

//-------------------------------------------------------------------------

bool
Numbers::validEESc( const unsigned ism, const unsigned ix, const unsigned iy )
{

  int iz = 0;

  if( ism >=  1 && ism <=  9 ) iz = -1;
  if( ism >= 10 && ism <= 18 ) iz = +1;

  if( EcalScDetId::validDetId(ix, iy, iz) ) {

    EcalScDetId id(ix, iy, iz);

    if( Numbers::iSM( id ) == ism ) return true;

  }

  return false;
}

unsigned
Numbers::nCCUs(const unsigned ism)
{
  switch(ism){
  case 8:
  case 17:
    return 41;
  case 1:
  case 6:
  case 10:
  case 15:
    return 34;
  case 3:
  case 12:
  case 4:
  case 13:
  case 7:
  case 16:
  case 9:
  case 18:
    return 33;
  case 2:
  case 11:
  case 5:
  case 14:
    return 32;
  default:
    return 0;
  }
}

unsigned
Numbers::nTTs(const unsigned itcc)
{
  using namespace std;
  vector<DetId> crystals(map->tccConstituents(itcc));

  set<int> itts;
  for(vector<DetId>::iterator cItr(crystals.begin()); cItr != crystals.end(); ++cItr)
    itts.insert(map->iTT(mapTT->towerOf(*cItr)));

  return itts.size();
}

const EcalElectronicsMapping *
Numbers::getElectronicsMapping()
{

  if( !Numbers::map ) throw cms::Exception("ObjectUnavailable") << "ECAL Geometry not available";

  return Numbers::map;

}

float
Numbers::eta( const DetId &id )
{
  const GlobalPoint& pos = geometry->getPosition(id);
  return pos.eta();
}

float
Numbers::phi( const DetId &id )
{
  const GlobalPoint& pos = geometry->getPosition(id);
  return pos.phi();
}
  
