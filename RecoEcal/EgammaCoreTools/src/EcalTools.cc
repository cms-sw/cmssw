
#include "RecoEcal/EgammaCoreTools/interface/EcalTools.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalNextToDeadChannel.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalNextToDeadChannelRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

float EcalTools::swissCross( const DetId& id, 
			     const EcalRecHitCollection & recHits, 
			     float recHitThreshold , 
			     bool avoidIeta85){
  // compute swissCross
  if ( id.subdetId() == EcalBarrel ) {
    EBDetId ebId( id );
    // avoid recHits at |eta|=85 where one side of the neighbours is missing
    // (may improve considering also eta module borders, but no
    // evidence for the time being that there the performance is
    // different)
    if ( abs(ebId.ieta())==85 && avoidIeta85) return 0;
    // select recHits with Et above recHitThreshold
    if ( recHitApproxEt( id, recHits ) < recHitThreshold ) return 0;
    float s4 = 0;
    float e1 = recHitE( id, recHits );
    // protect against nan (if 0 threshold is given above)
    if ( e1 == 0 ) return 0;
    s4 += recHitE( id, recHits,  1,  0 );
    s4 += recHitE( id, recHits, -1,  0 );
    s4 += recHitE( id, recHits,  0,  1 );
    s4 += recHitE( id, recHits,  0, -1 );
    return 1 - s4 / e1;
  } else if ( id.subdetId() == EcalEndcap ) {
    EEDetId eeId( id );
    // select recHits with E above recHitThreshold
    float e1 = recHitE( id, recHits );
    if ( e1 < recHitThreshold ) return 0;
    float s4 = 0;
    // protect against nan (if 0 threshold is given above)
    if ( e1 == 0 ) return 0;
    s4 += recHitE( id, recHits,  1,  0 );
    s4 += recHitE( id, recHits, -1,  0 );
    s4 += recHitE( id, recHits,  0,  1 );
    s4 += recHitE( id, recHits,  0, -1 );
    return 1 - s4 / e1;
  }
  return 0;
}


bool EcalTools::isNextToDead( const DetId& id, const edm::EventSetup& es){
  
  edm::ESHandle<EcalNextToDeadChannel> dch;
  es.get<EcalNextToDeadChannelRcd>().get(dch);
  
  EcalNextToDeadChannel::const_iterator chIt = dch->find( id );

  if ( chIt != dch->end() ) {
    return *chIt;
  } else {
    edm::LogError("EcalDBError") 
      << "No NextToDead  status found for xtal " 
      << id.rawId() ;
  }

  return false;
  
}

bool EcalTools::isNextToDeadFromNeighbours( const DetId& id, 
					    const EcalChannelStatus& chs,
					    int chStatusThreshold) {
  
  if (deadNeighbour(id,chs, chStatusThreshold, 1, 0)) return true;
  if (deadNeighbour(id,chs, chStatusThreshold,-1, 0)) return true;
  if (deadNeighbour(id,chs, chStatusThreshold, 0, 1)) return true;
  if (deadNeighbour(id,chs, chStatusThreshold, 0,-1)) return true;
  if (deadNeighbour(id,chs, chStatusThreshold, 1,-1)) return true;
  if (deadNeighbour(id,chs, chStatusThreshold, 1, 1)) return true;
  if (deadNeighbour(id,chs, chStatusThreshold,-1, 1)) return true;
  if (deadNeighbour(id,chs, chStatusThreshold,-1,-1)) return true;

  return false;
}


bool EcalTools::deadNeighbour(const DetId& id, 
			      const EcalChannelStatus& chs,
                              int chStatusThreshold, 
			      int dx, int dy){

  
  DetId nid;
  if( id.subdetId() == EcalBarrel) nid = EBDetId::offsetBy( id, dx, dy );
  else if( id.subdetId() == EcalEndcap) nid = EEDetId::offsetBy( id, dx, dy );

  if (!nid) return false;

  EcalChannelStatus::const_iterator chIt = chs.find( nid );
  uint16_t dbStatus = 0;
  if ( chIt != chs.end() ) {
    // note that 
    dbStatus = chIt->getStatusCode() ;
  } else {
    edm::LogError("EcalDBError") 
      << "No channel status found for xtal " 
      << id.rawId() 
      << "! something wrong with EcalChannelStatus in your DB? ";
  }

  return (dbStatus>=chStatusThreshold );
  
}



float EcalTools::recHitE( const DetId id, 
			  const EcalRecHitCollection & recHits,
			  int di, int dj )
{
  // in the barrel:   di = dEta   dj = dPhi
  // in the endcap:   di = dX     dj = dY
  
  DetId nid;
  if( id.subdetId() == EcalBarrel) nid = EBDetId::offsetBy( id, di, dj );
  else if( id.subdetId() == EcalEndcap) nid = EEDetId::offsetBy( id, di, dj );
  
  return ( nid == DetId(0) ? 0 : recHitE( nid, recHits ) );
}

float EcalTools::recHitE( const DetId id, const EcalRecHitCollection &recHits ){
  if ( id == DetId(0) ) {
    return 0;
  } else {
    EcalRecHitCollection::const_iterator it = recHits.find( id );
    if ( it != recHits.end() ) return (*it).energy();
  }
  return 0;
}

float EcalTools::recHitApproxEt( const DetId id, const EcalRecHitCollection &recHits ){
  // for the time being works only for the barrel
  if ( id.subdetId() == EcalBarrel ) {
    return recHitE( id, recHits ) / cosh( EBDetId::approxEta( id ) );
  }
  return 0;
}


bool isNextToBoundary(const DetId& id){

  if ( id.subdetId() == EcalBarrel ) 
    return EBDetId::isNextToBoundary(EBDetId(id));
  else  if ( id.subdetId() == EcalEndcap )
    return EEDetId::isNextToBoundary(EEDetId(id));

  return false;
}
