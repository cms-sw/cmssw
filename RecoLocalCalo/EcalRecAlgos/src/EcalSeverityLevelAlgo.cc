#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

int EcalSeverityLevelAlgo::severityLevel( const DetId id, 
                const EcalRecHitCollection & recHits, 
                const EcalChannelStatus & chStatus,
                float recHitEtThreshold,
                SpikeId spId,
		float spIdThreshold,
		float recHitEnergyThresholdForTiming,
		float recHitEnergyThresholdForEE,
                float spIdThresholdIEta85
                )
{

        // get DB flag
        uint16_t dbStatus = retrieveDBStatus( id, chStatus );
        // get recHit flags
        EcalRecHitCollection::const_iterator it = recHits.find( id );
        if ( it == recHits.end() ) {
                // the channel is not in the recHit collection:
                // dead or zero-suppressed?
                if ( dbStatus >= 10 ) { // originally dead
                        return kBad;
                } else if ( dbStatus > 0 && dbStatus < 10 ) {
                        // zero-suppressed and originally problematic
                        return kProblematic;
                } else if ( dbStatus == 0 ) {
                        // zero-suppressed and originally good
                        return kGood;
                }
        } else {
                // the channel is in the recHit collection
                // .. is it a spike?
                // check the topology
                if ( id.subdetId() == EcalBarrel ) {
                        if ( abs(((EBDetId)id).ieta()) == 85 && spId == kSwissCrossBordersIncluded && spikeFromNeighbours(id, recHits, recHitEtThreshold, spId) > spIdThresholdIEta85 ) return kWeird;
                        if ( spikeFromNeighbours(id, recHits, recHitEtThreshold, spId) > spIdThreshold ) return kWeird;
                }
                // check the timing (currently only a trivial check)
		if ( id.subdetId() == EcalBarrel && spikeFromTiming(*it, recHitEnergyThresholdForTiming) ) return kTime;
                // filtering on VPT discharges in the endcap
                // introduced >= kSwisscross to take borders too, SA 20100913
                float re = 0; // protect the log function: make the computation possible only for re > 0
                if ( id.subdetId() == EcalEndcap && spId >= kSwissCross && (re = recHitE(id, recHits)) > 0 && ( 1-swissCross(id, recHits, recHitEnergyThresholdForEE, spId) < 0.02*log(re/4.) )  ) return kWeird;

                // .. not a spike, return the normal severity level
                return severityLevel( *it, chStatus );
        }
        return kGood;
}

int EcalSeverityLevelAlgo::severityLevel( const EcalRecHit &recHit, 
                const EcalChannelStatus &chStatus )
{
        // the channel is there, check its flags
        // and combine with DB (not needed at the moment)
        uint32_t rhFlag = recHit.recoFlag();
        uint16_t dbStatus = retrieveDBStatus( recHit.id(), chStatus );
        return severityLevel( rhFlag, dbStatus );
}

int EcalSeverityLevelAlgo::severityLevel( uint32_t rhFlag, uint16_t chStatus )
{
        // DB info currently not used at this level
        if       (  rhFlag == EcalRecHit::kPoorReco 
                 || rhFlag == EcalRecHit::kOutOfTime
                 || rhFlag == EcalRecHit::kNoisy
                 || rhFlag == EcalRecHit::kPoorCalib 
                 || rhFlag == EcalRecHit::kFaultyHardware
                 ) {
                // problematic
                return kProblematic;
        } else if ( rhFlag == EcalRecHit::kLeadingEdgeRecovered
                 || rhFlag == EcalRecHit::kNeighboursRecovered
                 || rhFlag == EcalRecHit::kTowerRecovered
                 ) {
                // recovered
                return kRecovered;
        } else if ( rhFlag == EcalRecHit::kDead
                 || rhFlag == EcalRecHit::kSaturated
                 //|| rhFlag == EcalRecHit::kFake // will be uncommented when validated
                 || rhFlag == EcalRecHit::kFakeNeighbours
                 || rhFlag == EcalRecHit::kKilled ) {
                // recovery failed (or not tried) or signal is fake or channel
                // is dead
                return kBad;
        }
        // good
        return kGood;
}

uint16_t EcalSeverityLevelAlgo::retrieveDBStatus( const DetId id, const EcalChannelStatus &chStatus )
{
        EcalChannelStatus::const_iterator chIt = chStatus.find( id );
        uint16_t dbStatus = 0;
        if ( chIt != chStatus.end() ) {
                dbStatus = chIt->getStatusCode();
        } else {
                edm::LogError("EcalSeverityLevelError") << "No channel status found for xtal " 
                        << id.rawId() 
                        << "! something wrong with EcalChannelStatus in your DB? ";
        }
        return dbStatus;
}

float EcalSeverityLevelAlgo::spikeFromNeighbours( const DetId id,
                                                  const EcalRecHitCollection & recHits,
                                                  float recHitThreshold,
                                                  SpikeId spId
                                                  )
{
  switch( spId ) {
  case kE1OverE9:
    return E1OverE9( id, recHits, recHitThreshold );
    break;
  case kSwissCross:
    return swissCross( id, recHits, recHitThreshold , true);
    break;
  case kSwissCrossBordersIncluded:
    return swissCross( id, recHits, recHitThreshold , false);
    break;
  default:
    edm::LogInfo("EcalSeverityLevelAlgo") << "Algorithm number " << spId
					  << " not known. Please check the enum in EcalSeverityLevelAlgo.h";
    break;
    
  }
        return 0;
}

float EcalSeverityLevelAlgo::E1OverE9( const DetId id, const EcalRecHitCollection & recHits, float recHitEtThreshold )
{
        // compute E1 over E9
        if ( id.subdetId() == EcalBarrel ) {
                // select recHits with Et above recHitEtThreshold
                if ( recHitApproxEt( id, recHits ) < recHitEtThreshold ) return 0;
                EBDetId ebId( id );
                float s9 = 0;
                for ( int deta = -1; deta <= +1; ++deta ) {
                        for ( int dphi = -1; dphi <= +1; ++dphi ) {
                                s9 += recHitE( id, recHits, deta, dphi );
                        }
                }
                return recHitE(id, recHits) / s9;
        } else if( id.subdetId() == EcalEndcap ) {
                // select recHits with Et above recHitEtThreshold
                if ( recHitApproxEt( id, recHits ) < recHitEtThreshold ) return 0;
                EEDetId eeId( id );
                float s9 = 0;
                for ( int dx = -1; dx <= +1; ++dx ) {
                        for ( int dy = -1; dy <= +1; ++dy ) {
                                s9 += recHitE( id, recHits, dx, dy );
                        }
                }
                return recHitE(id, recHits) / s9;

        }
        return 0;
}

float EcalSeverityLevelAlgo::swissCross( const DetId id, const EcalRecHitCollection & recHits, float recHitThreshold , bool avoidIeta85)
{
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

float EcalSeverityLevelAlgo::recHitE( const DetId id, const EcalRecHitCollection & recHits,
                                           int di, int dj )
{
        // in the barrel:   di = dEta   dj = dPhi
        // in the endcap:   di = dX     dj = dY
  
        DetId nid;
        if( id.subdetId() == EcalBarrel) nid = EBDetId::offsetBy( id, di, dj );
        else if( id.subdetId() == EcalEndcap) nid = EEDetId::offsetBy( id, di, dj );

        return ( nid == DetId(0) ? 0 : recHitE( nid, recHits ) );
}

float EcalSeverityLevelAlgo::recHitE( const DetId id, const EcalRecHitCollection &recHits )
{
        if ( id == DetId(0) ) {
                return 0;
        } else {
                EcalRecHitCollection::const_iterator it = recHits.find( id );
                if ( it != recHits.end() ) return (*it).energy();
        }
        return 0;
}


float EcalSeverityLevelAlgo::recHitApproxEt( const DetId id, const EcalRecHitCollection &recHits )
{
        // for the time being works only for the barrel
        if ( id.subdetId() == EcalBarrel ) {
                return recHitE( id, recHits ) / cosh( EBDetId::approxEta( id ) );
        }
        return 0;
}


bool EcalSeverityLevelAlgo::spikeFromTiming( const EcalRecHit &recHit, float recHitEnergyThreshold)
{
        if ( recHit.energy() < recHitEnergyThreshold )     return false;
        if ( recHit.recoFlag() == EcalRecHit::kOutOfTime ) return true;
        return false;
}



float EcalSeverityLevelAlgo::E2overE9( const DetId id, const EcalRecHitCollection & recHits, 
				       float recHitEtThreshold, float recHitEtThreshold2 , 
				       bool avoidIeta85, bool KillSecondHit)
{

        // compute e2overe9
        //  
        //   | | | |
        //   +-+-+-+
        //   | |1|2|
        //   +-+-+-+
        //   | | | |
        //
        //   1 - input hit,  2 - highest energy hit in a 3x3 around 1
        // 
        //   rechit 1 must have E_t > recHitEtThreshold
        //   rechit 2 must have E_t > recHitEtThreshold2
        //
        //   function returns value of E2/E9 centered around 1 (E2=energy of hits 1+2) if energy of 1>2
        //
        //   if energy of 2>1 and KillSecondHit is set to true, function returns value of E2/E9 centered around 2
        //   *provided* that 1 is the highest energy hit in a 3x3 centered around 2, otherwise, function returns 0


        if ( id.subdetId() == EcalBarrel ) {

                EBDetId ebId( id );

                // avoid recHits at |eta|=85 where one side of the neighbours is missing
                if ( abs(ebId.ieta())==85 && avoidIeta85) return 0;

                // select recHits with Et above recHitEtThreshold

 
                float e1 = recHitE( id, recHits );
                float ete1=recHitApproxEt( id, recHits );


		// check that rechit E_t is above threshold

		if (ete1 < std::min(recHitEtThreshold,recHitEtThreshold2) ) return 0;
		
		if (ete1 < recHitEtThreshold && !KillSecondHit ) return 0;
		

                float e2=-1;
                float ete2=0;
                float s9 = 0;

                // coordinates of 2nd hit relative to central hit
                int e2eta=0;
                int e2phi=0;

		// LOOP OVER 3x3 ARRAY CENTERED AROUND HIT 1

                for ( int deta = -1; deta <= +1; ++deta ) {
                   for ( int dphi = -1; dphi <= +1; ++dphi ) {
 
		      // compute 3x3 energy

                      float etmp=recHitE( id, recHits, deta, dphi );
                      s9 += etmp;

                      EBDetId idtmp=EBDetId::offsetBy(id,deta,dphi);
                      float eapproxet=recHitApproxEt( idtmp, recHits );

                      // remember 2nd highest energy deposit (above threshold) in 3x3 array 
                      if (etmp>e2 && eapproxet>recHitEtThreshold2 && !(deta==0 && dphi==0)) {

                         e2=etmp;
                         ete2=eapproxet;
                         e2eta=deta;
                         e2phi=dphi;
        
                      }

                   }
                }

                if ( e1 == 0 )  return 0;
  
                // return 0 if 2nd hit is below threshold
                if ( e2 == -1 ) return 0;

                // compute e2/e9 centered around 1st hit

                float e2nd=e1+e2;
                float e2e9=0;

                if (s9!=0) e2e9=e2nd/s9;
  
                // if central hit has higher energy than 2nd hit
                //  return e2/e9 if 1st hit is above E_t threshold

                if (e1 > e2 && ete1>recHitEtThreshold) return e2e9;

                // if second hit has higher energy than 1st hit

                if ( e2 > e1 ) { 


                  // return 0 if user does not want to flag 2nd hit, or
                  // hits are below E_t thresholds - note here we
		  // now assume the 2nd hit to be the leading hit.

		  if (!KillSecondHit || ete2<recHitEtThreshold || ete1<recHitEtThreshold2) {
 
                     return 0;
  
                 }


                  else {
 
                    // LOOP OVER 3x3 ARRAY CENTERED AROUND HIT 2

		    float s92nd=0;
           
                    float e2nd_prime=0;
                    int e2prime_eta=0;
                    int e2prime_phi=0;

                    EBDetId secondid=EBDetId::offsetBy(id,e2eta,e2phi);


                     for ( int deta = -1; deta <= +1; ++deta ) {
                        for ( int dphi = -1; dphi <= +1; ++dphi ) {
 
		           // compute 3x3 energy

                           float etmp=recHitE( secondid, recHits, deta, dphi );
                           s92nd += etmp;

                           if (etmp>e2nd_prime && !(deta==0 && dphi==0)) {
			     e2nd_prime=etmp;
                             e2prime_eta=deta;
                             e2prime_phi=dphi;
			   }

			}
		     }

		     // if highest energy hit around E2 is not the same as the input hit, return 0;

		     if (!(e2prime_eta==-e2eta && e2prime_phi==-e2phi)) 
		       { 
			 return 0;
		       }


		     // compute E2/E9 around second hit 
		     float e2e9_2=0;
		     if (s92nd!=0) e2e9_2=e2nd/s92nd;
                 
		     //   return the value of E2/E9 calculated around 2nd hit
                   
		     return e2e9_2;


		  }
		  
		}


        } else if ( id.subdetId() == EcalEndcap ) {
	  // only used for EB at the moment
          return 0;
        }
        return 0;
}

