#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <math.h>

EcalRecHit::EcalRecHit() : CaloRecHit() {
}

EcalRecHit::EcalRecHit(const DetId& id, float energy, float time, uint32_t flags) :
  CaloRecHit(id,energy,time,flags) {
}

bool EcalRecHit::isRecovered() const {
        return (    recoFlag() == kLeadingEdgeRecovered 
                 || recoFlag() == kNeighboursRecovered 
                 || recoFlag() == kTowerRecovered 
                 );
}

float EcalRecHit::chi2Prob() const
{
        uint32_t rawChi2Prob = 0x7F & (flags()>>4);
        return (float)rawChi2Prob / (float)((1<<7)-1);
}

float EcalRecHit::outOfTimeChi2Prob() const
{
        uint32_t rawChi2Prob = 0x7F & (flags()>>24);
        return (float)rawChi2Prob / (float)((1<<7)-1);
}

float EcalRecHit::outOfTimeEnergy() const
{
        uint32_t rawEnergy = (0x1FFF & flags()>>11);
        uint16_t exponent = rawEnergy>>10;
        uint16_t significand = ~(0xE<<9) & rawEnergy;
        return (float) significand*pow(10,exponent-5);
}

void EcalRecHit::setRecoFlag( uint32_t flag )
{
        setFlags( (~0xF & flags()) | (flag & 0xF) );
}

void EcalRecHit::setChi2Prob( float chi2Prob )
{
        if ( chi2Prob < 0 || chi2Prob > 1 ) {
                edm::LogWarning("EcalRecHit::setChi2Prob") << "chi2Prob outside limits [0, 1] : " << chi2Prob;
        } else {
                // use 7 bits
                uint32_t rawChi2Prob = lround( chi2Prob * ((1<<7)-1) );
                // shift by 4 bits (recoFlag)
                setFlags( (~(0x7F<<4) & flags()) | ((rawChi2Prob & 0x7F)<<4) );
        }
}

void EcalRecHit::setOutOfTimeEnergy( float energy )
{
        if ( energy > 0 ) {
                uint16_t exponent = lround(floor(log10(energy)))+3;
                uint16_t significand = lround(energy/pow(10,exponent-5));
                // use 13 bits (3 exponent, 10 significand)
                uint32_t rawEnergy = exponent<<10 | significand;
                // shift by 11 bits (recoFlag + chi2)
                setFlags( ( ~(0x1FFF<<11) & flags()) | ((rawEnergy & 0x1FFF)<<11) );
        }
}

void EcalRecHit::setOutOfTimeChi2Prob( float chi2Prob )
{
        if ( chi2Prob < 0 || chi2Prob > 1 ) {
                edm::LogWarning("EcalRecHit::setOutOfTimeChi2Prob") << "chi2Prob outside limits [0, 1] : " << chi2Prob;
        } else {
                // use 7 bits
                uint32_t rawChi2Prob = lround( chi2Prob * ((1<<7)-1) );
                // shift by 24 bits (recoFlag + chi2 + outOfTimeEnergy)
                setFlags( (~(0x7F<<24) & flags()) | ((rawChi2Prob & 0x7F)<<24) );
        }
}


std::ostream& operator<<(std::ostream& s, const EcalRecHit& hit) {
  if (hit.detid().det() == DetId::Ecal && hit.detid().subdetId() == EcalBarrel) 
    return s << EBDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::Ecal && hit.detid().subdetId() == EcalEndcap) 
    return s << EEDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else if (hit.detid().det() == DetId::Ecal && hit.detid().subdetId() == EcalPreshower) 
    return s << ESDetId(hit.detid()) << ": " << hit.energy() << " GeV, " << hit.time() << " ns";
  else
    return s << "EcalRecHit undefined subdetector" ;
}
