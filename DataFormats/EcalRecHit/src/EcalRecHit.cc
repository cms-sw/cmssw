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
        return ( recoFlag() == kLeadingEdgeRecovered );
}

float EcalRecHit::chi2Prob() const
{
        uint32_t rawChi2Prob = 0xF & (flags()>>4);
        return (float)rawChi2Prob / (float)((1<<4)-1);
}

float EcalRecHit::outOfTimeEnergy() const
{
        uint32_t rawEnergy = 0x3FFF & (flags()>>8);
        return (float)rawEnergy * 0.1;
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
                uint32_t rawChi2Prob = lround( chi2Prob * ((1<<4)-1) ) << 4;
                setFlags( (~0xF0 & flags()) | (rawChi2Prob & 0xF0) );
        }
}

void EcalRecHit::setOutOfTimeEnergy( float energy )
{
        if ( energy < 0 ) {
                edm::LogWarning("EcalRecHit::setOutOfTimeEnergy") << "Negative energy, cannot set it : " << energy;
        } else {
                uint32_t rawEnergy = lround( energy / 0.1); // 100 MeV resolution
                setFlags( ( ~(0x3FFF<<8) & flags()) | ((rawEnergy & 0x3FFF)<<8) );
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
