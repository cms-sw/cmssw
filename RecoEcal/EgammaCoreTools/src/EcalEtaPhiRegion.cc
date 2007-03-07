#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"

EcalEtaPhiRegion::EcalEtaPhiRegion(double etaLow, double etaHigh, double phiLow, double phiHigh):
  etaLow_(etaLow),etaHigh_(etaHigh),phiLow_(phiLow),phiHigh_(phiHigh)
{
  // put phi in range -pi to pi
  if(phiLow_ > Geom::pi()) phiLow_ -= Geom::twoPi();
  if(phiLow_ < -Geom::pi()) phiLow_ += Geom::twoPi();
  if(phiHigh_ > Geom::pi()) phiHigh_ -= Geom::twoPi();
  if(phiHigh_ < -Geom::pi()) phiHigh_ += Geom::twoPi();
}

bool EcalEtaPhiRegion::inRegion(const GlobalPoint& position) const {

  double phi = position.phi();

  // put phi in range -pi to pi
  if(phi > Geom::pi()) phi -= Geom::twoPi();
  if(phi < -Geom::pi()) phi += Geom::twoPi();

  //resolve ambiguity in case region crosses phi=pi
  if (phi - phiLow_ > Geom::pi() || phi - phiHigh_ > Geom::pi())
    phi -= Geom::twoPi();
  if (phi - phiLow_ < -Geom::pi() || phi - phiHigh_ < -Geom::pi())
    phi += Geom::twoPi();

  return (position.eta() > etaLow_ && position.eta() < etaHigh_ &&
	  phi > phiLow_ && phi < phiHigh_);
}
