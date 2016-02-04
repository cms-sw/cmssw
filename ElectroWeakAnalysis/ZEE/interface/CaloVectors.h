#include "RecoEcal/EgammaClusterAlgos/interface/EgammaSCEnergyCorrectionAlgo.h"

math::XYZTLorentzVector DetectorVector(const reco::SuperClusterRef& sc)
{
	double pt = sc->energy()/cosh(sc->eta());
	math::XYZTLorentzVector detVec(pt*cos(sc->phi()), pt*sin(sc->phi()), pt*sinh(sc->eta()), sc->energy());
	return detVec;
}
math::XYZTLorentzVector DetectorVector(const reco::SuperCluster sc)
{
	double pt = sc.energy()/cosh(sc.eta());
	math::XYZTLorentzVector detVec(pt*cos(sc.phi()), pt*sin(sc.phi()), pt*sinh(sc.eta()), sc.energy());
	return detVec;
}
math::XYZTLorentzVector DetectorVector(const GlobalPoint& pos, const math::XYZPoint& vertex, double energy)
{
	math::XYZPoint hitPos(pos.x(), pos.y(), pos.z());
	math::XYZVector Vec = hitPos - vertex;
	double eta = Vec.Eta();
	double phi = Vec.Phi();
	double pt = energy/cosh(eta);
	math::XYZTLorentzVector detVec(pt*cos(phi), pt*sin(phi), pt*sinh(eta), energy);
	return detVec;
}
math::XYZTLorentzVector PhysicsVector(const math::XYZPoint& vertex, const reco::SuperCluster& sc)
{
	math::XYZVector Vec = sc.position() - vertex;
	double eta = Vec.Eta();
	double phi = Vec.Phi();
	double pt = sc.energy()/cosh(eta);
	math::XYZTLorentzVector probe(pt*cos(phi), pt*sin(phi), pt*sinh(eta), sc.energy());
	return probe;
}
math::XYZTLorentzVector PhysicsVectorRaw(const math::XYZPoint& vertex, const reco::SuperCluster& sc)
{
	math::XYZVector Vec = sc.position() - vertex;
	double eta = Vec.Eta();
	double phi = Vec.Phi();
	double pt = sc.rawEnergy()/cosh(eta);
	math::XYZTLorentzVector probe(pt*cos(phi), pt*sin(phi), pt*sinh(eta), sc.rawEnergy());
	return probe;
}
