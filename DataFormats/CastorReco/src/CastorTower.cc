#include "DataFormats/CastorReco/interface/CastorTower.h"

using namespace reco;

CastorTower::CastorTower(const double energy, const ROOT::Math::XYZPoint& position, const double emEnergy, const double hadEnergy,
			 const double fem, const double depth, const double fhot, 
			 const CastorCellRefVector& usedCells) : LeafCandidate(0,LorentzVector(math::PtEtaPhiMLorentzVector(energy*sin(position.theta()),position.eta(),position.phi(),0)),Point(0,0,0)) {

  position_ = position;
  energy_ = energy;
  emEnergy_ = emEnergy;
  hadEnergy_ = hadEnergy;
  fem_ = fem;
  depth_ = depth;
  fhot_ = fhot;
  for(CastorCellRefVector::const_iterator cellit  = usedCells.begin(); cellit != usedCells.end();++cellit) {
    usedCells_.push_back( (*cellit) );
  }
}

CastorTower::~CastorTower() {

}
