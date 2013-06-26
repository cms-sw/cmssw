// $Id: Photon.cc,v 1.28 2011/11/07 21:16:15 nancy Exp $
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

using namespace reco;

Photon::Photon( const LorentzVector & p4,
		Point caloPos,
    const PhotonCoreRef & core,
		const Point & vtx) :
    RecoCandidate( 0, p4, vtx, 22 ),
    caloPosition_( caloPos ),
    photonCore_(core),
    pixelSeed_(false)
 {}


Photon::Photon( const Photon& rhs ) :
  RecoCandidate(rhs),
  caloPosition_(rhs.caloPosition_),
  photonCore_ ( rhs.photonCore_),
  pixelSeed_  ( rhs.pixelSeed_ ),
  fiducialFlagBlock_ ( rhs.fiducialFlagBlock_ ),
  isolationR04_ ( rhs.isolationR04_),
  isolationR03_ ( rhs.isolationR03_),
  showerShapeBlock_ ( rhs.showerShapeBlock_),
  eCorrections_(rhs.eCorrections_),
  mipVariableBlock_ (rhs.mipVariableBlock_),
  pfIsolation_ ( rhs.pfIsolation_ )
 {}




Photon::~Photon() { }

Photon * Photon::clone() const {
  return new Photon( * this );
}


bool Photon::overlap( const Candidate & c ) const {
  const RecoCandidate * o = dynamic_cast<const RecoCandidate *>( & c );
  return ( o != 0 &&
	   ( checkOverlap( superCluster(), o->superCluster() ) )
	   );
  return false;
}

void Photon::setVertex(const Point & vertex) {
  math::XYZVectorF direction = caloPosition() - vertex;
  double energy = this->energy();
  math::XYZVectorF momentum = direction.unit() * energy;
  math::XYZTLorentzVector lv(momentum.x(), momentum.y(), momentum.z(), energy );
  setP4(lv);
  vertex_ = vertex;
}

reco::SuperClusterRef Photon::superCluster() const {
  return this->photonCore()->superCluster();
}

int Photon::conversionTrackProvenance(const edm::RefToBase<reco::Track>& convTrack) const{

  const reco::ConversionRefVector & conv2leg = this->photonCore()->conversions();
  const reco::ConversionRefVector & conv1leg = this->photonCore()->conversionsOneLeg();

  int origin = -1;
  bool isEg=false, isPf=false;

  for (unsigned iConv=0; iConv<conv2leg.size(); iConv++){
    std::vector<edm::RefToBase<reco::Track> > convtracks = conv2leg[iConv]->tracks();
    for (unsigned itk=0; itk<convtracks.size(); itk++){
      if (convTrack==convtracks[itk]) isEg=true;
    }
  }

  for (unsigned iConv=0; iConv<conv1leg.size(); iConv++){
    std::vector<edm::RefToBase<reco::Track> > convtracks = conv1leg[iConv]->tracks();
    for (unsigned itk=0; itk<convtracks.size(); itk++){
      if (convTrack==convtracks[itk]) isPf=true;
    }
  }

  if (isEg) origin=egamma;
  if (isPf) origin=pflow;
  if (isEg && isPf) origin=both;

  return origin;
}


void Photon::setCorrectedEnergy( P4type type, float newEnergy, float delta_e,  bool setToRecoCandidate ) {

  math::XYZTLorentzVectorD newP4 = p4() ;
  newP4 *= newEnergy/newP4.e() ;
  switch(type)
    {
    case ecal_standard:
      eCorrections_.scEcalEnergy = newEnergy;
      eCorrections_.scEcalEnergyError = delta_e;
      break ;
    case ecal_photons:
      eCorrections_.phoEcalEnergy = newEnergy;
      eCorrections_.phoEcalEnergyError = delta_e;
      break ;
    case  regression1:
      eCorrections_.regression1Energy = newEnergy ;
      eCorrections_.regression1EnergyError = delta_e;
    case  regression2:
      eCorrections_.regression2Energy = newEnergy ;
      eCorrections_.regression2EnergyError = delta_e;
      break ;
    default:
      throw cms::Exception("reco::Photon")<<"unexpected p4 type: "<< type ;
    }
  setP4(type, newP4,  delta_e,  setToRecoCandidate); 

}




 float  Photon::getCorrectedEnergy( P4type type) const {
  switch(type)
    {
    case ecal_standard:
      return      eCorrections_.scEcalEnergy;
      break ;
    case ecal_photons:
      return eCorrections_.phoEcalEnergy;
      break ;
    case  regression1:
      return eCorrections_.regression1Energy;
    case  regression2:
      return eCorrections_.regression2Energy;
      break ;
    default:
      throw cms::Exception("reco::Photon")<<"unexpected p4 type " << type << " cannot return the energy value: " ;
   }
 }


 float  Photon::getCorrectedEnergyError( P4type type) const {
  switch(type)
    {
    case ecal_standard:
      return      eCorrections_.scEcalEnergyError;
      break ;
    case ecal_photons:
      return eCorrections_.phoEcalEnergyError;
      break ;
    case  regression1:
      return eCorrections_.regression1EnergyError;
    case  regression2:
      return eCorrections_.regression2EnergyError;
      break ;
    default:
      throw cms::Exception("reco::Photon")<<"unexpected p4 type " << type << " cannot return the uncertainty on the energy: " ;
   }
 }



void Photon::setP4(P4type type, const LorentzVector & p4, float error, bool setToRecoCandidate ) {


  switch(type)
   {
    case ecal_standard:
      eCorrections_.scEcalP4 = p4 ;
      eCorrections_.scEcalEnergyError = error ;
      break ;
    case ecal_photons:
      eCorrections_.phoEcalP4 = p4 ;
      eCorrections_.phoEcalEnergyError = error ;
      break ;
    case  regression1:
      eCorrections_.regression1P4 = p4 ;
      eCorrections_.regression1EnergyError = error ;
    case  regression2:
      eCorrections_.regression2P4 = p4 ;
      eCorrections_.regression2EnergyError = error ;
      break ;
    default:
      throw cms::Exception("reco::Photon")<<"unexpected p4 type: "<< type ;
   }
  if (setToRecoCandidate)
   {
    setP4(p4) ;
    eCorrections_.candidateP4type = type ;
   }


}

const Candidate::LorentzVector& Photon::p4( P4type type ) const
 {
  switch(type)
    {
    case ecal_standard: return eCorrections_.scEcalP4 ;
    case ecal_photons: return eCorrections_.phoEcalP4 ;
    case regression1: return eCorrections_.regression1P4 ;
    case regression2: return eCorrections_.regression2P4 ;
    default: throw cms::Exception("reco::Photon")<<"unexpected p4 type: "<< type << " cannot return p4 ";
   }
 }
