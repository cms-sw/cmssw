#include "TMath.h"
#include "TVectorD.h"
#include "PhysicsTools/CandUtils/interface/EventShapeVariables.h"


/// constructor from reco::Candidates
EventShapeVariables::EventShapeVariables(const edm::View<reco::Candidate>& inputVectors)
{
  inputVectors_.reserve( inputVectors.size() );
  for(edm::View<reco::Candidate>::const_iterator vec = inputVectors.begin(); vec!=inputVectors.begin(); ++vec){
    inputVectors_.push_back(math::XYZVector(vec->px(), vec->py(), vec->pz()));
  }
}

/// constructor from XYZ coordinates
EventShapeVariables::EventShapeVariables(const std::vector<math::XYZVector>& inputVectors) : inputVectors_(inputVectors)
{
}

/// constructor from rho eta phi coordinates
EventShapeVariables::EventShapeVariables(const std::vector<math::RhoEtaPhiVector>& inputVectors)
{
  inputVectors_.reserve( inputVectors.size() );
  for(std::vector<math::RhoEtaPhiVector>::const_iterator vec = inputVectors.begin(); vec!=inputVectors.begin(); ++vec){
    inputVectors_.push_back(math::XYZVector(vec->x(), vec->y(), vec->z()));
  }
}

/// constructor from r theta phi coordinates
EventShapeVariables::EventShapeVariables(const std::vector<math::RThetaPhiVector>& inputVectors)
{
  inputVectors_.reserve( inputVectors.size() );
  for(std::vector<math::RThetaPhiVector>::const_iterator vec = inputVectors.begin(); vec!=inputVectors.begin(); ++vec){
    inputVectors_.push_back(math::XYZVector(vec->x(), vec->y(), vec->z()));
  }
}

/// helper function to fill the 3 dimensional momentum tensor from the inputVecotrs where needed
TMatrixDSym 
EventShapeVariables::momentumTensor() const
{
  TMatrixDSym momentumTensor(3);
  momentumTensor.Zero();

  if(inputVectors_.size()<2){
    return momentumTensor;
  }
  // fill momentumTensor from inputVectors
  double norm=1.;
  for(int i=0; i<(int)inputVectors_.size(); ++i){
    norm+=inputVectors_[i].Dot(inputVectors_[i]);
    momentumTensor(1,1)+=inputVectors_[i].x()*inputVectors_[i].x();
    momentumTensor(1,2)+=inputVectors_[i].x()*inputVectors_[i].y();
    momentumTensor(1,3)+=inputVectors_[i].x()*inputVectors_[i].z();
    momentumTensor(2,1)+=inputVectors_[i].y()*inputVectors_[i].x();
    momentumTensor(2,2)+=inputVectors_[i].y()*inputVectors_[i].y();
    momentumTensor(2,3)+=inputVectors_[i].y()*inputVectors_[i].z();
    momentumTensor(3,1)+=inputVectors_[i].z()*inputVectors_[i].x();
    momentumTensor(3,2)+=inputVectors_[i].z()*inputVectors_[i].y();
    momentumTensor(3,3)+=inputVectors_[i].z()*inputVectors_[i].z();
  }
  // return normalized to 1
  return 1./norm*momentumTensor;
}

/// the return value is 1 for spherical events and 0 for events linear in r-phi. This function 
/// needs the number of steps to determine how fine the granularity of the algorithm in phi 
/// should be
double 
EventShapeVariables::isotropy(const unsigned int& numberOfSteps) const
{
  const double deltaPhi=2*TMath::Pi()/numberOfSteps;
  double phi = 0, eIn =-1., eOut=-1.;
  for(unsigned int i=0; i<numberOfSteps; ++i){
    phi+=deltaPhi;
    double sum=0;
    for(unsigned int j=0; j<inputVectors_.size(); ++j){
      // sum over inner product of unit vectors and momenta
      sum+=TMath::Abs(TMath::Cos(phi)*inputVectors_[j].x()+TMath::Sin(phi)*inputVectors_[j].y());
    }
    if( eOut<0. || sum<eOut ) eOut=sum;
    if( eIn <0. || sum>eIn  ) eIn =sum;
  }
  return (eIn-eOut)/eIn;
}

/// 1.5*(q1+q2) where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor sum{p_j[a]*p_j[b]}/sum{p_j**2} 
/// normalized to 1. Return values are 1 for spherical, 3/4 for plane and 0 for linear events
double 
EventShapeVariables::sphericity() const
{
  TVectorD eigenValues(3);
  TMatrixDSym myTensor=momentumTensor();
  if( myTensor.IsSymmetric() ){
    if( myTensor.NonZeros()!=0 ) 
      myTensor.EigenVectors(eigenValues);
  }
  double q1=eigenValues(0);
  double q2=eigenValues(1);
  if(eigenValues(2)<q1) {
    q1=eigenValues(2);
    if(eigenValues(0)<q2){
      q2=eigenValues(0);
    }
  }
  else if(eigenValues(2)<q2){
    q2=eigenValues(2);
  }
  return 1.5*(q1+q2);
}

/// 1.5*q1 where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor sum{p_j[a]*p_j[b]}/sum{p_j**2} 
/// normalized to 1. Return values are 0.5 for spherical and 0 for plane and linear events
double 
EventShapeVariables::aplanarity() const
{
  TVectorD eigenValues(3);
  TMatrixDSym myTensor=momentumTensor();
  if( myTensor.IsSymmetric() ){
    if( myTensor.NonZeros()!=0 )
      myTensor.EigenVectors(eigenValues);
  }
  double q1=eigenValues(0);
  if( eigenValues(1)<q1 ) {
    q1=eigenValues(1);
    if(eigenValues(2)<q1){
      q1=eigenValues(2);
    }
  }
  else if( eigenValues(2)<q1 ){
    q1=eigenValues(2);
  }
  return 1.5*q1;
}

/// the return value is 1 for spherical and 0 linear events in r-phi. This function needs the
/// number of steps to determine how fine the granularity of the algorithm in phi should be
double 
EventShapeVariables::circularity(const unsigned int& numberOfSteps) const
{
  const double deltaPhi=2*TMath::Pi()/numberOfSteps;
  double circularity=-1, phi=0, area = 0;
  for(unsigned int i=0;i<inputVectors_.size();i++) {
    area+=TMath::Sqrt(inputVectors_[i].x()*inputVectors_[i].x()+inputVectors_[i].y()*inputVectors_[i].y());
  }
  for(unsigned int i=0; i<numberOfSteps; ++i){
    phi+=deltaPhi;
    double sum=0, tmp=0.;
    for(unsigned int j=0; j<inputVectors_.size(); ++j){
      sum+=TMath::Abs(TMath::Cos(phi)*inputVectors_[j].x()+TMath::Sin(phi)*inputVectors_[j].y());
    }
    tmp=TMath::Pi()/2*sum/area;
    if( circularity<0 || tmp<circularity ){
      circularity=tmp;
    }
  }
  return circularity;
}

