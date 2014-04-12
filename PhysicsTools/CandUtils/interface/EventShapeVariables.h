#ifndef EventShapeVariables_h
#define EventShapeVariables_h

/**
   \class   EventShapeVariables EventShapeVariables.h "PhysicsTools/CandUtils/interface/EventShapeVariables.h"

   \brief   Class for the calculation of several event shape variables

   Class for the calculation of several event shape variables. Isotropy, sphericity,
   aplanarity and circularity are supported. The class supports vectors of 3d vectors
   and edm::Views of reco::Candidates as input. The 3d vectors can be given in 
   cartesian, cylindrical or polar coordinates. It exploits the ROOT::TMatrixDSym 
   for the calculation of the sphericity and aplanarity.

   See http://cepa.fnal.gov/psm/simulation/mcgen/lund/pythia_manual/pythia6.3/pythia6301/node213.html
   for an explanation of sphericity, aplanarity and the quantities C and D.

   Author: Sebastian Naumann-Emme, University of Hamburg
           Roger Wolf, University of Hamburg
	   Christian Veelken, UC Davis
*/

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "TMatrixDSym.h"
#include "TVectorD.h"

#include <vector>

class EventShapeVariables {

 public:
  /// constructor from reco::Candidates
  explicit EventShapeVariables(const edm::View<reco::Candidate>& inputVectors);  
  /// constructor from XYZ coordinates
  explicit EventShapeVariables(const std::vector<math::XYZVector>& inputVectors);  
  /// constructor from rho eta phi coordinates
  explicit EventShapeVariables(const std::vector<math::RhoEtaPhiVector>& inputVectors);  
  /// constructor from r theta phi coordinates
  explicit EventShapeVariables(const std::vector<math::RThetaPhiVector>& inputVectors);  
  /// default destructor
  ~EventShapeVariables(){};

  /// the return value is 1 for spherical events and 0 for events linear in r-phi. This function 
  /// needs the number of steps to determine how fine the granularity of the algorithm in phi 
  /// should be
  double isotropy(const unsigned int& numberOfSteps = 1000) const;
  
  /// the return value is 1 for spherical and 0 linear events in r-phi. This function needs the 
  /// number of steps to determine how fine the granularity of the algorithm in phi should be
  double circularity(const unsigned int& numberOfSteps = 1000) const;

  /// 1.5*(q1+q2) where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor 
  /// sum{p_j[a]*p_j[b]}/sum{p_j**2} normalized to 1. Return values are 1 for spherical, 3/4 for 
  /// plane and 0 for linear events
  double sphericity(double = 2.)  const;
  /// 1.5*q1 where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor 
  /// sum{p_j[a]*p_j[b]}/sum{p_j**2} normalized to 1. Return values are 0.5 for spherical and 0 
  /// for plane and linear events
  double aplanarity(double = 2.)  const;
  /// 3.*(q1*q2+q1*q3+q2*q3) where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor 
  /// sum{p_j[a]*p_j[b]}/sum{p_j**2} normalized to 1. Return value is between 0 and 1 
  /// and measures the 3-jet structure of the event (C vanishes for a "perfect" 2-jet event)
  double C(double = 2.) const;
  /// 27.*(q1*q2*q3) where 0<=q1<=q2<=q3 are the eigenvalues of the momemtum tensor 
  /// sum{p_j[a]*p_j[b]}/sum{p_j**2} normalized to 1. Return value is between 0 and 1 
  /// and measures the 4-jet structure of the event (D vanishes for a planar event)
  double D(double = 2.) const;
  
 private:
  /// helper function to fill the 3 dimensional momentum tensor from the inputVectors where 
  /// needed
  TMatrixDSym compMomentumTensor(double = 2.) const;
  TVectorD compEigenValues(double = 2.) const;

  /// cashing of input vectors
  std::vector<math::XYZVector> inputVectors_;
};

#endif

