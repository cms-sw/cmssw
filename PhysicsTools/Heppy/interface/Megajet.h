//-------------------------------------------------------
// Description:  Razor Megajet calculator
// Authors:      Maurizio Pierini, CERN
// Authors:      Christopher Rogan, Javier Duarte, Caltech
// Authors:      Emanuele Di Marco, CERN
//-------------------------------------------------------

///  The Megajet class implements the 
///  algorithm to combine jets in hemispheres
///  as defined in CMS SUSY  searches

#ifndef Megajet_h
#define Megajet_h

#include <vector>
#include <iostream>
#include <TLorentzVector.h>

namespace heppy {

class Megajet {

public:

  // There are 2 constructors:
  // 1. Constructor taking as argument vectors of Px, Py, Pz and E of the objects in the event and the hemisphere
  // association method,
  // 2. Constructor taking as argument vectors of Px, Py, Pz and E of the objects in the event. 
  // The hemisphere association method should then be defined by SetMethod(association_method).
  //
  // Hemisphere association method:
  //   1: minimum sum of the invariant masses of the two hemispheres 
  //   2: minimum difference of HT for the two hemispheres
  //   3: minimum m1^2/E1 + m2^2/E2
  //   4: Georgi distance: maximum (E1-Beta*m1^2/E1 + E2-Beta*m1^2/E2)

  Megajet(){};

  Megajet(std::vector<float> Px_vector, std::vector<float> Py_vector, std::vector<float> Pz_vector, std::vector<float> E_vector, int megajet_association_method);

  Megajet(std::vector<float> Px_vector, std::vector<float> Py_vector, std::vector<float> Pz_vector, std::vector<float> E_vector);

  /// Destructor
  ~Megajet(){};

  // return Nx, Ny, Nz, P, E of the axis of group 1
  std::vector<float> getAxis1(); 
  // return Nx, Ny, Nz, P, E of the axis of group 2
  std::vector<float> getAxis2(); 

  // where Nx, Ny, Nz are the direction cosines e.g. Nx = Px/P, 
  // P is the momentum, E is the energy

  // set or overwrite the seed and association methods
  void SetMethod(int megajet_association_method) {
    megajet_meth = megajet_association_method;
    status = 0;
  }

private:

  /// Combining the jets in two hemispheres minimizing the 
  /// sum of the invariant masses of the two hemispheres
  void CombineMinMass();

  /// Combining the jets in two hemispheres minimizing the 
  /// difference of HT for the two hemispheres
  void CombineMinHT();

  /// Combining the jets in two hemispheres by minimizing m1^2/E1 + m2^2/E2
  void CombineMinEnergyMass();

  /// Combining the jets in two hemispheres by maximizing (E1-Beta*m1^2/E1 + E2-Beta*m1^2/E2)
  void CombineGeorgi();

  /// Combine the jets in all the possible pairs of hemispheres
  void Combine();


  std::vector<float> Object_Px;
  std::vector<float> Object_Py;
  std::vector<float> Object_Pz;
  std::vector<float> Object_E;

  std::vector<float> Axis1;
  std::vector<float> Axis2;


  int megajet_meth;
  int status;
  
  std::vector<TLorentzVector> jIN;
  std::vector<TLorentzVector> jOUT;
  std::vector<TLorentzVector> j1;
  std::vector<TLorentzVector> j2;

};
}

#endif
