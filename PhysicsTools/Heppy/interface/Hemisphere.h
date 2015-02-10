#ifndef PhysicsTools_Heppy_Hemisphere_h
#define PhysicsTools_Heppy_Hemisphere_h


/*  \class Hemisphere
 *
 *  Class that, given the 4-momenta of the objects in the event, 
 *  allows to split up the event into two "hemispheres" according 
 *  to different criteria (see below)/
 *
 *  Authors: Luc Pape & Filip Moortgat      Date: July 2005
 *                                          Updated: July 2006
 *                                          Updated: 15 Sept 2006
 *                                          Updated: 15 November 2006
 *                                          Updated: 01 November 2008
 *                                          Updated: 13 December 2010 (P.Nef)
 *                                          Updated: 09 February 2011
 *                                          Updated: 26 September 2011 (P.Nef)
 */


#include <vector>
#include <iostream>
#include <cmath>

namespace heppy {

class Hemisphere {

 public:

  // There are 2 constructors:
  // 1. Constructor taking as argument vectors of Px, Py, Pz and E of the objects in
  // the event that should be separated, the seeding method and the hemisphere
  // association method,
  // 2. Constructor taking as argument vectors of Px, Py, Pz and E of the objects in
  // the event that should be separated. The seeding method and the hemisphere
  // association method should then be defined by SetMethod(seeding_method, association_method).
  // 
  // Seeding method: choice of 2 inital axes
  //   1: 1st: max P ; 2nd: max P * delta R wrt first one (if delta R > 0.5)
  //   2: 2 objects who give maximal invariant mass (recommended)
  //   3: 2 objects who give maximal transverse mass
  //   4: 2 leading jets (min dR can be set by SetDRminSeed1())
  //
  // Hemisphere association method:
  //   1: maximum pt longitudinal projected on the axes
  //   2: minimal mass squared sum of the hemispheres
  //   3: minimal Lund distance (recommended)
  //
  // Note that SetMethod() also allows the seeding and/or association method to be
  // redefined for an existing hemisphere object. The GetAxis() or GetGrouping() is 
  // then recomputed using the newly defined methods.
  //
  // Some parameters possibly affecting the logic of hemisphere reconstruction can also be set
  // by the user:
  // SetNoSeed() to prevent the given object from being used as a seed
  //  (but it can be associated to hemispheres).
  // SetNoAssoc() to prevent the given object from being used in the hemisphere association
  //  (then it cannot be used as seed either).
  // ClearAllNoLists() to reset the list of NoSeed and NoAssoc objects to empty.
  // SetDRminSeed1() minimum value of DeltaR between the two hemisphere seed for seed method 1
  //   (default = 0.5). Does not affect other methods.
  // SetnItermax() maximum number of iterations allowed for the association methods (default = 100).
  //
  // Methods are provided to avoid including ISR jets into the hemispheres.
  // This is done by declaring the variable type and cut value above which an object should not
  // be included in the hemispheres.
  // Only 1 cut can be used at the time (a new declaration overwrites the previous one).
  // RejectISRPtmax() max Pt w.r.t. the hemisphere below which objects can be included
  //   (default = 10000. GeV)
  // RejectISRDRmax() max DeltaR below which objects can be included
  //   (default = 10.)

  Hemisphere(){};
  
  Hemisphere(std::vector<float> Px_vector, std::vector<float> Py_vector, std::vector<float> Pz_vector, std::vector<float> E_vector, int seed_method, int hemisphere_association_method);

  Hemisphere(std::vector<float> Px_vector, std::vector<float> Py_vector, std::vector<float> Pz_vector, std::vector<float> E_vector);

  // Destructor
  ~Hemisphere(){};

  // return Nx, Ny, Nz, P, E of the axis of group 1
  std::vector<float> getAxis1(); 
  // return Nx, Ny, Nz, P, E of the axis of group 2
  std::vector<float> getAxis2(); 

  // where Nx, Ny, Nz are the direction cosines e.g. Nx = Px/P, 
  // P is the momentum, E is the energy

  // return vector with "1" and "2"'s according to which group the object belongs 
  //   and 0 if the object is not associated to any hemisphere
  //   (order of objects in vector is same as input)
  std::vector<int> getGrouping();

  // set or overwrite the seed and association methods
  void SetMethod(int seed_method, int hemisphere_association_method) {
    seed_meth = seed_method;
    hemi_meth = hemisphere_association_method;
    status = 0;
  }

  // prevent an object from being used as a seed (but it can be associated to hemispheres)
  // (method introduced on 15/09/06)
  void SetNoSeed(int object_number) {
    Object_Noseed[object_number] = 1; 
    status = 0;
  } 

  // prevent an object from being used for hemisphere asociation (and for seeding)
  // (method introduced on 01/11/08)
  void SetNoAssoc(int object_number) {
    Object_Noassoc[object_number] = 1; 
    Object_Noseed[object_number] = 1; 
    status = 0;
  } 

  // reset the list of NoSeed and NoAssoc objects to empty
  // (method introduced on 01/11/08)
  void ClearAllNoLists() {
    for (int i = 0; i < (int)Object_Noseed.size(); ++i) {
      Object_Noassoc[i] = 0; 
      Object_Noseed[i] = 0; 
      status = 0;
    }
  }

  // set the min DeltaR for the seeds of seed method 1
  // (method introduced on 01/11/08)
  void SetDRminSeed1(float rmin) {
    dRminSeed1 = rmin;
  }

  // set the maximum number of iterations allowed for association
  // (method introduced on 01/11/08)
  void SetnItermax(int niter) {
    nItermax = niter;
  }

  // set max Pt w.r.t. the hemisphere below which objects can be included
  void RejectISRPtmax(float ptmax) {
    rejectISRPtmax = ptmax;
    rejectISRPt = 1;
    rejectISRDR = 0;
    rejectISR = 1;
    status = 0;
  }

  // set max DeltaR below which objects can be included
  void RejectISRDRmax(float drmax) {
    rejectISRDRmax = drmax;
    rejectISRDR = 1;
    rejectISRPt = 0;
    rejectISR = 1;
    status = 0;
  }

  // controls the level of debug prints
  void SetDebug(int debug) { dbg = debug; } 
  int  GetNumLoop(){return numLoop;}
 
 
 private:

  // the hemisphere separation algorithm
  int Reconstruct();
  int RejectISR(); 


  std::vector<float> Object_Px;
  std::vector<float> Object_Py;
  std::vector<float> Object_Pz;
  std::vector<float> Object_P;
  std::vector<float> Object_Pt;
  std::vector<float> Object_E;
  std::vector<float> Object_Phi;
  std::vector<float> Object_Eta;
  std::vector<int> Object_Group;
  std::vector<int> Object_Noseed;
  std::vector<int> Object_Noassoc;

  std::vector<float> Axis1;
  std::vector<float> Axis2;

  //static const float hemivsn = 1.01;
  int seed_meth;
  int hemi_meth;
  int status;
  float dRminSeed1;
  int nItermax;
  int rejectISR;
  int rejectISRPt;
  float rejectISRPtmax;
  int rejectISRDR;
  float rejectISRDRmax;
  int dbg;
  int numLoop;

};
}

#endif
