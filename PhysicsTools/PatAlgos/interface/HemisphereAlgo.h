#ifndef HemisphereAlgo_h
#define HemisphereAlgo_h


/*  \class HemisphereAlgo
*
*  Class that, given the 4-momenta of the objects in the event, 
*  allows to split up the event into two "hemispheres" according 
*  to different criteria (see below)/
*
*  Authors: Luc Pape & Filip Moortgat      Date: July 2005
*                                          Updated: July 2006
*                                          Updated: 15 Sept 2006
*
*/


#include <vector>
#include <iostream>
#include <cmath>

class HemisphereAlgo {

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
//   1: 1st: max P ; 2nd: max P * delta R wrt first one 
//   2: 2 objects who give maximal invariant mass (recommended)
//
// HemisphereAlgo association method:
//   1: maximum pt longitudinal projected on the axes
//   2: minimal mass squared sum of the hemispheres
//   3: minimal Lund distance (recommended)
//
// Note that SetMethod also allows the seeding and/or association method to be
// redefined for an existing hemisphere object. The GetAxis or GetGrouping is 
// then recomputed using the newly defined methods.
//

HemisphereAlgo(std::vector<float> Px_vector, std::vector<float> Py_vector, std::vector<float> Pz_vector,
            std::vector<float> E_vector, int seed_method, int hemisphere_association_method);

HemisphereAlgo(std::vector<float> Px_vector, std::vector<float> Py_vector, std::vector<float> Pz_vector,
            std::vector<float> E_vector);

// Destructor
~HemisphereAlgo(){};


std::vector<float> getAxis1(); // returns Nx, Ny, Nz, P, E of the axis of group 1
std::vector<float> getAxis2(); // returns Nx, Ny, Nz, P, E of the axis of group 2

// where Nx, Ny, Nz are the direction cosines e.g. Nx = Px/P, 
// P is the momentum, E is the energy

std::vector<int> getGrouping(); // returns vector with "1" and "2"'s according to
                           // which group the object belongs 
			   // (order of objects in vector is same as input)

void SetMethod(int seed_method, int hemisphere_association_method){
  seed_meth = seed_method;
  hemi_meth = hemisphere_association_method;
  status = 0;
}                    // sets or overwrites the seed and association methods

void SetNoSeed(int object_number)  {
  Object_Noseed[object_number] = 1; 
  status = 0;
} 
                          // prevents an object from being used as a seed
                          // (method introduced on 15/09/06)

void SetDebug(int debug)  { dbg = debug; } 
 
 
private:


int reconstruct(); // the hemisphere separation algorithm

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

std::vector<float> Axis1;
std::vector<float> Axis2;

//static const float hemivsn = 1.01;
int seed_meth;
int hemi_meth;
int status;
int dbg;


float DeltaPhi(float, float);

};

#endif
