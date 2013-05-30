#ifndef EventShape_rivet_hh
#define EventShape_rivet_hh

/*  \class EventShape
*
*  Class that, given the 4-momenta of the objects in the event, 
*  allows to calculate event shapes (see below)
*
*  Authors: Matthias Weber                 Date: July/October 2007
*
*/


#include <vector>
#include <iostream>
#include <cmath>
#include "CLHEP/Vector/LorentzVector.h"
using namespace std;
using namespace CLHEP;
//#include <pair>

const int nevtvar=3; //Number of event shape variables

class EventShape_rivet {

public:

  //constructor taking as argument vectors of 
  //four vectors as input Objects 
  //in the event to calculate the event shapes 
  //
  EventShape_rivet(std::vector<HepLorentzVector> four_vector);

  //Destructor
~EventShape_rivet(){};

 std::vector<double> getEventShapes(); 
//returns the values of the event shapes

//1. central transverse thrust
//2. central thrust minor
//3. total jet broadening 

 void getSphericity(double* );
 //return transverse sphericity and C-parameter

 std::vector<double> getThrustAxis(); //returns the global thrust axis Nx, Ny, Nz=0
 std::vector<double> getThrustAxis_C(); //returns the central thrust axis Nx, Ny, Nz=0 


 //eta_c: choice of the central region
 //recommended: the two hardest jets should be within the central region

 // void SetEtac(double eta_central){
 //   eta_c = eta_central;
 // }


 // //wether to take the rapidity y (rap==1)  or the pseudorapidity eta (rap==0)
 // void SetMethod(int rapidity){
 //   rap = rapidity;
 // }

 private:

 int calculate();
 std::vector<HepLorentzVector> Object_4v;
 std::vector<double> testval;
 // std::vector<double> Object_Px;
 // std::vector<double> Object_Py;
 // std::vector<double> Object_Pz;

 std::vector<double> Object_P;
 std::vector<double> Object_Pt;
 std::vector<double> Object_E;
 std::vector<double> Object_Phi;
 std::vector<double> Object_Eta;
 std::vector<double> EventShapes;
 std::vector<double> ThrustAxis;
 std::vector<double> ThrustAxis_c;

 //returns the difference in phi between two vectors
 double DeltaPhi(double, double);

 // double max (double, double);

 // double min (double, double);

 // //the lorentz scalar product
 // double lorentz_sp (std::vector<double>, std::vector<double>);


 // //calculates the three-jet resolutions
 // std::pair<double, double> three_jet_res(std::vector<HepLorentzVector>, int);

 //calculates the thrust axis and the tau values
 std::vector<double> Thrust_calculate(std::vector<HepLorentzVector>);

 



};

#endif


