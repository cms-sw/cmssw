#include "Utilities/General/interface/precomputed_value_sort.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/GeometricSorting.h"
#include <iostream>
#include <iterator>

using namespace std;
using namespace geomsort;


typedef Surface::PositionType PositionType;
typedef Surface::RotationType RotationType;

// A simple helper for printing the object
struct dump {
  void operator() (const Surface& o) {
    cout << o.position()
	 << " R  : " << o.position().perp()
	 << " Phi: " << o.position().phi()
	 << " Z  : " << o.position().z()
	 << endl;
  }
  void operator() (const Surface* o) {
    operator()(*o);
  }  
};


int main() {

  //
  // Example of sorting a vector of objects store by value.
  //

  // Fill the vector to be sorted
  vector<Plane> v;
  v.push_back(Plane(PositionType(2,1,1),RotationType()));
  v.push_back(Plane(PositionType(1,1,2),RotationType()));
  v.push_back(Plane(PositionType(1,2,3),RotationType()));
  v.push_back(Plane(PositionType(2,2,4),RotationType()));


  cout << "Original  vector: " << endl;
  for_each(v.begin(),v.end(), dump());

  cout << "Sort in R       : " << endl;
  // Here we sort in R
  precomputed_value_sort(v.begin(),v.end(),ExtractR<Surface>());
  for_each(v.begin(),v.end(), dump());

  cout << "Sort in phi     : " << endl;
  // Here we sort in phi
  precomputed_value_sort(v.begin(),v.end(),ExtractPhi<Surface>());
  for_each(v.begin(),v.end(), dump());

  cout << "Sort in z       : " << endl;
  // Here we sort in Z
  precomputed_value_sort(v.begin(),v.end(),ExtractZ<Surface>());
  for_each(v.begin(),v.end(), dump());



  //
  // Now do the same with a vector of pointers
  //
  cout << endl << "Again with pointers" << endl;

  vector<const Plane*> vp;
  for (vector<Plane>::const_iterator i=v.begin(); i!=v.end(); i++){
    vp.push_back(&(*i));
  }

  cout << "Sort in R       : " << endl;
  // Here we sort in R
  precomputed_value_sort(vp.begin(),vp.end(),ExtractR<Surface>());
  for_each(vp.begin(),vp.end(), dump());

  cout << "Sort in phi     : " << endl;
  // Here we sort in phi
  precomputed_value_sort(vp.begin(),vp.end(),ExtractPhi<Surface>());
  for_each(vp.begin(),vp.end(), dump());

  cout << "Sort in z       : " << endl;
  // Here we sort in Z
  precomputed_value_sort(vp.begin(),vp.end(),ExtractZ<Surface>());
  for_each(vp.begin(),vp.end(), dump());


  return 0;  
}



