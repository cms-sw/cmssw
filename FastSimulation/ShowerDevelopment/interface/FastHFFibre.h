#ifndef FastHFFibre_H
#define FastHFFibre_H
///////////////////////////////////////////////////////////////////////////////
// File: FastHFFibre.h 
// Description: Calculates attenuation length
///////////////////////////////////////////////////////////////////////////////

#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4ThreeVector.hh"

#include <vector>
#include <string>

class DDCompactView;    

class FastHFFibre {
  
public:
  
  //Constructor and Destructor
  FastHFFibre(std::string & name, const DDCompactView & cpv, double cFibre);
  ~FastHFFibre();

  double              attLength(double lambda);
  double              tShift(const G4ThreeVector& point, int depth, 
			     int fromEndAbs=0);
  double              zShift(const G4ThreeVector& point, int depth, 
			     int fromEndAbs=0);

protected:

  std::vector<double> getDDDArray(const std::string&, 
				  const DDsvalues_type&, int&);

private:

  double                      cFibre;
  std::vector<double>         gpar, radius;
  std::vector<double>         shortFL, longFL;
  std::vector<double>         attL;
  int                         nBinR, nBinAtt;
  double                      lambLim[2];

};
#endif
