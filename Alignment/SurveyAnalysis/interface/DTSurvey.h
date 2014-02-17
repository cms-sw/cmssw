/** \class DTSurvey
 *
 *  Implements a set of measurements given by survey, tipically a wheel.  
 *  Contains DTSurveyChambers and the displacements and rotations for each are 
 *  calculated.
 *
 *  $Date: 2008/04/11 05:08:01 $
 *  $Revision: 1.4 $
 *  \author Pablo Martinez Ruiz del Arbol
 */



#ifndef Alignment_SurveyAnalysis_DTSurvey_H
#define Alignment_SurveyAnalysis_DTSurvey_H

#include "TMatrixD.h"

class DTGeometry;
class DTSurveyChamber;

namespace edm { template<class> class ESHandle; }

class DTSurvey {

  
 public:
  DTSurvey(const std::string&, const std::string&, int);
  ~DTSurvey();
 
  void ReadChambers(edm::ESHandle<DTGeometry>);
  void CalculateChambers();

  const DTSurveyChamber * getChamber(int, int) const;

  int getId() const { return id; }

  //void ToDB(MuonAlignment *);
 
  private:
  void FillWheelInfo();

  std::string nameOfWheelInfoFile, nameOfChamberInfoFile;
  int id; 
  
  //This is the displacement (vector) and rotation (matrix) for the wheel
  float OffsetZ;
  TMatrixD delta;
  TMatrixD Rot; 

  DTSurveyChamber ***chambers;
  
};


std::ostream &operator<<(std::ostream &, const DTSurvey&);

#endif
