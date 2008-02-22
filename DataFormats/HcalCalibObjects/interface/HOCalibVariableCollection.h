#ifndef AlCaHOCalibProducer_HOCalibVariableCollection_h
#define AlCaHOCalibProducer_HOCalibVariableCollection_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

//struct HOCalibVariables;
//#include "DataFormats/HOCalibHit/interface/HOCalibVariables.h"
  //class Track;
  class HOCalibVariables;
  /// collection of HOcalibration variabale
  //  typedef std::vector<Track> HOCalibVariableCollection;
  typedef std::vector<HOCalibVariables> HOCalibVariableCollection;
  /// persistent reference to a HOcalibration varibale
  //  typedef edm::Ref<HOCalibVariableCollection> HOCalibRef;
  /// persistent reference to a HOcalibration varibales collection
  //  typedef edm::RefProd<HOCalibVariableCollection> HOcalibRefProd;
  /// vector of reference to HOcalibration in the same collection
  //  typedef edm::RefVector<HOCalibVariableCollection> HOCalibRefVector;
  /// iterator over a vector of reference to HOcalibration in the same collection
  //  typedef HOCalibRefVector::iterator hoCalib_iterator;

#endif
