#ifndef MUONERRORMATRIX_H
#define MUONERRORMATRIX_H

/** \class MuonErrorMatrix
 *
 * This class holds an error matrix paramertization. Either the error matrix value or scale factors.
 * The implementation uses TProfile3D with pt, eta, phi axis.
 * Error/Scale factor matrix is obtained using get(GlobalVector momentum)
 *
 * $Dates: 2007/09/04 13:28 $
 * $Revision: 1.7 $
 *
 * \author Jean-Roch Vlimant  UCSB
 * \author Finn Rebassoo      UCSB
 */





#include <TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h>
#include <TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <TFile.h>
#include <TProfile3D.h>
#include <TString.h>
#include <TAxis.h>


class MuonErrorMatrix{
 public:
  /// enum type to define if the class is used as a tool or to be created
  enum action { use , constructor};
  /// constructor from a parameter set
  MuonErrorMatrix(const edm::ParameterSet & pset);
  
  /// destructor
  ~MuonErrorMatrix();

  /// close the root file attached to the class
  void close();
  
  /// main method to be used. Retrieve a 5x5 symetrical matrix according to parametrization of error or scale factor
  CurvilinearTrajectoryError get(GlobalVector momentum,bool convolute=true);
  CurvilinearTrajectoryError getFast(GlobalVector momentum);

  /// multiply term by term the two matrix
  static  void multiply(CurvilinearTrajectoryError & initial_error, const CurvilinearTrajectoryError & scale_error);

  /// divide term by term the two matrix
  static  bool divide(CurvilinearTrajectoryError & num_error, const CurvilinearTrajectoryError & denom_error);

  /// actually get access to the TProfile3D used for the parametrization
  inline TProfile3D * get(int i , int j) {return Index(i,j);}
  inline unsigned int index(int i, int j){return Pindex(i,j);}

  /// names of the variables of the 5x5 error matrix
  static const TString vars[5];

  /// provide the numerical value used. sigma or correlation factor
  static double Term(const AlgebraicSymMatrix55 & curv, int i, int j);

  ///method to get the bin index, taking care of under/overlow: first(1)/last(GetNbins())returned
  int findBin(TAxis * axis, double value);

  /// convert sigma2/COV -> sigma/rho
  void simpleTerm(const AlgebraicSymMatrix55 & input, AlgebraicSymMatrix55 & output);
  
  ///convert sigma/rho -> sigma2/COV
  void complicatedTerm(const AlgebraicSymMatrix55 & input, AlgebraicSymMatrix55 & output);

  /// adjust the error matrix on the state
  void adjust(FreeTrajectoryState & state);

  /// adjust the error matrix on the state
  void adjust(TrajectoryStateOnSurface & state);

 private:
  /// log category: "MuonErrorMatrix"
  std::string theCategory;

  /// the attached root file, where the parametrization is saved
  TDirectory * theD;
  /// 15 TProfile, each holding he parametrization of each term of the 5x5 
  TProfile3D * theData[15];
  TProfile3D * theData_fast[5][5];

  /// decide whether to scale of to assigne terms
  enum TermAction  { error, scale, assign };
  TermAction theTermAction[15];  

  /// internal methods to get the index of a matrix term.
  inline int Pindex(int i , int j) {
    static const int offset[5]={0,5,5+4,5+4+3,5+4+3+2};
    return offset[i]+abs(j-i);}
  /// internal method to get access to the profiles
  inline TProfile3D * Index(int i , int j) {
    return theData[Pindex(i,j)];}


  /// internal method that retreives the value of the parametrization for term i,j  
  double Value(GlobalVector & momentum, int i, int j,bool convolute=true);
  /// internal method that retreives the error on the value of the parametrization for term i,j  
  double Rms(GlobalVector & momentum, int i, int j);
  

};



#endif
