#ifndef DataFormats_METReco_PFMetSignCovMatrix_h__
#define DataFormats_METReco_PFMEtSignCovMatrix_h__

/** \class PFMEtSignCovMatrix
 *
 * Covariance matrix representing expected MET resolution,
 * computed by (PF)MEt significance algorithm
 * (see CMS AN-10/400 for description of the algorithm)
 * 
 * \author Christian Veelken, UC Davis
 *
 * \version $Revision: 1.3 $
 *
 * $Id: PFMEtSignCovMatrix.h,v 1.3 2013/02/22 15:23:22 veelken Exp $
 *
 */

#include "FWCore/Utilities/interface/Exception.h"

#include <TMatrixD.h>

class PFMEtSignCovMatrix
{
 public:
  /// constructor 
  explicit PFMEtSignCovMatrix() 
  {
    covXX_ = 0.;
    covXY_ = 0.;
    covYY_ = 0.;
  }
  explicit PFMEtSignCovMatrix(const PFMEtSignCovMatrix& bluePrint) 
  {
    covXX_ = bluePrint.covXX_;
    covXY_ = bluePrint.covXY_;
    covYY_ = bluePrint.covYY_;
  }  
  /// convert TMatrixD to PFMEtSignCovMatrix 
  explicit PFMEtSignCovMatrix(const TMatrixD& bluePrint) 
  {
    if ( !(bluePrint.GetNrows() == 2 && bluePrint.GetNcols() == 2) )
      throw cms::Exception("PFMEtSignCovMatrix")
	<< "Matrix must have 2x2 rows and colums !!\n";
    covXX_ = bluePrint(0,0);
    covXY_ = bluePrint(0,1);
    covYY_ = bluePrint(1,1);
  }

  /// destructor
  ~PFMEtSignCovMatrix() {}

  /// get number of rows and colums
  int GetNrows() const { return 2; }
  int GetNcols() const { return 2; }

  /// get value of element(row, column)
  double operator()(int row, int column) const
  {
    checkRange(row, column);
    if      ( row == 0 && column == 0 ) return covXX_;
    else if ( row == 0 && column == 1 ) return covXY_;
    else if ( row == 1 && column == 0 ) return covXY_;
    else if ( row == 1 && column == 1 ) return covYY_;
    else assert(0);
  }

  /// set value of element(row, column)
  double& operator()(int row, int column)
  {
    checkRange(row, column);
    if      ( row == 0 && column == 0 ) return covXX_;
    else if ( row == 0 && column == 1 ) return covXY_;
    else if ( row == 1 && column == 0 ) return covXY_;
    else if ( row == 1 && column == 1 ) return covYY_;
    else assert(0);
  }

  /// convert PFMEtSignCovMatrix to TMatrixD
  operator TMatrixD() const
  {
    TMatrixD matrix(2,2);
    matrix(0,0) = covXX_;
    matrix(0,1) = covXY_;
    matrix(1,0) = covXY_;
    matrix(1,1) = covYY_;
    return matrix;
  }

 protected:
  void checkRange(int row, int column) const
  {
    if ( !(row >= 0 && row <= 1) )
      throw cms::Exception("PFMEtSignCovMatrix")
	<< "Invalid row = " << row << ", expected range = 0..1 !!\n";
    if ( !(column >= 0 && column <= 1) )
      throw cms::Exception("PFMEtSignCovMatrix")
	<< "Invalid column = " << column << ", expected range = 0..1 !!\n";
  }  

 private:
  double covXX_;
  double covXY_;
  double covYY_;
};

#endif
