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
 * \version $Revision: 1.1 $
 *
 * $Id: PFMEtSignCovMatrix.h,v 1.1 2012/02/13 13:56:46 veelken Exp $
 *
 */

#include "FWCore/Utilities/interface/Exception.h"

#include <TMatrixD.h>

class PFMEtSignCovMatrix : public TMatrixD
{
 public:
  /// constructor 
  explicit PFMEtSignCovMatrix() 
    : TMatrixD(2,2)
  {}
  explicit PFMEtSignCovMatrix(const PFMEtSignCovMatrix& bluePrint) 
    : TMatrixD(bluePrint)
  {}  
  explicit PFMEtSignCovMatrix(const TMatrixD& bluePrint) 
    : TMatrixD(bluePrint)
  {
    if ( !(GetNrows() == 2 && GetNcols() == 2) )
      throw cms::Exception("PFMEtSignCovMatrix")
	<< "Matrix must have 2x2 rows and colums !!\n";
  }

  /// destructor
  ~PFMEtSignCovMatrix() {}
  ClassDef(PFMEtSignCovMatrix, 1);
};

#endif
