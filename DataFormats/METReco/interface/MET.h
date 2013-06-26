#ifndef METRECO_MET_H
#define METRECO_MET_H

/** \class MET
 *
 * This is the fundemental class for missing transverse momentum (a.k.a MET).
 * The class inherits from RecoCandidate, and so is stored in the event as 
 * a candidate.  The actual MET information is contained in the RecoCandidate 
 * LorentzVector while supplimentary information is stored as varibles in 
 * the MET class itself (such as SumET). As there may have been more than 
 * one correction applied to the missing transverse momentum, a vector of 
 * corrections to the missing px and the missing py is maintained
 * so that one can always recover the uncorrected MET by applying the 
 * negative of each correction.  
 *
 * \authors Michael Schmitt, Richard Cavanaugh The University of Florida
 * changes by Freya Blekman, Cornell University: Added significance interface
 * 
 * \version   1st Version May 31st, 2005.
 *
 ************************************************************/

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include <cmath>
#include <vector>
#include <cstring>
#include "TMatrixD.h"

namespace reco
{
  class MET : public RecoCandidate
    {
    public:
      //-----------------------------------------------------------------
      //Define Constructors
      MET();
      MET( const LorentzVector& p4_, const Point& vtx_ );
      MET( double sumet_, const LorentzVector& p4_, const Point& vtx_ );
      MET( double sumet_, const std::vector<CorrMETData>& corr_, 
	   const LorentzVector& p4_, const Point& vtx_ );
      //----------------------------------------------------------------- 
      //explicit clone function
      MET * clone() const;
      //-----------------------------------------------------------------
      //Define methods to extract elements related to MET
      //scalar sum of transverse energy over all objects
      double sumEt() const { return sumet; }       
      //MET Significance = MET / std::sqrt(SumET)
      double mEtSig() const { return ( sumet ? (this->et() / std::sqrt(sumet)) : (0.0) ); }
      //real MET significance
      double significance() const;
      //longitudinal component of the vector sum of energy over all object
      //(useful for data quality monitoring)
      double e_longitudinal() const {return elongit; }  
      //-----------------------------------------------------------------
      //Define different methods for the corrections to individual MET elements
      std::vector<double> dmEx() const ;
      std::vector<double> dmEy() const ;
      std::vector<double> dsumEt() const ;
      std::vector<double> dSignificance() const ;
      //Define method to extract the entire "block" of MET corrections
      std::vector<CorrMETData> mEtCorr() const { return corr; }
      //-----------------------------------------------------------------
      void setSignificanceMatrix(const TMatrixD& matrix);
      TMatrixD getSignificanceMatrix(void) const;
    private:
      virtual bool overlap( const Candidate & ) const;
      double sumet;
      double elongit;
      // bookkeeping for the significance
      double signif_dxx;
      double signif_dyy;
      double signif_dyx;
      double signif_dxy;
      std::vector<CorrMETData> corr;
    };
}

#endif // METRECO_MET_H
