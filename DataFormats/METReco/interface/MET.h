#ifndef METRECO_MET_H
#define METRECO_MET_H

/** \class MET
 *
 * The MET EDProduct type. Stores a few basic variables
 * critical to all higher level MET products.
 *
 * \authors Michael Schmitt, Richard Cavanaugh The University of Florida
 *
 * \version   1st Version May 31st, 2005.
 *
 ************************************************************/

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/METReco/interface/CommonMETData.h"
#include "DataFormats/METReco/interface/CorrMETData.h"
#include <Rtypes.h>
#include <cmath>
#include <vector>
#include <cstring>

namespace reco
{
  class MET : public RecoCandidate
    {
    public:
      //Define different Constructors
      MET();
      MET(                                                const LorentzVector& p4_, const Point& vtx_ );
      MET( double sumet_,                                 const LorentzVector& p4_, const Point& vtx_ );
      MET( double sumet_, std::vector<CorrMETData> corr_, const LorentzVector& p4_, const Point& vtx_ );
      //Define different methods to extract individual MET data elements
      double sumEt() const { return sumet; }
      //Define different methods to extract corrections to individual MET elements
      std::vector<double> dmEx();
      std::vector<double> dmEy();
      std::vector<double> dsumEt();
      //Define different methods to extract MET block data & corrections
      std::vector<CorrMETData> mEtCorr() const { return corr; }
    private:
      virtual bool overlap( const Candidate & ) const;
      double sumet;
      //CommonMETData data;
      std::vector<CorrMETData> corr;
    };
}

#endif // METRECO_MET_H
