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
#include "DataFormats/METReco/interface/CommonMETData.h"
#include <Rtypes.h>
#include <cmath>
#include <vector>
#include <cstring>

namespace reco
{
  class MET 
    {
    public:
      //Define different Constructors
      MET();
      MET( double mex, double mey );
      MET( double mex, double mey, double sumet );
      MET( double mex, double mey, double sumet, double mez );
      MET( CommonMETData data_ );
      MET( CommonMETData data_, std::vector<CommonMETData> corr_ );
      //Define different methods to extract individual MET data elements
      double mEt()   const { return data.met; }
      double mEx()   const { return data.mex; }
      double mEy()   const { return data.mey; }
      double mEz()   const { return data.mez; }
      double sumEt() const { return data.sumet; }
      double phi()   const { return data.phi; }
      //Define different methods to extract corrections to individual MET elements
      std::vector<double> dmEt();
      std::vector<double> dmEx();
      std::vector<double> dmEy();
      std::vector<double> dsumEt();
      std::vector<double> dphi();
      //Define different methods to extract MET block data & corrections
      CommonMETData mEtData() const { return data; }
      std::vector<CommonMETData> mEtCorr() const { return corr; }
    private:
      CommonMETData data;
      std::vector<CommonMETData> corr;
    };
}

#endif // METRECO_MET_H
