#ifndef METSignificance_H_
#define METSignificance_H_

/*
Class to just encapsulate a 2x2 TMatrixD to put into edm Event
*/

#include <TMatrixD.h>

namespace cmg
{
  
  class METSignificance{
  public:
    
    METSignificance():
      significance_(TMatrixD(2,2))
    {
    }

    METSignificance(const TMatrixD & matrix):
      significance_(TMatrixD(2,2))
    {
      significance_ = matrix;
    }

    virtual ~METSignificance(){
    }
    
    const TMatrixD& significance() const {return significance_;}

  private:

    TMatrixD significance_;

  };

}

#endif /*METSignificance_H_*/
