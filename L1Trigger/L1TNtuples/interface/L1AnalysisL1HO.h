#ifndef __L1Analysis_L1AnalysisL1HO_H__
#define __L1Analysis_L1AnalysisL1HO_H__

#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"

#include "L1AnalysisL1HODataFormat.h"
namespace L1Analysis
{
  class L1AnalysisL1HO 
  {
  public:
    L1AnalysisL1HO();
    ~L1AnalysisL1HO();
    void Reset() {l1ho_.Reset();}
    void SetHO (const edm::SortedCollection<HODataFrame>& hoDataFrame);
    L1AnalysisL1HODataFormat * getData() {return &l1ho_;}

  private :
    L1AnalysisL1HODataFormat l1ho_;
  }; 
}
#endif


