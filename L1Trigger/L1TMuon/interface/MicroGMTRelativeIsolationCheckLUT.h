#ifndef __l1microgmtrelativeisolationlut_h
#define __l1microgmtrelativeisolationlut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTRelativeIsolationCheckLUT : public MicroGMTLUT {
    public: 
      MicroGMTRelativeIsolationCheckLUT() {};
      explicit MicroGMTRelativeIsolationCheckLUT(const std::string& fname);
      explicit MicroGMTRelativeIsolationCheckLUT(l1t::LUT* lut);
      virtual ~MicroGMTRelativeIsolationCheckLUT() {};

      // returns the index corresponding to the calo tower sum 
      int lookup(int energySum, int pt) const;
      
      int hashInput(int energySum, int pt) const;
      void unHashInput(int input, int& energySum, int& pt) const;
    private:
      int m_ptMask; 
      int m_energySumMask;
      int m_energySumInWidth;
      int m_ptInWidth;
  };
}
#endif /* defined(__l1microgmtrelativeisolationlut_h) */
