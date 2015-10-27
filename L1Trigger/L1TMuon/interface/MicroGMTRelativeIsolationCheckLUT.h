#ifndef __l1microgmtrelativeisolationlut_h
#define __l1microgmtrelativeisolationlut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTRelativeIsolationCheckLUT : MicroGMTLUT {
    public: 
      MicroGMTRelativeIsolationCheckLUT (const edm::ParameterSet& iConfig, const std::string& setName);
      MicroGMTRelativeIsolationCheckLUT (const edm::ParameterSet& iConfig, const char* setName);
      virtual ~MicroGMTRelativeIsolationCheckLUT ();



      // returns the index corresponding to the calo tower sum 
      int lookup(int energySum, int pt) const;
      
      int hashInput(int energySum, int pt) const;
      void unHashInput(int input, int& energySum, int& pt) const;
    private:
      void getParameters(const edm::ParameterSet& iConfig, const char* setName);

      int m_ptMask; 
      int m_energySumMask;
      int m_energySumInWidth;
      int m_ptInWidth;
  };
}
#endif /* defined(__l1microgmtrelativeisolationlut_h) */