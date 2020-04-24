#ifndef __l1microgmtabsoluteisolationlut_h
#define __l1microgmtabsoluteisolationlut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"


namespace l1t {
  class MicroGMTAbsoluteIsolationCheckLUT : public MicroGMTLUT {
    public: 
      MicroGMTAbsoluteIsolationCheckLUT() {};
      explicit MicroGMTAbsoluteIsolationCheckLUT(const std::string& fname);
      explicit MicroGMTAbsoluteIsolationCheckLUT(l1t::LUT* lut);
      ~MicroGMTAbsoluteIsolationCheckLUT() override {};

      // returns the index corresponding to the calo tower sum 
      int lookup(int energySum) const;
      
      int hashInput(int energySum) const { return energySum; }; 
      void unHashInput(int input, int& energySum) const { energySum = input; };
    private:
      void getParameters(const edm::ParameterSet& iConfig, const char* setName);

      int m_energySumInWidth;
  };
}

#endif /* defined(__l1microgmtabsoluteisolationlut_h) */
