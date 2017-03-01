#ifndef __l1microgmtrankptquallut_h
#define __l1microgmtrankptquallut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1t {
  class MicroGMTRankPtQualLUT : public MicroGMTLUT {
    public:
      MicroGMTRankPtQualLUT() {};
      explicit MicroGMTRankPtQualLUT(const std::string&, const unsigned, const unsigned);
      explicit MicroGMTRankPtQualLUT(l1t::LUT*);
      virtual ~MicroGMTRankPtQualLUT() {};

      int lookup(int pt, int qual) const;
      virtual int lookupPacked(int in) const;

      int hashInput(int pt, int qual) const;
      void unHashInput(int input, int& pt, int& qual) const;
    private:
      int m_ptMask; 
      int m_qualMask; 
      int m_ptInWidth;
      int m_qualInWidth;

      // factor defining the weight of the two inputs when building the LUT
      unsigned m_ptFactor;
      unsigned m_qualFactor;
  };
}

#endif /* defined(__l1microgmtrankptquallut_h) */
