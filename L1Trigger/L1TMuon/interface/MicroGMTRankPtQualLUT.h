#ifndef __l1microgmtrankptquallut_h
#define __l1microgmtrankptquallut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1t {
  class MicroGMTRankPtQualLUT : public MicroGMTLUT {
    public:
      MicroGMTRankPtQualLUT ();
      explicit MicroGMTRankPtQualLUT (const edm::ParameterSet&);
      virtual ~MicroGMTRankPtQualLUT ();

      int lookup(int pt, int qual) const;
      virtual int lookupPacked(int in) const;

      int hashInput(int pt, int qual) const;
      void unHashInput(int input, int& pt, int& qual) const;
    private:
      int m_ptMask; 
      int m_qualMask; 
      int m_ptInWidth;
      int m_qualInWidth;

  };
}

#endif /* defined(__l1microgmtrankptquallut_h) */