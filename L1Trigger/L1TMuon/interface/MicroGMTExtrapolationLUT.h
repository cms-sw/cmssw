#ifndef __l1microgmtextrapolationlut_h
#define __l1microgmtextrapolationlut_h

#include "MicroGMTLUT.h"

//FIXME move to cc
#include "MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTExtrapolationLUT : public MicroGMTLUT {
    public:
      MicroGMTExtrapolationLUT() {};
      explicit MicroGMTExtrapolationLUT(const std::string& fname, int outWidth, int etaRedInWidth, int ptRedInWidth);
      explicit MicroGMTExtrapolationLUT(l1t::LUT* lut, int outWidth, int etaRedInWidth, int ptRedInWidth);
      virtual ~MicroGMTExtrapolationLUT() {};

      // returns the index corresponding to the calo tower sum 
      int lookup(int angle, int pt) const;
      
      int hashInput(int angle, int pt) const;
      void unHashInput(int input, int& angle, int& pt) const;

      int getEtaRedInWidth() const;
      int getPtRedInWidth() const;
    private:
      int m_etaRedInWidth;
      int m_ptRedInWidth;

      int m_etaRedMask;
      int m_ptRedMask;
  };
}
#endif /* defined(__l1microgmtextrapolationlut_h) */
