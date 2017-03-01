#ifndef __l1microgmtcaloindexselectionlut_h
#define __l1microgmtcaloindexselectionlut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTCaloIndexSelectionLUT : public MicroGMTLUT {
    public: 
      MicroGMTCaloIndexSelectionLUT() {};
      explicit MicroGMTCaloIndexSelectionLUT (const std::string& fname, int type);
      explicit MicroGMTCaloIndexSelectionLUT (l1t::LUT* lut, int type);
      virtual ~MicroGMTCaloIndexSelectionLUT() {};

      // returns the index corresponding to the calo tower sum 
      int lookup(int angle) const;
      
      int hashInput(int angle) const { return angle; };
      void unHashInput(int input, int &angle) const { angle = input; }
    private:
      int m_angleInWidth; 

  };
}

#endif /* defined(__l1microgmtcaloindexselectionlut_h) */
