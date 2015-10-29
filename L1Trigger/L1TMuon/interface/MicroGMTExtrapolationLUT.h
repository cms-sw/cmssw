#ifndef __l1microgmtextrapolationlut_h
#define __l1microgmtextrapolationlut_h

#include "MicroGMTLUT.h"

//FIXME move to cc
#include "MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTExtrapolationLUT : MicroGMTLUT {
    public: 
      MicroGMTExtrapolationLUT (const edm::ParameterSet& iConfig, const std::string& setName, int type);
      MicroGMTExtrapolationLUT (const edm::ParameterSet& iConfig, const char* setName, int type);
      virtual ~MicroGMTExtrapolationLUT ();



      // returns the index corresponding to the calo tower sum 
      int lookup(int angle, int pt) const;
      
      int hashInput(int angle, int pt) const;
      void unHashInput(int input, int& angle, int& pt) const;
    private:
      void getParameters(const edm::ParameterSet& iConfig, const char* setName, int type);

      int m_etaRedInWidth;
      int m_ptRedInWidth;

      int m_etaRedMask;
      int m_ptRedMask;
  };
}
#endif /* defined(__l1microgmtextrapolationlut_h) */