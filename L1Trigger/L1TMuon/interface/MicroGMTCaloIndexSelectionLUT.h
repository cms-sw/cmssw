#ifndef __l1microgmtcaloindexselectionlut_h
#define __l1microgmtcaloindexselectionlut_h

#include "MicroGMTLUT.h"
#include "MicroGMTConfiguration.h"

namespace l1t {
  class MicroGMTCaloIndexSelectionLUT : MicroGMTLUT {
    public: 
      MicroGMTCaloIndexSelectionLUT (const edm::ParameterSet& iConfig, const std::string& setName, int type);
      MicroGMTCaloIndexSelectionLUT (const edm::ParameterSet& iConfig, const char* setName, int type);
      virtual ~MicroGMTCaloIndexSelectionLUT ();



      // returns the index corresponding to the calo tower sum 
      int lookup(int angle) const;
      
      int hashInput(int angle) const { return angle; };
      void unHashInput(int input, int &angle) const { angle = input; }
    private:
      void getParameters(const edm::ParameterSet& iConfig, const char* setName, int type);

      int m_angleInWidth; 

  };
}

#endif /* defined(__l1microgmtcaloindexselectionlut_h) */