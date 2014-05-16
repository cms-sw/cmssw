///
/// \class l1t::CaloStage2JetAlgorithm
///
///

//

#ifndef Stage1Layer2JetAlgorithm_h
#define Stage1Layer2JetAlgorithm_h

#include "DataFormats/L1TCalorimeter/interface/CaloRegion.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1TCalorimeter/interface/CaloEmCand.h"

#include <vector>

namespace l1t {

  class Stage1Layer2JetAlgorithm {
  public:
    virtual void processEvent(const std::vector<l1t::CaloRegion> & regions,
			      const std::vector<l1t::CaloEmCand> & EMCands,
			      std::vector<l1t::Jet> * jets) = 0;
    virtual ~Stage1Layer2JetAlgorithm(){};
    double jetLsb;
    int jetSeedThreshold;
    std::string regionPUSType;
    std::vector<double> regionPUSParams;
    std::string jetCalibrationType;
    std::vector<double> jetCalibrationParams;

    /* private: */
    /*   double regionLSB_; // least significant bit value. Usually = 0.5 */
  };

}

#endif
