///
/// Description: Firmware headers
///
/// Implementation:
///    Concrete firmware implementations
///
/// \author: Jim Brooke - University of Bristol
/// Modified: Adam Elwood - ICL

//
//

#ifndef Stage2Layer2DemuxJetAlgoFirmware_H
#define Stage2Layer2DemuxJetAlgoFirmware_H

#include "L1Trigger/L1TCalorimeter/interface/Stage2Layer2DemuxJetAlgo.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

namespace l1t {

  // Imp1 is for v1 and v2
  class Stage2Layer2DemuxJetAlgoFirmwareImp1 : public Stage2Layer2DemuxJetAlgo {
  public:
    Stage2Layer2DemuxJetAlgoFirmwareImp1(CaloParams* params);
    virtual ~Stage2Layer2DemuxJetAlgoFirmwareImp1();
    virtual void processEvent(const std::vector<l1t::Jet> & inputJets,
			      std::vector<l1t::Jet> & outputJets);

  private:

    CaloParams* const params_;

  };

}

#endif
