#ifndef ECAL_TPGFINEGRAINTOWER_H
#define ECAL_TPGFINEGRAINTOWER_H

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainTowerEERcd.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon {
  class EcalTPGFineGrainTowerfromFile : public popcon::PopConSourceHandler<EcalTPGFineGrainTowerEE> {

  public:
    void getNewObjects() override;
    ~EcalTPGFineGrainTowerfromFile() override;
    EcalTPGFineGrainTowerfromFile(edm::ParameterSet const & ); 
    
    std::string id() const override { return m_name;}

  private:
    std::string m_name;
    std::string fname;
  };
}
#endif
