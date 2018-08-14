#ifndef ECAL_TPGFINEGRAINSTRIP_H
#define ECAL_TPGFINEGRAINSTRIP_H

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
#include "CondFormats/DataRecord/interface/EcalTPGFineGrainStripEERcd.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

namespace popcon {
  class EcalTPGFineGrainStripfromFile : public popcon::PopConSourceHandler<EcalTPGFineGrainStripEE> {

  public:
    void getNewObjects() override;
    ~EcalTPGFineGrainStripfromFile() override;
    EcalTPGFineGrainStripfromFile(edm::ParameterSet const & ); 
    
    std::string id() const override { return m_name;}

  private:
    std::string m_name;
    std::string fname;
  };
}
#endif
