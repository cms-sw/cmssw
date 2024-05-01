#ifndef CondTools_Ecal_EcalTimeCalibHandler_h
#define CondTools_Ecal_EcalTimeCalibHandler_h

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondTools/Ecal/interface/EcalCondHeader.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include <string>

namespace popcon {
  class EcalTimeCalibHandler : public popcon::PopConSourceHandler<EcalTimeCalibConstants> {
  public:
    EcalTimeCalibHandler(edm::ParameterSet const&);
    ~EcalTimeCalibHandler() override = default;

    void getNewObjects() override;
    void readXML(const std::string& filename, EcalFloatCondObjectContainer& record);
    void readTXT(const std::string& filename, EcalFloatCondObjectContainer& record);

    std::string id() const override { return m_name; }
    EcalCondDBInterface* econn;

  private:
    const std::string m_name;
    const unsigned int m_firstRun;
    const std::string m_file_name;
    const std::string m_file_type;
  };
}  // namespace popcon
#endif
