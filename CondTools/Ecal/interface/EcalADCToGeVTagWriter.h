#ifndef CondTools_Ecal_EcalADCToGeVTagWriter_h
#define CondTools_Ecal_EcalADCToGeVTagWriter_h

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"

namespace popcon {

  class EcalADCToGeVTagWriter : public popcon::PopConSourceHandler<EcalADCToGeVConstant> {
  public:
    EcalADCToGeVTagWriter(edm::ParameterSet const&);
    ~EcalADCToGeVTagWriter() override = default;

    void getNewObjects() override;

    std::string id() const override { return name_; }

  private:
    std::string const name_;
    unsigned int const since_;
    double const ebval_;
    double const eeval_;
  };

}  // namespace popcon
#endif
