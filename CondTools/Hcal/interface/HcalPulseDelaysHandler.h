#ifndef CondTools_Hcal_HcalPulseDelaysHandler_h
#define CondTools_Hcal_HcalPulseDelaysHandler_h

#include <string>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/HcalObjects/interface/HcalPulseDelays.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalPulseDelaysHandler : public popcon::PopConSourceHandler<HcalPulseDelays> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalPulseDelaysHandler() override;
  HcalPulseDelaysHandler(edm::ParameterSet const&);

  void initObject(HcalPulseDelays*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalPulseDelays* myDBObject;
  std::string m_name;
};
#endif
