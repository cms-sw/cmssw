#ifndef CondTools_Hcal_HcalPFCutsHandler_h
#define CondTools_Hcal_HcalPFCutsHandler_h

#include <string>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondFormats/HcalObjects/interface/HcalPFCuts.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HcalPFCutsHandler : public popcon::PopConSourceHandler<HcalPFCuts> {
public:
  void getNewObjects() override;
  std::string id() const override { return m_name; }
  ~HcalPFCutsHandler() override;
  HcalPFCutsHandler(edm::ParameterSet const&);

  void initObject(HcalPFCuts*);

private:
  unsigned int sinceTime;
  edm::FileInPath fFile;
  HcalPFCuts* myDBObject;
  std::string m_name;
};
#endif
