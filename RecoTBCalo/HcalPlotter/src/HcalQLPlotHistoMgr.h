#ifndef HcalQLPlotHistoMgr_included
#define HcalQLPlotHistoMgr_included 1

#include "TH1.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalCalibDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

class HcalQLPlotHistoMgr {
public:
  enum HistType { ENERGY=0, TIME=1, PULSE=2, ADC=3 };
  enum EventType { UNKNOWN=0, PEDESTAL=1, LED=2, LASER=3, BEAM=4 };
  HcalQLPlotHistoMgr(TFileDirectory& parent, const edm::ParameterSet& histoParams);

  TH1* GetAHistogram(const HcalDetId& id,
		     const HcalElectronicsId& eid,
		     HistType ht, EventType et);

  TH1* GetAHistogram(const HcalCalibDetId& id,
		     const HcalElectronicsId& eid,
		     HistType ht, EventType et);

  static std::string nameForFlavor(HistType ht);
  static std::string nameForEvent(EventType et);
private:
  TH1* GetAHistogramImpl(const char *name, HistType ht, EventType et);

  TFileDirectory* pedHistDir;
  TFileDirectory* ledHistDir;
  TFileDirectory* laserHistDir;
  TFileDirectory* beamHistDir;
  TFileDirectory* ctrHistDir;
  TFileDirectory* otherHistDir;
  edm::ParameterSet histoParams_;
  std::map<std::string,TH1*> hists_;
};

#endif
