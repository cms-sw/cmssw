#ifndef PlotAllAnalAlgos_included
#define PlotAllAnalAlgos_included 1

// user include files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "DataFormats/HcalDigi/interface/HcalCalibDataFrame.h"
#include "TBDataFormats/HcalTBObjects/interface/HcalTBTriggerData.h"
#include "RecoTBCalo/HcalPlotter/src/HcalQLPlotHistoMgr.h"
#include "TFile.h"

//
// class declaration
//

class HcalQLPlotAnalAlgos {
public:
  HcalQLPlotAnalAlgos(const char *outputFilename,
		   edm::ParameterSet histoParams);

  void end(void);
  void SetEventType(const HcalTBTriggerData& trigd) ;
  void processRH(const HBHERecHitCollection& hbherhc,
		 const HBHEDigiCollection& hbhedgc);
  void processRH(const HORecHitCollection& horhc,
		 const HODigiCollection& hodgc);
  void processRH(const HFRecHitCollection& hfrhc,
		 const HFDigiCollection& hfdgc);
  void processDigi(const HBHEDigiCollection& hbhedigic);
  void processDigi(const HODigiCollection& hodigic);
  void processDigi(const HFDigiCollection& hfdigic);
  void processDigi(const HcalCalibDigiCollection& calibdigic,double calibFC2GeV);

private:
  HcalCalibRecHit recoCalib(const HcalCalibDataFrame& cdigi,
			    double calibFC2GeV);

  // ----------member data ---------------------------
  HcalQLPlotHistoMgr::EventType triggerID_;
  HcalQLPlotHistoMgr *histos_;
  TFile              *mf_;
};

#endif // HcalQLPlotAnalAlgos_included
