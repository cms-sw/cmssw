#ifndef CondTools_Ecal_StoreEcalCondition_h
#define CondTools_Ecal_StoreEcalCondition_h

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include <string>
#include <map>
#include <memory>
#include <fstream>
#include <cstdio>
#include <typeinfo>
#include <sstream>

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"
#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalIntercalibConstantsMC.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

//
// class decleration
//

class StoreEcalCondition : public edm::one::EDAnalyzer<> {
public:
  std::shared_ptr<EcalWeightXtalGroups> readEcalWeightXtalGroupsFromFile(const char*);
  std::shared_ptr<EcalTBWeights> readEcalTBWeightsFromFile(const char*);
  std::shared_ptr<EcalADCToGeVConstant> readEcalADCToGeVConstantFromFile(const char*);
  std::shared_ptr<EcalIntercalibConstants> readEcalIntercalibConstantsFromFile(const char*, const char*);
  std::shared_ptr<EcalPFRecHitThresholds> readEcalPFRecHitThresholdsFromFile(const char*, const char*);
  std::shared_ptr<EcalIntercalibConstantsMC> readEcalIntercalibConstantsMCFromFile(const char*, const char*);
  std::shared_ptr<EcalGainRatios> readEcalGainRatiosFromFile(const char*);
  std::shared_ptr<EcalChannelStatus> readEcalChannelStatusFromFile(const char*);
  void writeToLogFile(std::string, std::string, unsigned long long);
  void writeToLogFileResults(char*);
  int convertFromConstructionSMToSlot(int, int);

  explicit StoreEcalCondition(const edm::ParameterSet& iConfig);
  ~StoreEcalCondition() override;

  void analyze(const edm::Event& evt, const edm::EventSetup& evtSetup) override;
  void endJob() override;

private:
  void fillHeader(char*);

  std::vector<std::string> objectName_;
  // it can be of type: EcalWeightXtalGroups, EcalTBWeights, EcalADCToGeVConstant, EcalIntercalibConstants, EcalGainRatios
  std::vector<std::string> inpFileName_;
  std::vector<std::string> inpFileNameEE_;
  std::string prog_name_;
  int sm_constr_;                          // SM number from data file
  int sm_slot_;                            // SM slot to map data to
  std::vector<unsigned long long> since_;  // beginning IOV for objects
  std::string logfile_;

  std::string to_string(char value[]) {
    std::ostringstream streamOut;
    streamOut << value;
    return streamOut.str();
  }
};
#endif
