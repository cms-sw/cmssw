#ifndef ECALPEDESTALHISTORY_H
#define ECALPEDESTALHISTORY_H

/**\class EcalPedestalHistory

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// $Id: EcalPedestalHistory.h,v 0.0 2016/04/28 $
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include "CaloOnlineTools/EcalTools/interface/EcalFedMap.h"

#include "OnlineDB/EcalCondDB/interface/all_monitoring_types.h"
#include "OnlineDB/Oracle/interface/Oracle.h"
#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"

#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <string>
#include "TH1.h"
#include "TGraph.h"
#include "TH2.h"
#include <fstream>
#include <map>

class EcalPedestalHistory : public edm::EDAnalyzer {
public:
  explicit EcalPedestalHistory(const edm::ParameterSet&);
  ~EcalPedestalHistory() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void beginJob() override;
  void endJob() override;
  EcalCondDBInterface* econn;

  enum { kChannels = 75848, kEBChannels = 61200, kEEChannels = 14648 };
  enum { kGains = 3, kFirstGainId = 1 };

private:
  int runnumber_;
  unsigned int cnt_evt_;
  std::string ECALType_;  // EB or EE
  std::string runType_;   // Pedes or Other
  unsigned int startevent_;
  std::vector<int> maskedChannels_;
  std::vector<int> maskedEEChannels_;
  unsigned int m_firstRun;
  unsigned int m_lastRun;
  std::string m_location;
  std::string m_gentag;
  std::string m_sid;
  std::string m_user;
  std::string m_pass;
  std::string m_locationsource;
  std::string m_name;
};
#endif
