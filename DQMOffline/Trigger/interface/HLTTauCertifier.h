/*HLT Tau DQM Certification Module
Author : Michail Bachtis
University of Wisconsin-Madison
bachtis@hep.wisc.edu
*/

#include <memory>
#include <unistd.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"

class HLTTauCertifier : public DQMEDHarvester {
public:
  HLTTauCertifier(const edm::ParameterSet &);
  ~HLTTauCertifier() override;

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

private:
  std::string targetME_;
  std::string targetFolder_;
  std::vector<std::string> inputMEs_;
  bool setBadRunOnWarnings_;
  bool setBadRunOnErrors_;
};
