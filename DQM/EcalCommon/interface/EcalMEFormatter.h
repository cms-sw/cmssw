#ifndef EcalMEFormatter_H
#define EcalMEFormatter_H

#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include "DQM/EcalCommon/interface/DQWorker.h"
#include "DQM/EcalCommon/interface/MESet.h"

class EcalMEFormatter : public DQMEDHarvester, public ecaldqm::DQWorker {
 public:
  EcalMEFormatter(edm::ParameterSet const&);
  ~EcalMEFormatter() {};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

 private:
  void dqmEndLuminosityBlock(DQMStore::IBooker&, DQMStore::IGetter&, edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void dqmEndJob(DQMStore::IBooker&, DQMStore::IGetter&) override;

  void format_(DQMStore::IGetter&, bool);
  void formatDet2D_(ecaldqm::MESet&);
};

#endif
