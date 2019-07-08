#ifndef DQMHcalIsolatedBunchAlCaReco_H
#define DQMHcalIsolatedBunchAlCaReco_H

/** \class DQMHcalIsolatedBunchAlCaReco
 * *
 *  DQM Source for Hcal iolated bunch stream
 *
 *
 */

#include <string>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"


class DQMHcalIsolatedBunchAlCaReco : public DQMEDAnalyzer {
public:
  DQMHcalIsolatedBunchAlCaReco(const edm::ParameterSet &);
  ~DQMHcalIsolatedBunchAlCaReco() override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  //
  // Monitor elements
  //
  MonitorElement *h_Event_, *h_hbhehit_, *h_hfhit_, *h_hohit_;

  /// DQM folder name
  std::string folderName_, trigName_;
  bool plotAll_;

  /// object to monitor
  edm::EDGetTokenT<HBHERecHitCollection> hbhereco_;
  edm::EDGetTokenT<HORecHitCollection> horeco_;
  edm::EDGetTokenT<HFRecHitCollection> hfreco_;
  edm::EDGetTokenT<edm::TriggerResults> trigResult_;
};

#endif
