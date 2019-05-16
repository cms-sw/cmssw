#ifndef DQMSourceEleCalib_H
#define DQMSourceEleCalib_H

/** \class DQMSourceEleCalib
 * *
 *  DQM Source for phi symmetry stream
 *
 *  \author Stefano Argiro'
 *          Andrea Gozzelino - Universita� e INFN Torino
 *
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"


class DQMSourceEleCalib : public DQMEDAnalyzer {
public:
  DQMSourceEleCalib(const edm::ParameterSet &);
  ~DQMSourceEleCalib() override;

protected:
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

private:
  //! find the MOX
  DetId findMaxHit(const std::vector<std::pair<DetId, float>> &,
                   const EcalRecHitCollection *,
                   const EcalRecHitCollection *);
  //! fills local occupancy graphs
  void fillAroundBarrel(const EcalRecHitCollection *, int, int);
  void fillAroundEndcap(const EcalRecHitCollection *, int, int);

  int eventCounter_;

  //! Number of recHits per electron
  MonitorElement *recHitsPerElectron_;
  //! Number of electrons
  MonitorElement *ElectronsNumber_;
  //! ESCoP
  MonitorElement *ESCoP_;
  //! Occupancy
  MonitorElement *OccupancyEB_;
  MonitorElement *OccupancyEEP_;
  MonitorElement *OccupancyEEM_;
  MonitorElement *LocalOccupancyEB_;
  MonitorElement *LocalOccupancyEE_;

  //! recHits over associated recHits
  MonitorElement *HitsVsAssociatedHits_;

  /// object to monitor
  edm::EDGetTokenT<EcalRecHitCollection> productMonitoredEB_;

  /// object to monitor
  edm::EDGetTokenT<EcalRecHitCollection> productMonitoredEE_;
  //! electrons to monitor
  edm::EDGetTokenT<reco::GsfElectronCollection> productMonitoredElectrons_;

  /// Monitor every prescaleFactor_ events
  unsigned int prescaleFactor_;

  /// DQM folder name
  std::string folderName_;

  /// Write to file
  bool saveToFile_;

  /// Output file name if required
  std::string fileName_;
};

#endif
