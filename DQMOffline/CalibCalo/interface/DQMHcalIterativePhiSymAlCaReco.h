#ifndef DQMHcalIterativePhiSymAlCaReco_H
#define DQMHcalIterativePhiSymAlCaReco_H

/** \class DQMHcalIterativePhiSymAlCaReco
 * *
 *  DQM Source for iterative phi symmetry stream
 *
 *  \author Sunanda Banerjee
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include <string>

class DQMHcalIterativePhiSymAlCaReco : public DQMOneEDAnalyzer<> {
public:
  DQMHcalIterativePhiSymAlCaReco(const edm::ParameterSet &);
  ~DQMHcalIterativePhiSymAlCaReco() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  //  void beginRun(const edm::Run& r, const edm::EventSetup& c);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &e, const edm::EventSetup &c) override;

  void dqmEndRun(const edm::Run &r, const edm::EventSetup &c) override {}

private:
  static constexpr int maxDepth_ = 7;
  //
  // Monitor elements
  //
  MonitorElement *hiDistr2D_[maxDepth_];

  MonitorElement *hiDistrHBHEsize1D_;
  MonitorElement *hiDistrHFsize1D_;
  MonitorElement *hiDistrHOsize1D_;

  std::string folderName_;
  int hiDistr_y_nbin_;
  int hiDistr_x_nbin_;
  double hiDistr_y_min_;
  double hiDistr_y_max_;
  double hiDistr_x_min_;
  double hiDistr_x_max_;
  /// object to monitor

  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
};

#endif
