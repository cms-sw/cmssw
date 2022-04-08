#ifndef DQM_GEM_GEMEffByGEMCSCSegmentClient_h
#define DQM_GEM_GEMEffByGEMCSCSegmentClient_h

/** \class GEMEffByGEMCSCSegmentClient
 * 
 * `GEMEffByGEMCSCSegmentSource` measures the efficiency of GE11-L1(2) using GE11-L2(1) and ME11 as trigger detectors.
 * See https://github.com/cms-sw/cmssw/blob/CMSSW_12_3_0_pre5/RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegAlgoRR.cc
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQM/GEM/interface/GEMDQMEfficiencyCalculator.h"

class GEMEffByGEMCSCSegmentClient : public DQMEDHarvester {
public:
  GEMEffByGEMCSCSegmentClient(const edm::ParameterSet &);
  ~GEMEffByGEMCSCSegmentClient() override{};
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

protected:
  void dqmEndLuminosityBlock(DQMStore::IBooker &,
                             DQMStore::IGetter &,
                             edm::LuminosityBlock const &,
                             edm::EventSetup const &) override;

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override{};

  // initialized in the constructor initializer list
  const std::string kFolder_;
  const std::string kLogCategory_;

  std::unique_ptr<GEMDQMEfficiencyCalculator> eff_calculator_;
};

#endif  // DQM_GEM_GEMEffByGEMCSCSegmentClient_h
