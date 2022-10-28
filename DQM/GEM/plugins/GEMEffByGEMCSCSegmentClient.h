#ifndef DQM_GEM_GEMEffByGEMCSCSegmentClient_h
#define DQM_GEM_GEMEffByGEMCSCSegmentClient_h

/** \class GEMEffByGEMCSCSegmentClient
 * 
 * `GEMEffByGEMCSCSegmentSource` measures the efficiency of GE11-L1(2) using GE11-L2(1) and ME11 as trigger detectors.
 * See https://github.com/cms-sw/cmssw/blob/CMSSW_12_3_0_pre5/RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegAlgoRR.cc
 *
 * \author Seungjin Yang <seungjin.yang@cern.ch>
 */

#include "DQM/GEM/interface/GEMDQMEfficiencyClientBase.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class GEMEffByGEMCSCSegmentClient : public GEMDQMEfficiencyClientBase {
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

  const std::string kFolder_;
};

#endif  // DQM_GEM_GEMEffByGEMCSCSegmentClient_h
