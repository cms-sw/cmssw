#ifndef DQM_TRACKINGMONITOR_V0MONITOR_H
#define DQM_TRACKINGMONITOR_V0MONITOR_H

#include <string>
#include <vector>
#include <map>

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "DataFormats/OnlineMetaData/interface/OnlineLuminosityRecord.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/V0Candidate/interface/V0Candidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

class GenericTriggerEventFlag;

struct MEbinning {
  int nbins;
  double xmin;
  double xmax;
};

//
// class declaration
//

class V0Monitor : public DQMEDAnalyzer {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  V0Monitor(const edm::ParameterSet&);
  ~V0Monitor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  MonitorElement* bookHisto1D(DQMStore::IBooker& ibooker,
                              std::string name,
                              std::string title,
                              std::string xaxis,
                              std::string yaxis,
                              MEbinning binning);
  MonitorElement* bookHisto2D(DQMStore::IBooker& ibooker,
                              std::string name,
                              std::string title,
                              std::string xaxis,
                              std::string yaxis,
                              MEbinning xbinning,
                              MEbinning ybinning);
  MonitorElement* bookProfile(DQMStore::IBooker& ibooker,
                              std::string name,
                              std::string title,
                              std::string xaxis,
                              std::string yaxis,
                              MEbinning xbinning,
                              MEbinning ybinning);
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;

private:
  void getHistoPSet(edm::ParameterSet pset, MEbinning& mebinning);

  const std::string folderName_;

  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> v0Token_;
  const edm::EDGetTokenT<reco::BeamSpot> bsToken_;
  const edm::EDGetTokenT<reco::VertexCollection> pvToken_;
  const edm::EDGetTokenT<LumiScalersCollection> lumiscalersToken_;
  const edm::EDGetTokenT<OnlineLuminosityRecord> metaDataToken_;

  const bool forceSCAL_;
  const int pvNDOF_;

  GenericTriggerEventFlag* genTriggerEventFlag_;

  MonitorElement* v0_N_;
  MonitorElement* v0_mass_;
  MonitorElement* v0_pt_;
  MonitorElement* v0_eta_;
  MonitorElement* v0_phi_;
  MonitorElement* v0_Lxy_;
  MonitorElement* v0_Lxy_wrtBS_;
  MonitorElement* v0_chi2oNDF_;
  MonitorElement* v0_mass_vs_p_;
  MonitorElement* v0_mass_vs_pt_;
  MonitorElement* v0_mass_vs_eta_;
  MonitorElement* v0_deltaMass_;
  MonitorElement* v0_deltaMass_vs_pt_;
  MonitorElement* v0_deltaMass_vs_eta_;

  MonitorElement* v0_Lxy_vs_deltaMass_;
  MonitorElement* v0_Lxy_vs_pt_;
  MonitorElement* v0_Lxy_vs_eta_;

  MonitorElement* n_vs_BX_;
  MonitorElement* v0_N_vs_BX_;
  MonitorElement* v0_mass_vs_BX_;
  MonitorElement* v0_Lxy_vs_BX_;
  MonitorElement* v0_deltaMass_vs_BX_;

  MonitorElement* n_vs_lumi_;
  MonitorElement* v0_N_vs_lumi_;
  MonitorElement* v0_mass_vs_lumi_;
  MonitorElement* v0_Lxy_vs_lumi_;
  MonitorElement* v0_deltaMass_vs_lumi_;

  MonitorElement* n_vs_PU_;
  MonitorElement* v0_N_vs_PU_;
  MonitorElement* v0_mass_vs_PU_;
  MonitorElement* v0_Lxy_vs_PU_;
  MonitorElement* v0_deltaMass_vs_PU_;

  MonitorElement* n_vs_LS_;
  MonitorElement* v0_N_vs_LS_;

  MEbinning mass_binning_;
  MEbinning pt_binning_;
  MEbinning eta_binning_;
  MEbinning Lxy_binning_;
  MEbinning chi2oNDF_binning_;
  MEbinning lumi_binning_;
  MEbinning pu_binning_;
  MEbinning ls_binning_;
};

#endif  // DQM_TRACKINGMONITOR_V0MONITOR_H
