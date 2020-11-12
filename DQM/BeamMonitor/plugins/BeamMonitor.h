#ifndef BeamMonitor_H
#define BeamMonitor_H

/** \class BeamMonitor
 * *
 *  \author  Geng-yuan Jeng/UC Riverside
 *           Francisco Yumiceva/FNAL
 *
 */
// C++
#include <string>
#include <memory>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "RecoVertex/BeamSpotProducer/interface/BSTrkParameters.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"
#include "CondCore/DBOutputService/interface/OnlineDBOutputService.h"
#include <fstream>

//
// class declaration
//

class BeamMonitor : public DQMOneEDAnalyzer<edm::one::WatchLuminosityBlocks> {
public:
  BeamMonitor(const edm::ParameterSet&);

protected:
  // BeginRun
  void bookHistograms(DQMStore::IBooker& i, const edm::Run& r, const edm::EventSetup& c) override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& context) override;

  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c) override;
  // EndRun
  void dqmEndRun(const edm::Run& r, const edm::EventSetup& c) override;

private:
  void FitAndFill(const edm::LuminosityBlock& lumiSeg, int&, int&, int&);
  void RestartFitting();
  void scrollTH1(TH1*, std::time_t);
  bool testScroll(std::time_t&, std::time_t&);
  void formatFitTime(char*, const std::time_t&);
  const int dxBin_;
  const double dxMin_;
  const double dxMax_;

  const int vxBin_;
  const double vxMin_;
  const double vxMax_;

  const int phiBin_;
  const double phiMin_;
  const double phiMax_;

  const int dzBin_;
  const double dzMin_;
  const double dzMax_;
  std::string monitorName_;
  std::string recordName_;                  // output BeamSpotOnline Record name
  edm::EDGetTokenT<reco::BeamSpot> bsSrc_;  // beam spot
  edm::EDGetTokenT<reco::TrackCollection> tracksLabel_;
  edm::EDGetTokenT<reco::VertexCollection> pvSrc_;                      // primary vertex
  edm::EDGetTokenT<edm::TriggerResults> hltSrc_;                        //hlt collection
  edm::Service<cond::service::OnlineDBOutputService> onlineDbService_;  // DB service
  int DBloggerReturn_;                                                  // DB logger return value

  int fitNLumi_;
  int fitPVNLumi_;
  int resetFitNLumi_;
  int resetPVNLumi_;
  int intervalInSec_;
  bool debug_;
  bool onlineMode_;
  std::vector<std::string> jetTrigger_;

  std::unique_ptr<BeamFitter> theBeamFitter;

  int countEvt_;   //counter
  int countLumi_;  //counter
  int beginLumiOfBSFit_;
  int endLumiOfBSFit_;
  int beginLumiOfPVFit_;
  int endLumiOfPVFit_;
  int lastlumi_;  // previous LS processed
  int nextlumi_;  // next LS of Fit
  std::time_t refBStime[2];
  std::time_t refPVtime[2];
  unsigned int nthBSTrk_;
  int nFitElements_;
  int nFits_;
  double deltaSigCut_;
  unsigned int min_Ntrks_;
  double maxZ_;
  unsigned int minNrVertices_;
  double minVtxNdf_;
  double minVtxWgt_;

  bool resetHistos_;
  bool StartAverage_;
  int firstAverageFit_;
  int countGapLumi_;

  bool processed_;

  // ----------member data ---------------------------

  //   std::vector<BSTrkParameters> fBSvector;
  reco::BeamSpot refBS;
  reco::BeamSpot preBS;

  // MonitorElements:
  MonitorElement* h_nTrk_lumi;
  MonitorElement* h_nVtx_lumi;
  MonitorElement* h_nVtx_lumi_all;
  MonitorElement* h_d0_phi0;
  MonitorElement* h_trk_z0;
  MonitorElement* h_vx_vy;
  MonitorElement* h_vx_dz;
  MonitorElement* h_vy_dz;
  MonitorElement* h_trkPt;
  MonitorElement* h_trkVz;
  MonitorElement* fitResults;
  MonitorElement* h_x0;
  MonitorElement* h_y0;
  MonitorElement* h_z0;
  MonitorElement* h_sigmaX0;
  MonitorElement* h_sigmaY0;
  MonitorElement* h_sigmaZ0;
  MonitorElement* h_nVtx;
  MonitorElement* h_nVtx_st;
  MonitorElement* h_PVx[2];
  MonitorElement* h_PVy[2];
  MonitorElement* h_PVz[2];
  MonitorElement* h_PVxz;
  MonitorElement* h_PVyz;
  MonitorElement* pvResults;
  std::vector<MonitorElement*> hs;

  // The histo of the primary vertex for  DQM gui
  std::map<int, std::vector<float> > mapPVx, mapPVy, mapPVz;
  std::map<int, std::vector<int> > mapNPV;
  //keep track of beginLuminosity block and time
  std::map<int, int> mapBeginBSLS, mapBeginPVLS;
  std::map<int, std::time_t> mapBeginBSTime, mapBeginPVTime;
  //these maps are needed to keep track of size of theBSvector and pvStore
  std::map<int, std::size_t> mapLSBSTrkSize;
  std::map<int, size_t> mapLSPVStoreSize;
  //to correct the cutFlot Table
  std::map<int, TH1F> mapLSCF;

  // Summary:
  Float_t reportSummary_;
  Float_t summarySum_;
  Float_t summaryContent_[3];
  MonitorElement* reportSummary;
  MonitorElement* reportSummaryContents[3];
  MonitorElement* reportSummaryMap;
  MonitorElement* cutFlowTable;
  // variables for beam fit

  //
  std::time_t tmpTime;
  std::time_t startTime;
  std::time_t refTime;
  edm::TimeValue_t ftimestamp;
  // Current run
  int frun;
  int lastNZbin;  // last non zero bin of time histos
};

#endif

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
