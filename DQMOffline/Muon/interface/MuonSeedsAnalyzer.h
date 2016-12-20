#ifndef MuonSeedsAnalyzer_H
#define MuonSeedsAnalyzer_H


/** \class MuonSeedsAnalyzer
 *
 *  DQM monitoring source for muon track seeds
 *
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

class TrajectoryStateOnSurface;
class TrajectorySeed;
class MuonServiceProxy;

class MuonSeedsAnalyzer : public  DQMEDAnalyzer {
 public:

  /// Constructor
  MuonSeedsAnalyzer(const edm::ParameterSet&);
  
  /// Destructor
  virtual ~MuonSeedsAnalyzer();

  void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  
  private:
  // ----------member data ---------------------------
  DQMStore *theDbe;
  MuonServiceProxy *theService;
  edm::ParameterSet parameters;

  // ------- Data ----
  edm::EDGetTokenT<TrajectorySeedCollection> theSeedsCollectionLabel_;

  // Switch for verbosity
  std::string metname;
  
  //histo binning parameters
  int seedHitBin;
  double seedHitMin;
  double seedHitMax;
  
  int PhiBin;
  double PhiMin;
  double PhiMax;
  
  int EtaBin;
  double EtaMin;
  double EtaMax;
  
  int ThetaBin;
  double ThetaMin;
  double ThetaMax;
  
  int Chi2Bin;
  double Chi2Min;
  double Chi2Max;
  
  int seedPtBin;
  double seedPtMin;
  double seedPtMax;
  
  int seedPxyzBin;
  double seedPxyzMin;
  double seedPxyzMax;
 
  int pErrBin;
  double pErrMin;
  double pErrMax;

  int pxyzErrBin;
  double pxyzErrMin;
  double pxyzErrMax;

  int phiErrBin;
  double phiErrMin;
  double phiErrMax;

  int etaErrBin;
  double etaErrMin;
  double etaErrMax;
  

  //the histos
  MonitorElement* NumberOfRecHitsPerSeed;
  MonitorElement* seedPhi;
  MonitorElement* seedEta;
  MonitorElement* seedTheta;
  MonitorElement* seedPt;
  MonitorElement* seedPx;
  MonitorElement* seedPy;
  MonitorElement* seedPz;
  MonitorElement* seedPtErr;
  MonitorElement* seedPtErrVsPhi;
  MonitorElement* seedPtErrVsEta;
  MonitorElement* seedPtErrVsPt;
  MonitorElement* seedPxErr;
  MonitorElement* seedPyErr;
  MonitorElement* seedPzErr;
  MonitorElement* seedPErr;
  MonitorElement* seedPErrVsPhi;
  MonitorElement* seedPErrVsEta;
  MonitorElement* seedPErrVsPt;
  MonitorElement* seedPhiErr;
  MonitorElement* seedEtaErr;

};
#endif  
