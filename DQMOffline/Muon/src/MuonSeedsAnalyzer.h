#ifndef MuonSeedsAnalyzer_H
#define MuonSeedsAnalyzer_H


/** \class MuonSeedsAnalyzer
 *
 *  DQM monitoring source for muon track seeds
 *
 *  $Date$
 *  $Revision$
 *  \author G. Mila - INFN Torino
 */


#include <memory>
#include <fstream>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/Muon/src/MuonAnalyzerBase.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class TrajectoryStateOnSurface;
class TrajectorySeed;
class MuonServiceProxy;

class MuonSeedsAnalyzer : public MuonAnalyzerBase {
 public:

  /// Constructor
  MuonSeedsAnalyzer(const edm::ParameterSet&, MuonServiceProxy *theService);
  
  /// Destructor
  virtual ~MuonSeedsAnalyzer();

  /// Inizialize parameters for histo binning
  void beginJob(edm::EventSetup const& iSetup, DQMStore * dbe);
  
  /// Get the analysis
  void analyze(const edm::Event&, const edm::EventSetup&, const TrajectorySeed& seed);
  
  /// Get the TrajectoryStateOnSurface
  TrajectoryStateOnSurface getSeedTSOS(const TrajectorySeed& seed);
  

  private:
  // ----------member data ---------------------------
  
  edm::ParameterSet parameters;
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
  
  int seedPxBin;
  double seedPxMin;
  double seedPxMax;
  
  int seedPyBin;
  double seedPyMin;
  double seedPyMax;
  
  int seedPzBin;
  double seedPzMin;
  double seedPzMax;

  int ptErrBin;
  double ptErrMin;
  double ptErrMax;

  int pxErrBin;
  double pxErrMin;
  double pxErrMax;

  int pyErrBin;
  double pyErrMin;
  double pyErrMax;

  int pzErrBin;
  double pzErrMin;
  double pzErrMax;

  int pErrBin;
  double pErrMin;
  double pErrMax;

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
  MonitorElement* seedPxErr;
  MonitorElement* seedPyErr;
  MonitorElement* seedPzErr;
  MonitorElement* seedPErr;
  MonitorElement* seedPhiErr;
  MonitorElement* seedEtaErr;

};
#endif  
