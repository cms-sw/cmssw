#ifndef PhotonOfflineClient_H
#define PhotonOfflineClient_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"


#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//

#include <vector>

/** \class PhotonOfflineClient
 **  
 **
 **  $Id: PhotonOfflineClient
 **  $Date: 2009/06/09 12:25:07 $ 
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **   Jamie Antonelli, U. of Notre Dame, US
 **     
 ***/


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;


class PhotonOfflineClient : public edm::EDAnalyzer
{

 public:
   
  //
  explicit PhotonOfflineClient( const edm::ParameterSet& pset ) ;
  virtual ~PhotonOfflineClient();
                                   
      
  virtual void analyze(const edm::Event&, const edm::EventSetup&  ) ;
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual void endJob() ;
  virtual void endLuminosityBlock( const edm::LuminosityBlock& , const edm::EventSetup& ) ;
 private:
  //

  MonitorElement*  h_filters_;
  MonitorElement* p_efficiencyVsEtaLoose_;
  MonitorElement* p_efficiencyVsEtLoose_;
  MonitorElement* p_efficiencyVsEtaTight_;
  MonitorElement* p_efficiencyVsEtTight_;
  MonitorElement* p_efficiencyVsEtaHLT_;
  MonitorElement* p_efficiencyVsEtHLT_;

  MonitorElement* p_convFractionVsEtaLoose_;
  MonitorElement* p_convFractionVsEtLoose_;
  MonitorElement* p_convFractionVsEtaTight_;
  MonitorElement* p_convFractionVsEtTight_;

  std::vector<MonitorElement*> p_convFractionVsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_convFractionVsEta_;
  std::vector<MonitorElement*> p_convFractionVsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_convFractionVsEt_;

  MonitorElement* p_vertexReconstructionEfficiencyVsEta_;

  std::vector<MonitorElement*> p_nTrackIsolSolidVsEta_isol_;
  std::vector<MonitorElement*> p_trackPtSumSolidVsEta_isol_;
  std::vector<MonitorElement*> p_nTrackIsolHollowVsEta_isol_;
  std::vector<MonitorElement*> p_trackPtSumHollowVsEta_isol_;
  std::vector<MonitorElement*> p_ecalSumVsEta_isol_;
  std::vector<MonitorElement*> p_hcalSumVsEta_isol_;

  std::vector<std::vector<MonitorElement*> > p_nTrackIsolSolidVsEta_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumSolidVsEta_;
  std::vector<std::vector<MonitorElement*> > p_nTrackIsolHollowVsEta_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumHollowVsEta_;
  std::vector<std::vector<MonitorElement*> > p_ecalSumVsEta_;
  std::vector<std::vector<MonitorElement*> > p_hcalSumVsEta_;

  std::vector<MonitorElement*> p_nTrackIsolSolidVsEt_isol_;
  std::vector<MonitorElement*> p_trackPtSumSolidVsEt_isol_;
  std::vector<MonitorElement*> p_nTrackIsolHollowVsEt_isol_;
  std::vector<MonitorElement*> p_trackPtSumHollowVsEt_isol_;
  std::vector<MonitorElement*> p_ecalSumVsEt_isol_;
  std::vector<MonitorElement*> p_hcalSumVsEt_isol_;

  std::vector<std::vector<MonitorElement*> > p_nTrackIsolSolidVsEt_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumSolidVsEt_;
  std::vector<std::vector<MonitorElement*> > p_nTrackIsolHollowVsEt_;
  std::vector<std::vector<MonitorElement*> > p_trackPtSumHollowVsEt_;
  std::vector<std::vector<MonitorElement*> > p_ecalSumVsEt_;
  std::vector<std::vector<MonitorElement*> > p_hcalSumVsEt_;

  std::vector<MonitorElement*> p_r9VsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_r9VsEt_;

  std::vector<MonitorElement*> p_e1x5VsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_e1x5VsEt_;

  std::vector<MonitorElement*> p_r9VsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_r9VsEta_;

  std::vector<MonitorElement*> p_e1x5VsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_e1x5VsEta_;

  std::vector<MonitorElement*> p_e2x5VsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_e2x5VsEt_;

  std::vector<MonitorElement*> p_e2x5VsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_e2x5VsEta_;

  std::vector<MonitorElement*> p_r1x5VsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_r1x5VsEt_;

  std::vector<MonitorElement*> p_r1x5VsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_r1x5VsEta_;

  std::vector<MonitorElement*> p_r2x5VsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_r2x5VsEt_;

  std::vector<MonitorElement*> p_r2x5VsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_r2x5VsEta_;

  std::vector<MonitorElement*> p_sigmaIetaIetaVsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_sigmaIetaIetaVsEta_;

  std::vector<MonitorElement*> p_sigmaEtaEtaVsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_sigmaEtaEtaVsEta_;

  std::vector<MonitorElement*> p_nHitsVsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_nHitsVsEta_;

  std::vector<MonitorElement*> p_tkChi2VsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_tkChi2VsEta_;

  std::vector<MonitorElement*> p_dCotTracksVsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_dCotTracksVsEta_;

  void doProfileX(TH2 * th2, MonitorElement* me);
  void doProfileX(MonitorElement * th2m, MonitorElement* me);
  void dividePlots(MonitorElement* dividend, MonitorElement* numerator, MonitorElement* denominator);
  void dividePlots(MonitorElement* dividend, MonitorElement* numerator, double denominator); 
      

  DQMStore *dbe_;
  int verbosity_;

  edm::ParameterSet parameters_;

  double cutStep_;
  int numberOfSteps_;
  double etMin;
  double etMax;
  int etBin;
  double etaMin;
  double etaMax;
  int etaBin;
  bool standAlone_;
  bool batch_;
  std::string outputFileName_;
  std::string inputFileName_;

  std::stringstream currentFolder_;
   
};





#endif
