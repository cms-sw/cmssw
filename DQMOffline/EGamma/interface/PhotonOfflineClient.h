#ifndef PhotonOfflineClient_H
#define PhotonOfflineClient_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
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
 **  $Date: 2010/01/13 12:08:46 $ 
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
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void endLuminosityBlock( const edm::LuminosityBlock& , const edm::EventSetup& ) ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& ) ;
  virtual void runClient();

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
  std::vector<MonitorElement*> p_convFractionVsPhi_isol_;
  std::vector<std::vector<MonitorElement*> > p_convFractionVsPhi_;
  std::vector<MonitorElement*> p_convFractionVsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_convFractionVsEt_;

  std::vector<MonitorElement*> p_badChannelsFractionVsEta_isol_;
  std::vector<std::vector<MonitorElement*> > p_badChannelsFractionVsEta_;
  std::vector<MonitorElement*> p_badChannelsFractionVsPhi_isol_;
  std::vector<std::vector<MonitorElement*> > p_badChannelsFractionVsPhi_;
  std::vector<MonitorElement*> p_badChannelsFractionVsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_badChannelsFractionVsEt_;

  MonitorElement* p_vertexReconstructionEfficiencyVsEta_;


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
  double phiMin;
  double phiMax;
  int phiBin;
  bool standAlone_;
  bool batch_;
  std::string outputFileName_;
  std::string inputFileName_;

  std::stringstream currentFolder_;
   
};





#endif
