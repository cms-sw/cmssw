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
#include <string>

using std::string;
using std::vector;

/** \class PhotonOfflineClient
 **  
 **
 **  $Id: PhotonOfflineClient
 **  $Date: 2010/06/03 15:47:22 $ 
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

  MonitorElement* bookHisto(string histoName, string title, int bin, double min, double max);

  vector<vector<MonitorElement*> > book2DHistoVector(string histoType, string histoName, string title, 
							       int xbin, double xmin, double xmax,
							       int ybin=1,double ymin=1, double ymax=2);
  vector<vector<vector<MonitorElement*> > > book3DHistoVector(string histoType, string histoName, string title, 
							       int xbin, double xmin, double xmax,
							       int ybin=1,double ymin=1, double ymax=2);



  MonitorElement* retrieveHisto(string dir, string name);

 private:

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

  vector<vector<MonitorElement*> > p_convFractionVsEta_;
  vector<vector<vector<MonitorElement*> > > p_convFractionVsPhi_;
  vector<vector<MonitorElement*> > p_convFractionVsEt_;

  vector<vector<MonitorElement*> > p_badChannelsFractionVsEta_;
  vector<vector<MonitorElement*> > p_badChannelsFractionVsPhi_;
  vector<vector<MonitorElement*> > p_badChannelsFractionVsEt_;

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
  int    etBin;
  double etaMin;
  double etaMax;
  int    etaBin;
  double phiMin;
  double phiMax;
  int    phiBin;

  bool   standAlone_;
  bool   batch_;

  string outputFileName_;
  string inputFileName_;

  std::stringstream currentFolder_;

  int histo_index_photons_;
  int histo_index_conversions_;
  int histo_index_efficiency_;
  int histo_index_invMass_;
  
  vector<string> types_;
  vector<string> parts_;


};





#endif
