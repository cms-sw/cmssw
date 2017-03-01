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

#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//

#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooBreitWigner.h"
#include "RooDataHist.h"
#include "RooFitResult.h"

#include <vector>
#include <string>

/** \class PhotonOfflineClient
 **  
 **
 **  $Id: PhotonOfflineClient
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


class PhotonOfflineClient : public  DQMEDHarvester
{

 public:
   
  //
  explicit PhotonOfflineClient( const edm::ParameterSet& pset ) ;
  virtual ~PhotonOfflineClient();
                                   
      
  //  virtual void analyze(const edm::Event&, const edm::EventSetup&  ) ;
  // virtual void beginJob() ;
  //virtual void endJob() ;
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  //  virtual void endLuminosityBlock( const edm::LuminosityBlock& , const edm::EventSetup& ) ;
  //virtual void endRun(const edm::Run& , const edm::EventSetup& ) ;
  //virtual void runClient();
  
  virtual void runClient(DQMStore::IBooker& iBooker, DQMStore::IGetter& iGetter );
  MonitorElement* bookHisto(DQMStore::IBooker& iBooker,std::string histoName, std::string title, int bin, double min, double max);

  void book2DHistoVector(DQMStore::IBooker& iBooker,std::vector<std::vector<MonitorElement*> >& vecOfHist, 
			 std::string histoType, std::string histoName, std::string title, 
			 int xbin, double xmin, double xmax,
			 int ybin=1,double ymin=1, double ymax=2);
  void book3DHistoVector(DQMStore::IBooker& iBooker, std::vector<std::vector<std::vector<MonitorElement*> > >& vecOfHist, 
			 std::string histoType, std::string histoName, std::string title, 
			 int xbin, double xmin, double xmax,
			 int ybin=1,double ymin=1, double ymax=2);




  MonitorElement* retrieveHisto(DQMStore::IGetter& iGetter, std::string dir, std::string name);

 private:

  std::string analyzerName_;
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

  std::vector<std::vector<MonitorElement*> > p_convFractionVsEta_;
  std::vector<std::vector<std::vector<MonitorElement*> > > p_convFractionVsPhi_;
  std::vector<std::vector<MonitorElement*> > p_convFractionVsEt_;

  std::vector<std::vector<MonitorElement*> > p_badChannelsFractionVsEta_;
  std::vector<std::vector<MonitorElement*> > p_badChannelsFractionVsPhi_;
  std::vector<std::vector<MonitorElement*> > p_badChannelsFractionVsEt_;

  MonitorElement* p_vertexReconstructionEfficiencyVsEta_;
  
  void dividePlots(MonitorElement* dividend, MonitorElement* numerator, MonitorElement* denominator);
  void dividePlots(MonitorElement* dividend, MonitorElement* numerator, double denominator); 
  
  DQMStore *dbe_;
  int verbosity_;

  edm::ParameterSet parameters_;

  double cutStep_;
  int numberOfSteps_;
  bool minimalSetOfHistos_;
  bool excludeBkgHistos_;

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

  std::string outputFileName_;
  std::string inputFileName_;

  std::stringstream currentFolder_;

  int histo_index_photons_;
  int histo_index_conversions_;
  int histo_index_efficiency_;
  int histo_index_invMass_;
  
  std::vector<std::string> types_;
  std::vector<std::string> parts_;


};





#endif
