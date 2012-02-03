#ifndef L1TdeDTTPGClient_H
#define L1TdeDTTPGClient_H

/*
 * \file L1TdeDTTPGClient.h
 *
 * $Date: 2010/11/18 09:42:52 $
 * $Revision: 1.0 $
 * \author C. Battilana - CIEMAT
 * \author M. Meneghelli - INFN BO
 *
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/MuonDetId/interface/DTChamberId.h"

#include <vector>
#include <string>
#include <map>

class L1TdeDTTPGClient: public edm::EDAnalyzer{

 public:
  
  /// Constructor
  L1TdeDTTPGClient(const edm::ParameterSet& parameterss );
  
  /// Destructor
  virtual ~L1TdeDTTPGClient();
  
 protected:
  
  // BeginJob
  void beginJob();

  ///Beginrun
  void beginRun(const edm::Run& run, const edm::EventSetup& context);

  ///Endrun
  void endRun(const edm::Run& run, const edm::EventSetup& context);

  /// Book the histograms
  void bookWheelHistos(int wh);
  
  /// Book the histograms
  void bookBarrelHistos();

  /// Analyze
  void analyze(const edm::Event& event, const edm::EventSetup& context);

  /// To count lumi
  void beginLuminosityBlock(const edm::LuminosityBlock& lumi, const edm::EventSetup& context) ;

  /// To perform summary plot generation Online
  void endLuminosityBlock(const edm::LuminosityBlock&  lumi, const edm::EventSetup& context);

  /// EndJob
  void endJob();

 private :
  
  /// Get the top folder
  std::string& topFolder() { return theBaseFolder; }

  float fracInRange(MonitorElement *me, int range);

  MonitorElement * getHisto(const DTChamberId & chId, std::string histoTag ) const;

  void performClientDiagnostic();

  float computeAgreement(MonitorElement *data, MonitorElement *emu);

 private:
  
  int theLumis;
  std::string theBaseFolder;
  DQMStore* theDQMStore;

  bool theRunOnline;

  float theHasBothTh;
  float theQualTh;
  float theStatQualTh;
  float thePhiTh;
  float thePhiBendTh;


  edm::ParameterSet theParams;
  std::map<int, std::map<std::string, MonitorElement*> > whHistos;
  std::map<std::string, MonitorElement*> barrelHistos;

};

#endif
