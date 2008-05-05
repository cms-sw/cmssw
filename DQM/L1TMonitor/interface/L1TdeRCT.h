// -*-C++-*-
#ifndef L1TdeRCT_H
#define L1TdeRCT_H

/*
 * \file L1TdeRCT.h
 *
 * Version 0.0. A.Savin 2008/04/26
 *
 * $Date: 2008/05/05 15:01:37 $
 * $Revision: 1.2 $
 * \author P. Wittich
 * $Id: L1TdeRCT.h,v 1.2 2008/05/05 15:01:37 asavin Exp $
 * $Log: L1TdeRCT.h,v $
 * Revision 1.2  2008/05/05 15:01:37  asavin
 * single channel histos are added
 *
 * Revision 1.4  2008/03/01 00:40:00  lat
 * DQM core migration.
 *
 * Revision 1.3  2007/09/03 15:14:42  wittich
 * updated RCT with more diagnostic and local coord histos
 *
 * Revision 1.2  2007/02/23 21:58:43  wittich
 * change getByType to getByLabel and add InputTag
 *
 * Revision 1.1  2007/02/19 22:49:53  wittich
 * - Add RCT monitor
 *
 *
 *
*/

// system include files
#include <memory>
#include <unistd.h>


#include <iostream>
#include <fstream>
#include <vector>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


// Trigger Headers



//
// class declaration
//

class L1TdeRCT : public edm::EDAnalyzer {

public:

// Constructor
  L1TdeRCT(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TdeRCT();

protected:
// Analyze
 void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginJob
 void beginJob(const edm::EventSetup& c);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;


  MonitorElement* rctIsoEmDataOcc_ ;
  MonitorElement* rctIsoEmEmulOcc_ ;
  MonitorElement* rctIsoEmEff1Occ_ ;
  MonitorElement* rctIsoEmEff2Occ_ ;
  MonitorElement* rctIsoEmIneffOcc_ ;
  MonitorElement* rctIsoEmOvereffOcc_ ; 
  MonitorElement* rctIsoEmEff1_ ;
  MonitorElement* rctIsoEmEff2_ ;
  MonitorElement* rctIsoEmIneff_ ;
  MonitorElement* rctIsoEmOvereff_ ;

  MonitorElement* rctIsoEmDataOcc1D_ ;
  MonitorElement* rctIsoEmEmulOcc1D_ ;
  MonitorElement* rctIsoEmEff1Occ1D_ ;
  MonitorElement* rctIsoEmEff2Occ1D_ ;
  MonitorElement* rctIsoEmIneffOcc1D_ ;
  MonitorElement* rctIsoEmOvereffOcc1D_ ; 
  MonitorElement* rctIsoEmEff1oneD_ ;
  MonitorElement* rctIsoEmEff2oneD_ ;
  MonitorElement* rctIsoEmIneff1D_ ;
  MonitorElement* rctIsoEmOvereff1D_ ;

  MonitorElement* rctNisoEmDataOcc_ ;
  MonitorElement* rctNisoEmEmulOcc_ ;
  MonitorElement* rctNisoEmEff1Occ_ ;
  MonitorElement* rctNisoEmEff2Occ_ ;
  MonitorElement* rctNisoEmIneffOcc_ ;
  MonitorElement* rctNisoEmOvereffOcc_ ;
  MonitorElement* rctNisoEmEff1_ ;
  MonitorElement* rctNisoEmEff2_ ;
  MonitorElement* rctNisoEmIneff_ ;
  MonitorElement* rctNisoEmOvereff_ ;

  MonitorElement* rctNisoEmDataOcc1D_ ;
  MonitorElement* rctNisoEmEmulOcc1D_ ;
  MonitorElement* rctNisoEmEff1Occ1D_ ;
  MonitorElement* rctNisoEmEff2Occ1D_ ;
  MonitorElement* rctNisoEmIneffOcc1D_ ;
  MonitorElement* rctNisoEmOvereffOcc1D_ ;
  MonitorElement* rctNisoEmEff1oneD_ ;
  MonitorElement* rctNisoEmEff2oneD_ ;
  MonitorElement* rctNisoEmIneff1D_ ;
  MonitorElement* rctNisoEmOvereff1D_ ;

  MonitorElement*  rctIsoEffChannel_[396] ;
  MonitorElement*  rctIsoIneffChannel_[396] ;
  MonitorElement*  rctIsoOvereffChannel_[396] ;

  MonitorElement*  rctNisoEffChannel_[396] ;
  MonitorElement*  rctNisoIneffChannel_[396] ;
  MonitorElement*  rctNisoOvereffChannel_[396] ;


  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool singlechannelhistos_;
  bool monitorDaemon_;
  ofstream logFile_;

  edm::InputTag rctSourceEmul_;
  edm::InputTag rctSourceData_;

protected:

void DivideME1D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result) ;
void DivideME2D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result) ;

};

#endif
