// -*-C++-*-
#ifndef L1TdeRCT_H
#define L1TdeRCT_H

/*
 * \file L1TdeRCT.h
 *
 * Version 0.0. A.Savin 2008/04/26
 *
 * $Date: 2012/03/29 12:17:55 $
 * $Revision: 1.16 $
 * \author P. Wittich
 * $Id: L1TdeRCT.h,v 1.16 2012/03/29 12:17:55 ghete Exp $
 * $Log: L1TdeRCT.h,v $
 * Revision 1.16  2012/03/29 12:17:55  ghete
 * Add filtering of events in analyze, to be able to remove the trigger type filter from L1 DQM sequence.
 *
 * Revision 1.15  2011/10/24 14:41:23  asavin
 * L1TdeRCT includes bit histos + cut of 2 GeV on EcalTPG hist
 *
 * Revision 1.14  2011/10/13 09:29:16  swanson
 * Added exper bit monitoring
 *
 * Revision 1.13  2010/09/30 22:26:45  bachtis
 * Add RCT FED vector monitoring
 *
 * Revision 1.12  2010/03/25 13:46:02  weinberg
 * removed quiet bit information
 *
 * Revision 1.11  2009/11/19 14:35:32  puigh
 * modify beginJob
 *
 * Revision 1.10  2009/10/11 21:12:58  asavin
 * *** empty log message ***
 *
 * Revision 1.9  2008/12/11 09:20:16  asavin
 * efficiency curves in L1TdeRCT
 *
 * Revision 1.8  2008/11/07 15:54:03  weinberg
 * Changed fine grain bit to HF plus tau bit
 *
 * Revision 1.7  2008/09/22 16:48:32  asavin
 * reg1D overeff added
 *
 * Revision 1.6  2008/07/25 13:06:48  weinberg
 * added GCT region/bit information
 *
 * Revision 1.5  2008/06/30 07:34:36  asavin
 * TPGs inculded in the RCT code
 *
 * Revision 1.4  2008/05/06 18:04:02  nuno
 * cruzet update
 *
 * Revision 1.3  2008/05/05 18:42:23  asavin
 * DataOcc added
 *
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
#include <bitset>


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
  void beginJob(void);

  //For FED vector monitoring 
  void beginRun(const edm::Run&, const edm::EventSetup&);
  void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  void readFEDVector(MonitorElement*,const edm::EventSetup&); 




// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  // begin GT decision information
  MonitorElement *triggerAlgoNumbers_;

  // trigger type information
  MonitorElement *triggerType_;

  // begin region information
  MonitorElement *rctRegDataOcc1D_;
  MonitorElement *rctRegEmulOcc1D_;
  MonitorElement *rctRegMatchedOcc1D_;
  MonitorElement *rctRegUnmatchedDataOcc1D_;
  MonitorElement *rctRegUnmatchedEmulOcc1D_;
  MonitorElement *rctRegSpEffOcc1D_;
  MonitorElement *rctRegSpIneffOcc1D_;

  MonitorElement *rctRegEff1D_;
  MonitorElement *rctRegIneff1D_;
  MonitorElement *rctRegOvereff1D_;
  MonitorElement *rctRegSpEff1D_;
  MonitorElement *rctRegSpIneff1D_;

  MonitorElement *rctRegDataOcc2D_;
  MonitorElement *rctRegEmulOcc2D_;
  MonitorElement *rctRegMatchedOcc2D_;
  MonitorElement *rctRegUnmatchedDataOcc2D_;
  MonitorElement *rctRegUnmatchedEmulOcc2D_;
//  MonitorElement *rctRegDeltaEt2D_;
  MonitorElement *rctRegSpEffOcc2D_;
  MonitorElement *rctRegSpIneffOcc2D_;

  MonitorElement *rctRegEff2D_;
  MonitorElement *rctRegIneff2D_;
  MonitorElement *rctRegOvereff2D_;
  MonitorElement *rctRegSpEff2D_;
  MonitorElement *rctRegSpIneff2D_;

  MonitorElement* rctRegBitOn_ ;
  MonitorElement* rctRegBitOff_ ;
  MonitorElement* rctRegBitDiff_ ;

  // end region information

  // begin bit information
  MonitorElement *rctBitEmulOverFlow2D_;
  MonitorElement *rctBitDataOverFlow2D_;
  MonitorElement *rctBitMatchedOverFlow2D_;
  MonitorElement *rctBitUnmatchedEmulOverFlow2D_;
  MonitorElement *rctBitUnmatchedDataOverFlow2D_;
  MonitorElement *rctBitOverFlowEff2D_;
  MonitorElement *rctBitOverFlowIneff2D_;
  MonitorElement *rctBitOverFlowOvereff2D_;
  MonitorElement *rctBitEmulTauVeto2D_;
  MonitorElement *rctBitDataTauVeto2D_;
  MonitorElement *rctBitMatchedTauVeto2D_;
  MonitorElement *rctBitUnmatchedEmulTauVeto2D_;
  MonitorElement *rctBitUnmatchedDataTauVeto2D_;
  MonitorElement *rctBitTauVetoEff2D_;
  MonitorElement *rctBitTauVetoIneff2D_;
  MonitorElement *rctBitTauVetoOvereff2D_;
  MonitorElement *rctBitEmulMip2D_;
  MonitorElement *rctBitDataMip2D_;
  MonitorElement *rctBitMatchedMip2D_;
  MonitorElement *rctBitUnmatchedEmulMip2D_;
  MonitorElement *rctBitUnmatchedDataMip2D_;
  MonitorElement *rctBitMipEff2D_;
  MonitorElement *rctBitMipIneff2D_;
  MonitorElement *rctBitMipOvereff2D_;
  MonitorElement *rctBitEmulQuiet2D_;
  MonitorElement *rctBitDataQuiet2D_;
  MonitorElement *rctBitMatchedQuiet2D_;
  MonitorElement *rctBitUnmatchedEmulQuiet2D_;
  MonitorElement *rctBitUnmatchedDataQuiet2D_;
  // QUIETBIT: To add quiet bit information, uncomment following 3 lines:
  // MonitorElement *rctBitQuietEff2D_;
  // MonitorElement *rctBitQuietIneff2D_;
  // MonitorElement *rctBitQuietOvereff2D_;
  MonitorElement *rctBitEmulHfPlusTau2D_;
  MonitorElement *rctBitDataHfPlusTau2D_;
  MonitorElement *rctBitMatchedHfPlusTau2D_;
  MonitorElement *rctBitUnmatchedEmulHfPlusTau2D_;
  MonitorElement *rctBitUnmatchedDataHfPlusTau2D_;
  MonitorElement *rctBitHfPlusTauEff2D_;
  MonitorElement *rctBitHfPlusTauIneff2D_;
  MonitorElement *rctBitHfPlusTauOvereff2D_;

  // end bit information

  MonitorElement* rctInputTPGEcalOcc_ ;
  MonitorElement* rctInputTPGEcalOccNoCut_ ;
  MonitorElement* rctInputTPGEcalRank_ ;
  MonitorElement* rctInputTPGHcalOcc_ ;
  MonitorElement* rctInputTPGHcalRank_ ;
  MonitorElement* rctInputTPGHcalSample_ ;

  MonitorElement* rctIsoEmDataOcc_ ;
  MonitorElement* rctIsoEmEmulOcc_ ;
  MonitorElement* rctIsoEmEff1Occ_ ;
  MonitorElement* rctIsoEmEff2Occ_ ;
  MonitorElement* rctIsoEmIneff2Occ_ ;
  MonitorElement* rctIsoEmIneffOcc_ ;
  MonitorElement* rctIsoEmOvereffOcc_ ;
  MonitorElement* rctIsoEmEff1_ ;
  MonitorElement* rctIsoEmEff2_ ;
  MonitorElement* rctIsoEmIneff2_ ;
  MonitorElement* rctIsoEmIneff_ ;
  MonitorElement* rctIsoEmOvereff_ ;

  MonitorElement* rctIsoEmDataOcc1D_ ;
  MonitorElement* rctIsoEmEmulOcc1D_ ;
  MonitorElement* rctIsoEmEff1Occ1D_ ;
  MonitorElement* rctIsoEmEff2Occ1D_ ;
  MonitorElement* rctIsoEmIneff2Occ1D_ ;
  MonitorElement* rctIsoEmIneffOcc1D_ ;
  MonitorElement* rctIsoEmOvereffOcc1D_ ;
  MonitorElement* rctIsoEmEff1oneD_ ;
  MonitorElement* rctIsoEmEff2oneD_ ;
  MonitorElement* rctIsoEmIneff2oneD_ ;
  MonitorElement* rctIsoEmIneff1D_ ;
  MonitorElement* rctIsoEmOvereff1D_ ;

  MonitorElement* rctIsoEmBitOn_ ;
  MonitorElement* rctIsoEmBitOff_ ;
  MonitorElement* rctIsoEmBitDiff_ ;

  MonitorElement* rctNisoEmDataOcc_ ;
  MonitorElement* rctNisoEmEmulOcc_ ;
  MonitorElement* rctNisoEmEff1Occ_ ;
  MonitorElement* rctNisoEmEff2Occ_ ;
  MonitorElement* rctNisoEmIneff2Occ_ ;
  MonitorElement* rctNisoEmIneffOcc_ ;
  MonitorElement* rctNisoEmOvereffOcc_ ;
  MonitorElement* rctNisoEmEff1_ ;
  MonitorElement* rctNisoEmEff2_ ;
  MonitorElement* rctNisoEmIneff2_ ;
  MonitorElement* rctNisoEmIneff_ ;
  MonitorElement* rctNisoEmOvereff_ ;

  MonitorElement* rctNisoEmDataOcc1D_ ;
  MonitorElement* rctNisoEmEmulOcc1D_ ;
  MonitorElement* rctNisoEmEff1Occ1D_ ;
  MonitorElement* rctNisoEmEff2Occ1D_ ;
  MonitorElement* rctNisoEmIneff2Occ1D_ ;
  MonitorElement* rctNisoEmIneffOcc1D_ ;
  MonitorElement* rctNisoEmOvereffOcc1D_ ;
  MonitorElement* rctNisoEmEff1oneD_ ;
  MonitorElement* rctNisoEmEff2oneD_ ;
  MonitorElement* rctNisoEmIneff2oneD_ ;
  MonitorElement* rctNisoEmIneff1D_ ;
  MonitorElement* rctNisoEmOvereff1D_ ;

  MonitorElement* rctNIsoEmBitOn_ ;
  MonitorElement* rctNIsoEmBitOff_ ;
  MonitorElement* rctNIsoEmBitDiff_ ;

  MonitorElement*  rctIsoEffChannel_[396] ;
  MonitorElement*  rctIsoIneffChannel_[396] ;
  MonitorElement*  rctIsoOvereffChannel_[396] ;

  MonitorElement*  rctNisoEffChannel_[396] ;
  MonitorElement*  rctNisoIneffChannel_[396] ;
  MonitorElement*  rctNisoOvereffChannel_[396] ;

  // begin region channel information
  MonitorElement* rctRegEffChannel_[396];
  MonitorElement* rctRegIneffChannel_[396];
  MonitorElement* rctRegOvereffChannel_[396];

  //efficiency
  MonitorElement* trigEffThresh_;
  MonitorElement* trigEffThreshOcc_;
  MonitorElement* trigEffTriggThreshOcc_;
  MonitorElement* trigEff_[396];
  MonitorElement* trigEffOcc_[396];
  MonitorElement* trigEffTriggOcc_[396];

  // end region channel information


  //begin fed vector information
  static const int crateFED[90];
  MonitorElement *fedVectorMonitorRUN_;
  MonitorElement *fedVectorMonitorLS_;
  ///////////////////////////////



  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  std::string histFolder_; // base dqm folder
  bool verbose_;
  bool singlechannelhistos_;
  bool monitorDaemon_;
  ofstream logFile_;

  edm::InputTag rctSourceEmul_;
  edm::InputTag rctSourceData_;
  edm::InputTag ecalTPGData_;
  edm::InputTag hcalTPGData_;
  edm::InputTag gtDigisLabel_;
  std::string gtEGAlgoName_; // name of algo to determine EG trigger threshold
  int doubleThreshold_; // value of ET at which to make 2-D eff plot

  /// filter TriggerType
  int filterTriggerType_;


  int trigCount,notrigCount;

protected:

void DivideME1D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result) ;
void DivideME2D(MonitorElement* numerator, MonitorElement* denominator, MonitorElement* result) ;

};

#endif
