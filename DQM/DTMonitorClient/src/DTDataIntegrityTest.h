#ifndef DTDataIntegrity_Test_H
#define DTDataIntegrity_Test_H

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  \author S. Bolognesi - INFN TO
 *
 *  threadsafe version (//-) oct/nov 2014 - WATWanAbdullah ncpp-um-my
 *
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <DQMServices/Core/interface/DQMEDHarvester.h>

class DQMStore;
class MonitorElement;
class DTReadOutMapping;

class DTDataIntegrityTest: public DQMEDHarvester{

public:

  /// Constructor
  DTDataIntegrityTest(const edm::ParameterSet& ps);

 /// Destructor
 ~DTDataIntegrityTest();

protected:

  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override;

  /// Get the ME name
  std::string getMEName(std::string histoType, int FEDId);

  /// Book the MEs
  void bookHistos(DQMStore::IBooker &, std::string histoType, int dduId);

  /// DQM Client Diagnostic
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, edm::LuminosityBlock const &, edm::EventSetup const &);

private:
  int readOutToGeometry(int dduId, int rosNumber, int& wheel, int& sector);

private:

  //Number of onUpdates
  int nupdates;

  // prescale on the # of LS to update the test
  int prescaleFactor;


  //Counter between 0 and nTimeBin
  int counter;

  int nevents;
  unsigned int nLumiSegs;

  int run;

  bool bookingdone;

  edm::ESHandle<DTReadOutMapping> mapping;
  
  // Monitor Elements
  std::map<std::string, std::map<int, MonitorElement*> > dduHistos;  
  std::map<std::string, std::map<int, std::vector <MonitorElement*> > > dduVectorHistos;

  MonitorElement *summaryHisto;
  MonitorElement *summaryTDCHisto;
  MonitorElement *glbSummaryHisto;
 };

#endif
