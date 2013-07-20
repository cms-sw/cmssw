#ifndef DTDataIntegrity_Test_H
#define DTDataIntegrity_Test_H

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  $Date: 2010/03/15 09:45:52 $
 *  $Revision: 1.14 $
 *  \author S. Bolognesi - INFN TO
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/ESHandle.h"
#include <FWCore/Framework/interface/EventSetup.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>

class DQMStore;
class MonitorElement;
class DTReadOutMapping;

class DTDataIntegrityTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDataIntegrityTest(const edm::ParameterSet& ps);

 /// Destructor
 ~DTDataIntegrityTest();

protected:

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// Get the ME name
  std::string getMEName(std::string histoType, int FEDId);
  /// Book the MEs
  void bookHistos(std::string histoType, int dduId);

  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c);

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


  DQMStore* dbe;
  edm::ESHandle<DTReadOutMapping> mapping;
  

  // Monitor Elements
  // <histoType, <DDU index , histo> >    
  std::map<std::string, std::map<int, MonitorElement*> > dduHistos;
  // <histoType, <DDU index , vector of histos> >    
  std::map<std::string, std::map<int, std::vector <MonitorElement*> > > dduVectorHistos;

  MonitorElement *summaryHisto;
  MonitorElement *summaryTDCHisto;
  MonitorElement *glbSummaryHisto;
 };

#endif
