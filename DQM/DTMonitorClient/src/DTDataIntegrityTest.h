#ifndef DTDataIntegrity_Test_H
#define DTDataIntegrity_Test_H

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  $Date: 2007/05/22 07:00:53 $
 *  $Revision: 1.3 $
 *  \author S. Bolognesi - INFN TO
 *   
 */
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/EventSetup.h>

class DaqMonitorBEInterface;
class MonitorElement;

class DTDataIntegrityTest: public edm::EDAnalyzer{

public:

  /// Constructor
  DTDataIntegrityTest(const edm::ParameterSet& ps);

 /// Destructor
 ~DTDataIntegrityTest();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);
 
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  /// Endjob
  void endJob();

  /// Get the ME name
  std::string getMEName(std::string histoType, int FEDId);
  /// Book the MEs
  void bookHistos(std::string histoType, int dduId);
  void bookTimeHistos(std::string histoType, int dduId, int evNumber);

private:

  //Number of onUpdates
  int nevents;
  
  //If you want info VS time histos
  bool doTimeHisto;
  //Number of bin in time histo
  int nTimeBin;
  //Counter between 0 and nTimeBin
  int counter;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  // Monitor Elements
  // <histoType, <DDU index , histo> >    
  std::map<std::string, std::map<int, MonitorElement*> > dduHistos;
  // <histoType, <DDU index , vector of histos> >    
  std::map<std::string, std::map<int, std::vector <MonitorElement*> > > dduVectorHistos;
 };

#endif
