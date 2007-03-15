#ifndef DTDataIntegrity_Test_H
#define DTDataIntegrity_Test_H

/** \class DTDataIntegrityTest
 * *
 *  DQM Client to check the data integrity
 *
 *  $Date: 2007/03/15 18:33:33 $
 *  $Revision: 1.1 $
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
  /// Get the MEs
  void bookHistos(std::string histoType, int dduId);

private:

  bool debug;
  int nevents;

  DaqMonitorBEInterface* dbe;

  edm::ParameterSet parameters;

  // Monitor Elements
  // <histoType, <index , histo> >    
  std::map<std::string, std::map<int, MonitorElement*> > dduHistos;
 };

#endif
