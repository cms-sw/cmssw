#ifndef DQMClientExample_H
#define DQMClientExample_H

/** \class DQMClientExample
 * *
 *  DQM Test Client
 *
 *  $Date: 2007/08/29 13:49:00 $
 *  $Revision: 1.2 $
 *  \author  M. Zanetti CERN
 *   
 */


#include "DQMServices/Components/interface/DQMAnalyzer.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>


class DQMClientExample: public DQMAnalyzer{

public:

  /// Constructor
  DQMClientExample(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DQMClientExample();

protected:

  /// BeginJob
  void beginJob(const edm::EventSetup& c);

  /// BeginRun
  void beginRun(const edm::EventSetup& c);

  /// Fake Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;

  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                            const edm::EventSetup& context) ;

  /// DQM Client Diagnostic
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& r, const edm::EventSetup& c);

  /// Endjob
  void endJob();

private:

  MonitorElement * clientHisto;

};

#endif


