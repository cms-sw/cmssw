#ifndef DTTPGParamsWriter_H
#define DTTPGParamsWriter_H

/* Program to write DT TPG pedestals correction into DB
 
 *  $Date: 2010/11/12 11:04:40 $
 *  $Revision: 1.1 $
 *  \author C. Battilana - CIEMAT
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include <string>
#include <fstream>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTChamberId;

class DTTPGParamsWriter : public edm::EDAnalyzer {
public:
  /// Constructor
  DTTPGParamsWriter(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTPGParamsWriter();

  // Operations

  /// Compute the ttrig by fiting the TB rising edge
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  /// Write ttrig in the DB
  void endJob();

private:

  void pharseLine(std::string &line, DTChamberId& chId, float &fine, int &coarse);

  bool debug_;
  std::string inputFileName_;
  DTTPGParameters *phaseMap_;
};
#endif
