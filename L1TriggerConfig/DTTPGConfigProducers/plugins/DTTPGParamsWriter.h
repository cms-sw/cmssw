#ifndef DTTPGParamsWriter_H
#define DTTPGParamsWriter_H

/* Program to write DT TPG pedestals correction into DB

 *  \author C. Battilana - CIEMAT
 */

#include "CondFormats/DTObjects/interface/DTTPGParameters.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include <fstream>
#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class DTChamberId;

class DTTPGParamsWriter : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  DTTPGParamsWriter(const edm::ParameterSet &pset);

  /// Destructor
  ~DTTPGParamsWriter() override;

  // Operations

  /// Compute the ttrig by fiting the TB rising edge
  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

  /// Write ttrig in the DB
  void endJob() override;

private:
  void pharseLine(std::string &line, DTChamberId &chId, float &fine, int &coarse);

  bool debug_;
  std::string inputFileName_;
  DTTPGParameters *phaseMap_;
};
#endif
