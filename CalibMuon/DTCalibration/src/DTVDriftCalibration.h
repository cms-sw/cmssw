#ifndef DTVDriftCalibration_H
#define DTVDriftCalibration_H

/** \class DTVDriftCalibration
 *  No description available.
 *
 *  $Date: $
 *  $Revision: $
 *  \author M. Giunta
 */


#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}


class DTVDriftCalibration : public edm::EDAnalyzer {
public:
  /// Constructor
  DTVDriftCalibration(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTVDriftCalibration();

  // Operations


  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  void endJob();

protected:

private:
  bool debug;

 // The label used to retrieve 4D segments from the event
 std::string  theRecHits4DLabel;
};
#endif

