#ifndef DTEfficiencyTask_H
#define DTEfficiencyTask_H


/** \class DTEfficiencyTask
 *  DQM Analysis of 4D DT segments, it produces plots about: <br>
 *      - single cell efficiency
 *  All histos are produced per Layer
 *
 *
 *  \author G. Mila - INFN Torino
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include <string>
#include <map>
#include <vector>

class DQMStore;
class MonitorElement;


class DTEfficiencyTask: public edm::EDAnalyzer{
public:
  /// Constructor
  DTEfficiencyTask(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTEfficiencyTask();

  /// BeginJob
  void beginJob();

  /// To reset the MEs
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) ;

  /// Endjob
  void endJob();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);

protected:


private:
  DQMStore* theDbe;

  // Switch for verbosity
  bool debug;

  // Lable of 4D segments in the event
  edm::EDGetTokenT<DTRecSegment4DCollection> recHits4DToken_;

  // Lable of 1D rechits in the event
  edm::EDGetTokenT<DTRecHitCollection> recHitToken_;

  edm::ParameterSet parameters;

  // Book a set of histograms for a give chamber
  void bookHistos(DTLayerId lId, int fisrtWire, int lastWire);

  // Fill a set of histograms for a given L
  void fillHistos(DTLayerId lId, int firstWire, int lastWire, int numWire);
  void fillHistos(DTLayerId lId, int firstWire, int lastWire, int missingWire, bool UnassHit);

  std::map<DTLayerId, std::vector<MonitorElement*> > histosPerL;

};
#endif


/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
