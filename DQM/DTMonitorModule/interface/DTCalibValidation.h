#ifndef DTCalibValidation_H
#define DTCalibValidation_H

/** \class DTCalibValidation
 *  Analysis on DT residuals to validate the kFactor
 *
 *
 *  $Date: 2010/06/22 19:10:06 $
 *  $Revision: 1.7 $
 *  \author G. Mila - INFN Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/LuminosityBlock.h>

#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include <FWCore/Framework/interface/ESHandle.h>

#include <string>
#include <map>
#include <vector>


// To remove into CMSSW versions before 20X
class DQMStore;
// To add into CMSSW versions before 20X
//class DaqMonitorBEInterface;

class MonitorElement;
class DTGeometry;
class DTChamber;


class DTCalibValidation: public edm::EDAnalyzer{
 public:
  /// Constructor
  DTCalibValidation(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTCalibValidation();

  /// BeginJob
  void beginJob();

  /// BeginRun
  void beginRun(const edm::Run&, const edm::EventSetup&);

  /// Endjob
  void endJob();

  // Operations
  void analyze(const edm::Event& event, const edm::EventSetup& setup);


 protected:

 private:

  // To remove into CMSSW versions before 20X
  DQMStore* theDbe;
  // To add into CMSSW versions before 20X
  //DaqMonitorBEInterface* theDbe;

  // Switch for verbosity
  //bool debug;
  edm::ParameterSet parameters;
  int wrongSegment;
  int rightSegment;
  int nevent;
  // the analysis type
  bool detailedAnalysis;
  // the geometry
  edm::ESHandle<DTGeometry> dtGeom;

  // Lable of 1D rechits in the event
  std::string recHits1DLabel;
  // Lable of 2D segments in the event
  std::string segment2DLabel;
  // Lable of 4D segments in the event
  std::string segment4DLabel;

  // Return a map between DTRecHit1DPair and wireId
  std::map<DTWireId, std::vector<DTRecHit1DPair> >
    map1DRecHitsPerWire(const DTRecHitCollection* dt1DRecHitPairs);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
    map1DRecHitsPerWire(const DTRecSegment2DCollection* segment2Ds);

  // Return a map between DTRecHit1D and wireId
  std::map<DTWireId, std::vector<DTRecHit1D> >
    map1DRecHitsPerWire(const DTRecSegment4DCollection* segment4Ds);

  template  <typename type>
  const type* 
  findBestRecHit(const DTLayer* layer,
		 DTWireId wireId,
		 const std::vector<type>& recHits,
		 const float simHitDist);

  // Compute the distance from wire (cm) of a hits in a DTRecHit1DPair
  float recHitDistFromWire(const DTRecHit1DPair& hitPair, const DTLayer* layer);
  // Compute the distance from wire (cm) of a hits in a DTRecHit1D
  float recHitDistFromWire(const DTRecHit1D& recHit, const DTLayer* layer);
  // Compute the position with respect to the wire (cm) of a hits in a DTRecHit1DPair
  float recHitPosition(const DTRecHit1DPair& hitPair, const DTLayer* layer, const DTChamber* chamber, float segmPos, int sl);
  // Compute the position with respect to the wire (cm) of a hits in a DTRecHit1D
  float recHitPosition(const DTRecHit1D& recHit, const DTLayer* layer, const DTChamber* chamber, float segmPos, int sl);
  
  // Does the real job
  template  <typename type>
    void compute(const DTGeometry *dtGeom,
		 const DTRecSegment4D& segment,
	       std::map<DTWireId, std::vector<type> > recHitsPerWire,
		 int step);

  // Book a set of histograms for a give chamber
  void bookHistos(DTSuperLayerId slId, int step);
  // Fill a set of histograms for a give chamber 
  void fillHistos(DTSuperLayerId slId,
		  float distance,
		  float residualOnDistance,
		  float position,
		  float residualOnPosition,
		  int step);

  std::map<std::pair<DTSuperLayerId,int>, std::vector<MonitorElement*> > histosPerSL;

};
#endif



