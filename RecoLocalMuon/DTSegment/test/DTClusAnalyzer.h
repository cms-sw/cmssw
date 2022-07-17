#ifndef DTCLUSANALYZER_H
#define DTCLUSANALYZER_H

/** \class DTClusAnalyzer
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 *
 * Modification:
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/Handle.h"
class TFile;
class TH1F;
class TH2F;

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecClusterCollection.h"

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/* Class DTClusAnalyzer Interface */

class DTClusAnalyzer : public edm::one::EDAnalyzer<> {
public:
  /* Constructor */
  DTClusAnalyzer(const edm::ParameterSet& pset);

  /* Destructor */
  ~DTClusAnalyzer();

  /* Operations */
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);

private:
  TH1F* histo(const std::string& name) const;
  TH2F* histo2d(const std::string& name) const;

private:
  bool debug;
  int _ev;
  std::string theRootFileName;
  TFile* theFile;

  std::string theRecClusLabel;
  std::string theRecHits2DLabel;
  std::string theRecHits1DLabel;

  edm::ESGetToken<DTGeometry, MuonGeometryRecord> theDtGeomToken;

  edm::EDGetTokenT<DTRecClusterCollection> theRecClusToken;
  edm::EDGetTokenT<DTRecHitCollection> theRecHits1DToken;
  edm::EDGetTokenT<DTRecSegment2DCollection> theRecHits2DToken;

protected:
};
#endif  // DTCLUSANALYZER_H
