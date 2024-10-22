#ifndef DTANALYZER_H
#define DTANALYZER_H

/** \class DTAnalyzerDetailed
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN Padova <stefano.lacaprara@pd.infn.it>
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
class DTLayerId;
class DTSuperLayerId;
class DTChamberId;
class DTTTrigBaseSync;
class MuonGeometryRecord;
class DTGeometry;

/* C++ Headers */
#include <iosfwd>
#include <bitset>

/* ====================================================================== */

/* Class DTAnalyzerDetailed Interface */

class DTAnalyzerDetailed : public edm::one::EDAnalyzer<> {
public:
  /* Constructor */
  DTAnalyzerDetailed(const edm::ParameterSet& pset);

  /* Destructor */
  ~DTAnalyzerDetailed();

  /* Operations */
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);

private:
  void analyzeDTHits(const edm::Event& event, const edm::EventSetup& eventSetup);
  void analyzeDTSegments(const edm::Event& event, const edm::EventSetup& eventSetup);

  TH1F* histo(const std::string& name) const;
  TH2F* histo2d(const std::string& name) const;

  void createTH1F(const std::string& name,
                  const std::string& title,
                  const std::string& suffix,
                  int nbin,
                  const double& binMin,
                  const double& binMax) const;

  void createTH2F(const std::string& name,
                  const std::string& title,
                  const std::string& suffix,
                  int nBinX,
                  const double& binXMin,
                  const double& binXMax,
                  int nBinY,
                  const double& binYMin,
                  const double& binYMax) const;

  std::string toString(const DTLayerId& id) const;
  std::string toString(const DTSuperLayerId& id) const;
  std::string toString(const DTChamberId& id) const;
  template <class T>
  std::string hName(const std::string& s, const T& id) const;

private:
  bool debug;
  int _ev;
  std::string theRootFileName;
  TFile* theFile;
  //static std::string theAlgoName;
  std::string theRecHits4DLabel;
  std::string theRecHits2DLabel;
  std::string theRecHits1DLabel;

  bool doHits;
  bool doSegs;

  std::unique_ptr<DTTTrigBaseSync> theSync;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> theDTGeomToken;
};
#endif  // DTANALYZER_H
