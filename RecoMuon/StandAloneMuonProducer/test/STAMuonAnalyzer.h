#ifndef RecoMuon_StandAloneMuonProducer_STAMuonAnalyzer_H
#define RecoMuon_StandAloneMuonProducer_STAMuonAnalyzer_H

/** \class STAMuonAnalyzer
 *  Analyzer of the StandAlone muon tracks
 *
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

// Base Class Headers
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class TFile;
class TH1F;
class TH2F;
class MagneticField;
class IdealMagneticFieldRecord;
class GlobalTrackingGeometry;
class GlobalTrackingGeometryRecord;

class STAMuonAnalyzer : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  STAMuonAnalyzer(const edm::ParameterSet &pset);

  /// Destructor
  ~STAMuonAnalyzer() override;

  // Operations

  void analyze(const edm::Event &event, const edm::EventSetup &eventSetup) override;

  void beginJob() override;
  void endJob() override;

protected:
private:
  std::string theRootFileName;
  TFile *theFile;

  std::string theSTAMuonLabel;
  std::string theSeedCollectionLabel;

  // Histograms
  TH1F *hPtRec;
  TH1F *hPtSim;
  TH1F *hPres;
  TH1F *h1_Pres;
  TH1F *hPTDiff;
  TH1F *hPTDiff2;
  TH2F *hPTDiffvsEta;
  TH2F *hPTDiffvsPhi;

  // Counters
  int numberOfSimTracks;
  int numberOfRecTracks;

  std::string theDataType;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  edm::ESGetToken<GlobalTrackingGeometry, GlobalTrackingGeometryRecord> theGeomToken;
};
#endif
