#ifndef JPTJetAnalyzer_H
#define JPTJetAnalyzer_H

/** \class JPTJetAnalyzer
 *
 *  DQM monitoring source for JPT Jets
 *
 *  $Date: 2009/10/02 10:43:04 $
 *  $Revision: 1.1 $
 *  \author N. Cripps - Imperial
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/JetAnalyzerBase.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
// forward declare classes which do not need to be defined for interface
class DQMStore;
namespace reco {
  class CaloJet;
}
namespace jptJetAnalysis {
  class TrackPropagatorToCalo;
  class StripSignalOverNoiseCalculator;
}
namespace jpt {
  class MatchedTracks;
}
class JetPlusTrackCorrector;
class TrackingRecHit;
class SiStripRecHit2D;


/// JPT jet analyzer class definition
class JPTJetAnalyzer : public JetAnalyzerBase {
 public:
  /// Constructor
  JPTJetAnalyzer(const edm::ParameterSet& config);
  
  /// Destructor
  virtual ~JPTJetAnalyzer();
  
  /// Inizialize parameters for histo binning
  void beginJob(const edm::EventSetup& eventSetup, DQMStore* dqmStore);
  
  /// Do the analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup, const reco::CaloJet& jptCorrectedJet);
  
 private:
   
  // Helpper classes
  /// Helpper class to hold the configuration for a histogram
  struct HistogramConfig {
    bool enabled;
    unsigned int nBins;
    double min;
    double max;
    unsigned int nBinsY;
    double minY;
    double maxY;
    HistogramConfig();
    HistogramConfig(const unsigned int theNBins, const double theMin, const double theMax);
    HistogramConfig(const unsigned int theNBinsX, const double theMinX, const double theMaxX,
                    const unsigned int theNBinsY, const double theMinY, const double theMaxY);
  };
  /// Helpper class for grouping histograms belowing to a set of tracks
  struct TrackHistograms {
    MonitorElement* nTracksHisto;
    MonitorElement* ptHisto;
    MonitorElement* phiHisto;
    MonitorElement* etaHisto;
    MonitorElement* nHitsHisto;
    MonitorElement* ptVsEtaHisto;
    MonitorElement* trackDirectionJetDRHisto;
    MonitorElement* trackImpactPointJetDRHisto;
    TrackHistograms();
    TrackHistograms(MonitorElement* theNTracksHisto, MonitorElement* thePtHisto, MonitorElement* thePhiHisto, MonitorElement* theEtaHisto,
                    MonitorElement* theNHitsHisto, MonitorElement* thePtVsEtaHisto,
                    MonitorElement* theTrackDirectionJetDRHisto, MonitorElement* theTrackImpactPointJetDRHisto);
  };
  
  // Private methods
  /// Load the config for a hitogram
  void getConfigForHistogram(const std::string& configName, const edm::ParameterSet& psetContainingConfigPSet, std::ostringstream* pDebugStream = NULL);
  /// Load the configs for histograms associated with a set of tracks
  void getConfigForTrackHistograms(const std::string& tag, const edm::ParameterSet& psetContainingConfigPSet,std::ostringstream* pDebugStream = NULL);
  /// Book histograms and profiles
  MonitorElement* bookHistogram(const std::string& name, const std::string& title, const std::string& xAxisTitle, DQMStore* dqm);
  MonitorElement* bookProfile(const std::string& name, const std::string& title, const std::string& xAxisTitle, const std::string& yAxisTitle, DQMStore* dqm);
  MonitorElement* bookProfile2D(const std::string& name, const std::string& title,
                                const std::string& xAxisTitle, const std::string& yAxisTitle, const std::string zAxisTitle, DQMStore* dqm);
  /// Book all histograms
  void bookHistograms(DQMStore* dqm);
  /// Book the histograms for a track
  void bookTrackHistograms(TrackHistograms* histos, const std::string& tag, const std::string& titleTag,
                           MonitorElement* trackDirectionJetDRHisto, MonitorElement* trackImpactPointJetDRHisto, DQMStore* dqm);
  /// Fill histogram or profile if it has been booked
  void fillHistogram(MonitorElement* histogram, const double value);
  void fillHistogram(MonitorElement* histogram, const double valueX, const double valueY);
  void fillHistogram(MonitorElement* histogram, const double valueX, const double valueY, const double valueZ);
  /// Fill all track histograms
  void fillTrackHistograms(TrackHistograms& allTracksHistos, TrackHistograms& inCaloInVertexHistos,
                           TrackHistograms& inCaloOutVertexHistos, TrackHistograms& outCaloInVertexHistos,
                           const jpt::MatchedTracks& tracks, const reco::CaloJet& rawJet);
  void fillTrackHistograms(TrackHistograms& histos, const reco::TrackRefVector& tracks, const reco::CaloJet& rawJet);
  /// Fill the SoN hisotgram for hits on tracks
  void fillSiStripSoNForTracks(const reco::TrackRefVector& tracks);
  void fillSiStripHitSoN(const TrackingRecHit& hit);
  void fillSiStripHitSoNForSingleHit(const SiStripRecHit2D& hit);
  /// Utility function to calculate the fraction of track Pt in cone
  static double findPtFractionInCone(const reco::TrackRefVector& inConeTracks, const reco::TrackRefVector& outOfConeTracks);
  
  /// String constant for message logger category
  static const char* messageLoggerCatregory;
  
  // Config
  /// Path of directory used to store histograms in DQMStore
  const std::string histogramPath_;
  /// Create verbose debug messages
  const bool verbose_;
  /// Collection to jet original, completely un-corrected, jets from
  const edm::InputTag rawJetsSrc_;
  /// JPT corrector object name
  const std::string jptCorrectorName_;
  /// Histogram configuration (nBins etc)
  std::map<std::string,HistogramConfig> histogramConfig_;
  
  /// JPT Corrector
  const JetPlusTrackCorrector* jptCorrector_;
  
  /// Helpper object to propagate tracks to the calo surface
  jptJetAnalysis::TrackPropagatorToCalo* trackPropagator_;
  /// Helpper object to calculate strip SoN for tracks
  jptJetAnalysis::StripSignalOverNoiseCalculator* sOverNCalculator_;
  
  // Histograms
  MonitorElement *nTracksPerJetVsJetEtaPhiHisto_;
  MonitorElement *TrackSiStripHitStoNHisto_;
  MonitorElement *InCaloTrackDirectionJetDRHisto_, *OutCaloTrackDirectionJetDRHisto_;
  MonitorElement *InVertexTrackImpactPointJetDRHisto_, *OutVertexTrackImpactPointJetDRHisto_;
  MonitorElement *PtFractionInConeVsJetRawEtHisto_, *PtFractionInConeVsJetEtaHisto_;
  MonitorElement *CorrFactorVsJetEtHisto_, *CorrFactorVsJetEtaHisto_, *CorrFactorVsJetEtaPhiHisto_, *CorrFactorVsJetEtaEtHisto_;
  TrackHistograms allPionHistograms_, inCaloInVertexPionHistograms_, inCaloOutVertexPionHistograms_, outCaloInVertexPionHistograms_;
  TrackHistograms allMuonHistograms_, inCaloInVertexMuonHistograms_, inCaloOutVertexMuonHistograms_, outCaloInVertexMuonHistograms_;
  TrackHistograms allElectronHistograms_, inCaloInVertexElectronHistograms_, inCaloOutVertexElectronHistograms_, outCaloInVertexElectronHistograms_;
};

inline void JPTJetAnalyzer::fillHistogram(MonitorElement* histogram, const double value)
{
  if (histogram) histogram->Fill(value);
}

inline void JPTJetAnalyzer::fillHistogram(MonitorElement* histogram, const double valueX, const double valueY)
{
  if (histogram) histogram->Fill(valueX,valueY);
}

inline void JPTJetAnalyzer::fillHistogram(MonitorElement* histogram, const double valueX, const double valueY, const double valueZ)
{
  if (histogram) histogram->Fill(valueX,valueY,valueZ);
}

#endif
