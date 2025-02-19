#ifndef JPTJetAnalyzer_H
#define JPTJetAnalyzer_H

/** \class JPTJetAnalyzer
 *
 *  DQM monitoring source for JPT Jets
 *
 *  $Date: 2012/09/24 09:39:53 $
 *  $Revision: 1.14 $
 *  \author N. Cripps - Imperial
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMOffline/JetMET/interface/JetAnalyzerBase.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include <memory>


#include "GlobalVariables.h"


// forward declare classes which do not need to be defined for interface
class DQMStore;
namespace reco {
  namespace helper {
    class JetIDHelper;
  }
}
namespace jptJetAnalysis {
  class TrackPropagatorToCalo;
  class StripSignalOverNoiseCalculator;
}
namespace jpt {
  class MatchedTracks;
}
class JetPlusTrackCorrector;
class JetCorrector;
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
  void beginJob(DQMStore * dbe);
  
  /// Do the analysis
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup, const reco::JPTJet& jptJet, double& pt1, double& pt2, double& pt3, const int numPV);
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup, const reco::JPTJetCollection& jptJets, const int numPV);
  
  /// Finish up a job
  virtual void endJob();
  
 private:
   
  // Helper classes
  /// Helper class to hold the configuration for a histogram
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
  /// Helper class for grouping histograms belowing to a set of tracks
  struct TrackHistograms {
    MonitorElement* nTracksHisto;
    MonitorElement* ptHisto;
    MonitorElement* phiHisto;
    MonitorElement* etaHisto;
    /*    MonitorElement* nHitsHisto;
	  MonitorElement* nLayersHisto;  */
    MonitorElement* ptVsEtaHisto;
    /*    MonitorElement* dzHisto;
    MonitorElement* dxyHisto;  */ 
    MonitorElement* trackDirectionJetDRHisto;
    MonitorElement* trackImpactPointJetDRHisto;
    TrackHistograms();
    TrackHistograms(MonitorElement* theNTracksHisto, MonitorElement* thePtHisto, MonitorElement* thePhiHisto, MonitorElement* theEtaHisto, 
		    /* MonitorElement* theNHitsHisto, MonitorElement* theNLayersHisto, */ 
MonitorElement* thePtVsEtaHisto, 
		    /* MonitorElement* dzHisto, MonitorElement* dxyHisto,  */
MonitorElement* theTrackDirectionJetDRHisto, MonitorElement* theTrackImpactPointJetDRHisto);
  };
  
  // Private methods
  /// Load the config for a hitogram
  void getConfigForHistogram(const std::string& configName, const edm::ParameterSet& psetContainingConfigPSet, std::ostringstream* pDebugStream = NULL);
  /// Load the configs for histograms associated with a set of tracks
  void getConfigForTrackHistograms(const std::string& tag, const edm::ParameterSet& psetContainingConfigPSet,std::ostringstream* pDebugStream = NULL);
  /// Book histograms and profiles
  MonitorElement* bookHistogram(const std::string& name, const std::string& title, const std::string& xAxisTitle, DQMStore* dqm);
  MonitorElement* book2DHistogram(const std::string& name, const std::string& title, const std::string& xAxisTitle, const std::string& yAxisTitle, DQMStore* dqm);
  MonitorElement* bookProfile(const std::string& name, const std::string& title, const std::string& xAxisTitle, const std::string& yAxisTitle, DQMStore* dqm);
  /// Book all histograms
  void bookHistograms(DQMStore* dqm);
  /// Book the histograms for a track
  void bookTrackHistograms(TrackHistograms* histos, const std::string& tag, const std::string& titleTag,
                           MonitorElement* trackDirectionJetDRHisto, MonitorElement* trackImpactPointJetDRHisto, DQMStore* dqm);
  /// Fill histogram or profile if it has been booked
  void fillHistogram(MonitorElement* histogram, const double value);
  void fillHistogram(MonitorElement* histogram, const double valueX, const double valueY);
  /// Fill all track histograms
  void fillTrackHistograms(TrackHistograms& allTracksHistos, TrackHistograms& inCaloInVertexHistos,
                           TrackHistograms& inCaloOutVertexHistos, TrackHistograms& outCaloInVertexHistos,
                           const reco::TrackRefVector& inVertexInCalo,
                           const reco::TrackRefVector& outVertexInCalo,
                           const reco::TrackRefVector& inVertexOutCalo,
                           const reco::Jet& rawJet);
  void fillTrackHistograms(TrackHistograms& histos, const reco::TrackRefVector& tracks, const reco::Jet& rawJet);
  /// Fill the SoN hisotgram for hits on tracks
  void fillSiStripSoNForTracks(const reco::TrackRefVector& tracks);
  void fillSiStripHitSoN(const TrackingRecHit& hit);

  // J.Piedra, 2012/09/24
  //  void fillSiStripHitSoNForSingleHit(const SiStripRecHit2D& hit);

  /// Utility function to calculate the fraction of track Pt in cone
  static double findPtFractionInCone(const reco::TrackRefVector& inConeTracks, const reco::TrackRefVector& outOfConeTracks);
  
  /// String constant for message logger category
  static const char* messageLoggerCatregory;
  
  // Config
  /// Path of directory used to store histograms in DQMStore
  const std::string histogramPath_;
  /// Create verbose debug messages
  const bool verbose_;
  /// Histogram configuration (nBins etc)
  std::map<std::string,HistogramConfig> histogramConfig_;
  
  /// Write DQM store to a file?
  const bool writeDQMStore_;
  /// DQM store file name
  std::string dqmStoreFileName_;
  
  /// Jet ID cuts
  const int n90HitsMin_;
  const double fHPDMax_;  
  const double resEMFMin_;
  const double correctedPtMin_;
  
  /// Helper object to propagate tracks to the calo surface
  std::auto_ptr<jptJetAnalysis::TrackPropagatorToCalo> trackPropagator_;
  /// Helper object to calculate strip SoN for tracks
  std::auto_ptr<jptJetAnalysis::StripSignalOverNoiseCalculator> sOverNCalculator_;
  /// Helper object to calculate jet ID parameters
  std::auto_ptr<reco::helper::JetIDHelper> jetID_;
  
  // Histograms
  MonitorElement *JetE_, *JetEt_, *JetP_,  *JetPt_;
  /*  MonitorElement *JetMass_   */
  MonitorElement *JetPt1_, *JetPt2_, *JetPt3_;
  MonitorElement *JetPx_, *JetPy_, *JetPz_;
  MonitorElement *JetEta_, *JetPhi_, *JetDeltaEta_, *JetDeltaPhi_, *JetPhiVsEta_;
  /*  MonitorElement *JetN90Hits_ 
      MonitorElement *JetfHPD_, *JetResEMF_, *JetfRBX_; 
  MonitorElement *TrackSiStripHitStoNHisto_;  */
  MonitorElement *InCaloTrackDirectionJetDRHisto_, *OutCaloTrackDirectionJetDRHisto_;
  MonitorElement *InVertexTrackImpactPointJetDRHisto_, *OutVertexTrackImpactPointJetDRHisto_;
  MonitorElement *NTracksPerJetHisto_, *NTracksPerJetVsJetEtHisto_, *NTracksPerJetVsJetEtaHisto_;
  /*  MonitorElement *PtFractionInConeHisto_, *PtFractionInConeVsJetRawEtHisto_, *PtFractionInConeVsJetEtaHisto_;
  MonitorElement *CorrFactorHisto_, *CorrFactorVsJetEtHisto_, *CorrFactorVsJetEtaHisto_;
  MonitorElement *ZSPCorrFactorHisto_, *ZSPCorrFactorVsJetEtHisto_, *ZSPCorrFactorVsJetEtaHisto_;
  MonitorElement *JPTCorrFactorHisto_, *JPTCorrFactorVsJetEtHisto_, *JPTCorrFactorVsJetEtaHisto_;  */ 
  TrackHistograms allPionHistograms_, inCaloInVertexPionHistograms_, inCaloOutVertexPionHistograms_, outCaloInVertexPionHistograms_;
  TrackHistograms allMuonHistograms_, inCaloInVertexMuonHistograms_, inCaloOutVertexMuonHistograms_, outCaloInVertexMuonHistograms_;
  TrackHistograms allElectronHistograms_, inCaloInVertexElectronHistograms_, inCaloOutVertexElectronHistograms_, outCaloInVertexElectronHistograms_;


  ///DQMStore. Used to write out to file
  DQMStore* dqm_;
};

inline void JPTJetAnalyzer::fillHistogram(MonitorElement* histogram, const double value)
{
  if (histogram) histogram->Fill(value);
}

inline void JPTJetAnalyzer::fillHistogram(MonitorElement* histogram, const double valueX, const double valueY)
{
  if (histogram) histogram->Fill(valueX,valueY);
}

#endif
