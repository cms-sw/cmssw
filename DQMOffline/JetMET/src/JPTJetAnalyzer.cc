#include "DQMOffline/JetMET/interface/JPTJetAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "JetMETCorrections/Algorithms/interface/JetPlusTrackCorrector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoJets/JetAssociationAlgorithms/interface/JetTracksAssociationDRCalo.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include <cmath>
#include <string>
#include <memory>
#include <vector>

namespace jptJetAnalysis {
  
  // Helpper class to propagate tracks to the calo surface using the same implementation as the JetTrackAssociator
  class TrackPropagatorToCalo
  {
   public:
    TrackPropagatorToCalo();
    void update(const edm::EventSetup& eventSetup);
    math::XYZPoint impactPoint(const reco::Track& track) const;
   private:
    const MagneticField* magneticField_;
    const Propagator* propagator_;
    uint32_t magneticFieldCacheId_;
    uint32_t propagatorCacheId_;
  };
  
  // Helpper class to calculate strip signal to noise and manage the necessary ES objects
  class StripSignalOverNoiseCalculator
  {
   public:
    StripSignalOverNoiseCalculator(const std::string& theQualityLabel = std::string(""));
    void update(const edm::EventSetup& eventSetup);
    double signalOverNoise(const SiStripCluster& cluster) const;
    double operator () (const SiStripCluster& cluster) const
      { return signalOverNoise(cluster); }
   private:
    const std::string qualityLabel_;
    const SiStripQuality* quality_;
    const SiStripNoises* noise_;
    const SiStripGain* gain_;
    uint32_t qualityCacheId_;
    uint32_t noiseCacheId_;
    uint32_t gainCacheId_;
  };
  
}

const char* JPTJetAnalyzer::messageLoggerCatregory = "JetPlusTrackDQM";

JPTJetAnalyzer::JPTJetAnalyzer(const edm::ParameterSet& config)
  : histogramPath_(config.getParameter<std::string>("HistogramPath")),
    verbose_(config.getUntrackedParameter<bool>("PrintDebugMessages",false)),
    rawJetsSrc_(config.getParameter<edm::InputTag>("RawJetCollection")),
    jptCorrectorName_(config.getParameter<std::string>("JPTCorrectorName")),
    trackPropagator_(new jptJetAnalysis::TrackPropagatorToCalo),
    sOverNCalculator_(new jptJetAnalysis::StripSignalOverNoiseCalculator)
{
  //print config to debug log
  std::ostringstream debugStream;
  if (verbose_) {
    debugStream << "Configuration for JPTJetAnalyzer: " << std::endl
                << "\tHistogramPath: " << histogramPath_ << std::endl
                << "\tPrintDebugMessages? " << (verbose_ ? "yes" : "no") << std::endl;
  }
  
  //don't generate debug mesages if debug is disabled
  std::ostringstream* pDebugStream = (verbose_ ? &debugStream : NULL);
  
  //get histogram configuration
  getConfigForHistogram("nTracksPerJetVsJetEtaPhi",config,pDebugStream);
  getConfigForHistogram("TrackSiStripHitStoN",config,pDebugStream);
  getConfigForHistogram("InCaloTrackDirectionJetDR",config,pDebugStream);
  getConfigForHistogram("OutCaloTrackDirectionJetDR",config,pDebugStream);
  getConfigForHistogram("InVertexPionTrackImpactPointJetDR",config,pDebugStream);
  getConfigForHistogram("OutVertexPionTrackImpactPointJetDR",config,pDebugStream);
  getConfigForHistogram("PtFractionInConeVsJetRawEt",config,pDebugStream);
  getConfigForHistogram("PtFractionInConeVsJetEta",config,pDebugStream);
  getConfigForHistogram("CorrFactorVsJetEt",config,pDebugStream);
  getConfigForHistogram("CorrFactorVsJetEta",config,pDebugStream);
  getConfigForHistogram("CorrFactorVsJetEtaPhi",config,pDebugStream);
  getConfigForHistogram("CorrFactorVsJetEtaEt",config,pDebugStream);
  getConfigForTrackHistograms("AllPions",config,pDebugStream);
  getConfigForTrackHistograms("InCaloInVertexPions",config,pDebugStream);
  getConfigForTrackHistograms("InCaloOutVertexPions",config,pDebugStream);
  getConfigForTrackHistograms("OutCaloInVertexPions",config,pDebugStream);
  getConfigForTrackHistograms("AllMuons",config,pDebugStream);
  getConfigForTrackHistograms("InCaloInVertexMuons",config,pDebugStream);
  getConfigForTrackHistograms("InCaloOutVertexMuons",config,pDebugStream);
  getConfigForTrackHistograms("OutCaloInVertexMuons",config,pDebugStream); 
  getConfigForTrackHistograms("AllElectrons",config,pDebugStream);
  getConfigForTrackHistograms("InCaloInVertexElectrons",config,pDebugStream);
  getConfigForTrackHistograms("InCaloOutVertexElectrons",config,pDebugStream);
  getConfigForTrackHistograms("OutCaloInVertexElectrons",config,pDebugStream);
  
  if (verbose_) LogTrace(messageLoggerCatregory) << debugStream.str();
}

JPTJetAnalyzer::~JPTJetAnalyzer()
{}

void JPTJetAnalyzer::beginJob(const edm::EventSetup& eventSetup, DQMStore* dqmStore)
{
  //get JPT corrector
  const JetCorrector* corrector = JetCorrector::getJetCorrector(jptCorrectorName_,eventSetup);
  if (!corrector) edm::LogError(messageLoggerCatregory) << "Failed to get corrector with name " << jptCorrectorName_ << "from the EventSetup";
  jptCorrector_ = dynamic_cast<const JetPlusTrackCorrector*>(corrector);
  if (!jptCorrector_) edm::LogError(messageLoggerCatregory) << "Corrector with name " << jptCorrectorName_ << " is not a JetPlusTrackCorrector";
  
  //book histograms
  dqmStore->setCurrentFolder(histogramPath_);
  bookHistograms(dqmStore);
}

void JPTJetAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& eventSetup, const reco::CaloJet& jptCorrectedJet)
{
  //update the track propagator and strip noise calculator
  trackPropagator_->update(eventSetup);
  sOverNCalculator_->update(eventSetup);
  
  //find raw jet
  edm::Handle<reco::CaloJetCollection> rawJetsHandle;
  event.getByLabel(rawJetsSrc_,rawJetsHandle);
  const reco::CaloJetCollection& rawJets = *rawJetsHandle;
  const reco::CaloJet* pRawJet = NULL;
  double minDeltaR = 1000;
  for (reco::CaloJetCollection::const_iterator iRawJet = rawJets.begin(); iRawJet != rawJets.end(); ++iRawJet) {
    const double dr = deltaR(*iRawJet,jptCorrectedJet);
    if (dr < minDeltaR) {
      minDeltaR = dr;
      pRawJet = &*iRawJet;
    }
  }
  
  //check jet is correctable by JPT
  if (!jptCorrector_->canCorrect(*pRawJet)) return;
  
  //get consitiuents of jet
  jpt::MatchedTracks pions;
  jpt::MatchedTracks muons;
  jpt::MatchedTracks electrons;
  const bool ok = jptCorrector_->matchTracks(*pRawJet,event,eventSetup,pions,muons,electrons);
  if (!ok) return;
  
  //fill histograms
  const uint16_t totalTracks = pions.inVertexInCalo_.size() + pions.outOfVertexInCalo_.size() + pions.inVertexOutOfCalo_.size() +
                              muons.inVertexInCalo_.size() + muons.outOfVertexInCalo_.size() + muons.inVertexOutOfCalo_.size() +
                              electrons.inVertexInCalo_.size() + electrons.outOfVertexInCalo_.size() + electrons.inVertexOutOfCalo_.size();
  fillHistogram(nTracksPerJetVsJetEtaPhiHisto_,pRawJet->eta(),pRawJet->phi(),totalTracks);
  const double correction = jptCorrectedJet.energy() / pRawJet->energy();
  fillHistogram(CorrFactorVsJetEtHisto_,pRawJet->et(),correction);
  fillHistogram(CorrFactorVsJetEtaHisto_,pRawJet->eta(),correction);
  fillHistogram(CorrFactorVsJetEtaEtHisto_,pRawJet->eta(),pRawJet->et(),correction);
  const double ptFractionInCone = findPtFractionInCone(pions.inVertexInCalo_,pions.inVertexOutOfCalo_);
  fillHistogram(PtFractionInConeVsJetRawEtHisto_,pRawJet->et(),ptFractionInCone);
  fillHistogram(PtFractionInConeVsJetEtaHisto_,pRawJet->eta(),ptFractionInCone);
}

void JPTJetAnalyzer::getConfigForHistogram(const std::string& configName, const edm::ParameterSet& psetContainingConfigPSet,
                                             std::ostringstream* pDebugStream)
{
  const std::string psetName = configName+std::string("HistogramConfig");
  if (!psetContainingConfigPSet.exists(psetName)) {
    edm::LogWarning(messageLoggerCatregory) << "Histogram " << configName << " config not found" << std::endl;
    histogramConfig_[configName] = HistogramConfig();
  } else {
    const edm::ParameterSet& pset = psetContainingConfigPSet.getParameter<edm::ParameterSet>(psetName);
    const bool enabled = (pset.exists("Enabled") ? pset.getParameter<bool>("Enabled") : true);
    if (!enabled) {
      histogramConfig_[configName] = HistogramConfig();
      if (pDebugStream) {
        (*pDebugStream) << "\tHistogram: " << configName << " Disabled" << std::endl;
      }
    } else {
      const unsigned int nBins = (pset.exists("NBins") ? pset.getParameter<unsigned int>("NBins") : 0);
      const double min = (pset.exists("Min") ? pset.getParameter<double>("Min") : 0);
      const double max = (pset.exists("Max") ? pset.getParameter<double>("Max") : 0);
      const unsigned int nBinsY = (pset.exists("NBinsY") ? pset.getParameter<unsigned int>("NBinsY") : 0);
      const double minY = (pset.exists("MinY") ? pset.getParameter<double>("MinY") : 0);
      const double maxY = (pset.exists("MaxY") ? pset.getParameter<double>("MaxY") : 0);
      if (nBins) {
        if (pDebugStream) {
          (*pDebugStream) << "\tHistogram: " << configName << "\tEnabled"
                          << "\tNBins: " << nBins << "\tMin: " << min << "\tMax: " << max;
          if (nBinsY) (*pDebugStream) << "\tNBinsY: " << nBinsY << "\tMinY: " << minY << "\tMaxY: " << maxY;
          (*pDebugStream) << std::endl;
        }
        if (nBinsY) {
          histogramConfig_[configName] = HistogramConfig(nBins,min,max,nBinsY,minY,maxY);
        } else {
          histogramConfig_[configName] = HistogramConfig(nBins,min,max);
        }
      }
    }
  }
}

void JPTJetAnalyzer::getConfigForTrackHistograms(const std::string& tag, const edm::ParameterSet& psetContainingConfigPSet,
                                                  std::ostringstream* pDebugStream)
{
  getConfigForHistogram("n"+tag+"TracksPerJet",psetContainingConfigPSet,pDebugStream);
  getConfigForHistogram(tag+"TrackPt",psetContainingConfigPSet,pDebugStream);
  getConfigForHistogram(tag+"TrackPhi",psetContainingConfigPSet,pDebugStream);
  getConfigForHistogram(tag+"TrackEta",psetContainingConfigPSet,pDebugStream);
  getConfigForHistogram(tag+"TrackNHits",psetContainingConfigPSet,pDebugStream);
  getConfigForHistogram(tag+"TrackPtVsEta",psetContainingConfigPSet,pDebugStream);
}

MonitorElement* JPTJetAnalyzer::bookHistogram(const std::string& name, const std::string& title, const std::string& xAxisTitle, DQMStore* dqm)
{
  std::map<std::string,HistogramConfig>::const_iterator configIterator = histogramConfig_.find(name);
  if (configIterator == histogramConfig_.end()) {
    edm::LogWarning(messageLoggerCatregory) << "Trying to book histogram with name " << name << " when no config was not retrieved from ParameterSet";
    return NULL;
  }
  const HistogramConfig& histoConfig = (*configIterator).second;
  if (histoConfig.enabled) {
    MonitorElement* histo = dqm->book1D(name,title,histoConfig.nBins,histoConfig.min,histoConfig.max);
    histo->setAxisTitle(xAxisTitle,1);
    return histo;
  } else {
    return NULL;
  }
}

MonitorElement* JPTJetAnalyzer::bookProfile(const std::string& name, const std::string& title,
                                            const std::string& xAxisTitle, const std::string& yAxisTitle, DQMStore* dqm)
{
  std::map<std::string,HistogramConfig>::const_iterator configIterator = histogramConfig_.find(name);
  if (configIterator == histogramConfig_.end()) {
    edm::LogWarning(messageLoggerCatregory) << "Trying to book histogram with name " << name << " when no config was not retrieved from ParameterSet";
    return NULL;
  }
  const HistogramConfig& histoConfig = (*configIterator).second;
  if (histoConfig.enabled) {
    TProfile* underlyingRootObject = new TProfile(name.c_str(),title.c_str(),histoConfig.nBins,histoConfig.min,histoConfig.max);
    MonitorElement* histo = dqm->bookProfile(name,underlyingRootObject);
    histo->setAxisTitle(xAxisTitle,1);
    histo->setAxisTitle(yAxisTitle,2);
    return histo;
  } else {
    return NULL;
  }
}

MonitorElement* JPTJetAnalyzer::bookProfile2D(const std::string& name, const std::string& title,
                                              const std::string& xAxisTitle, const std::string& yAxisTitle, const std::string zAxisTitle, DQMStore* dqm)
{
  std::map<std::string,HistogramConfig>::const_iterator configIterator = histogramConfig_.find(name);
  if (configIterator == histogramConfig_.end()) {
    edm::LogWarning(messageLoggerCatregory) << "Trying to book histogram with name " << name << " when no config was not retrieved from ParameterSet";
    return NULL;
  }
  const HistogramConfig& histoConfig = (*configIterator).second;
  if (histoConfig.enabled) {
    TProfile2D* underlyingRootObject = new TProfile2D(name.c_str(),title.c_str(),
                                                      histoConfig.nBins,histoConfig.min,histoConfig.max,
                                                      histoConfig.nBinsY,histoConfig.minY,histoConfig.maxY);
    MonitorElement* histo = dqm->bookProfile2D(name,underlyingRootObject);
    histo->setAxisTitle(xAxisTitle,1);
    histo->setAxisTitle(yAxisTitle,2);
    histo->setAxisTitle(zAxisTitle,3);
    return histo;
  } else {
    return NULL;
  }
}

void JPTJetAnalyzer::bookHistograms(DQMStore* dqm)
{
  TrackSiStripHitStoNHisto_            = bookHistogram("TrackSiStripHitStoN","Signal to noise of track SiStrip hits","S/N",dqm);
  InCaloTrackDirectionJetDRHisto_      = bookHistogram("InCaloTrackDirectionJetDR",
                                                       "#Delta R between track direrction at vertex and jet axis (track in cone at calo)","#Delta R",dqm);
  OutCaloTrackDirectionJetDRHisto_     = bookHistogram("OutCaloTrackDirectionJetDR",
                                                       "#Delta R between track direrction at vertex and jet axis (track out of cone at calo)","#Delta R",dqm);
  InVertexTrackImpactPointJetDRHisto_  = bookHistogram("InVertexTrackImpactPointJetDR",
                                                       "#Delta R between track impact point on calo and jet axis (track in cone at vertex)","#Delta R",dqm);
  OutVertexTrackImpactPointJetDRHisto_ = bookHistogram("OutVertexTrackImpactPointJetDR",
                                                       "#Delta R between track impact point on calo and jet axis (track out of cone at vertex)","#Delta R",dqm);
  
  PtFractionInConeVsJetRawEtHisto_ = bookProfile("PtFractionInConeVsJetRawEt","#frac{p_{T}^{in-cone}}{p_{T}^{in-cone}+p_{T}^{out-of-cone}} vs jet raw E_{T}",
                                                 "Jet raw E_{T} / GeV","#frac{p_{T}^{in-cone}}{p_{T}^{in-cone}+p_{T}^{out-of-cone}}",dqm);
  PtFractionInConeVsJetEtaHisto_   = bookProfile("PtFractionInConeVsJetEta","#frac{p_{T}^{in-cone}}{p_{T}^{in-cone}+p_{T}^{out-of-cone}} vs jet #eta",
                                                 "Jet #eta","#frac{p_{T}^{in-cone}}{p_{T}^{in-cone}+p_{T}^{out-of-cone}}",dqm);
  
  CorrFactorVsJetEtHisto_     = bookProfile("CorrFactorVsJetEt","Correction factor vs jet raw E_{T}","Jet raw E_{T}","#frac{E_{T}^{corr}}{E_{T}^{raw}}",dqm);
  CorrFactorVsJetEtaHisto_    = bookProfile("CorrFactorVsJetEta","Correction factor vs jet #eta","Jet #eta","#frac{E_{T}^{corr}}{E_{T}^{raw}}",dqm);
  CorrFactorVsJetEtaPhiHisto_ = bookProfile2D("CorrFactorVsJetEtaPhi","Correction factor vs jet #eta,#phi",
                                              "Jet #eta","Jet #phi","#frac{E_{T}^{corr}}{E_{T}^{raw}}",dqm);
  CorrFactorVsJetEtaEtHisto_  = bookProfile2D("CorrFactorVsJetEtaEt","Correction factor vs jet #eta, raw E_{T}","Jet #eta","Jet raw E_{T} / GeV",
                                              "#frac{E_{T}^{corr}}{E_{T}^{raw}}",dqm);
  
  bookTrackHistograms(&allPionHistograms_,"AllPions","pion",NULL,NULL,dqm);
  bookTrackHistograms(&inCaloInVertexPionHistograms_,"InCaloInVertexPions","pions in cone at calo and vertex",
                      InCaloTrackDirectionJetDRHisto_,InVertexTrackImpactPointJetDRHisto_,dqm);
  bookTrackHistograms(&inCaloOutVertexPionHistograms_,"InCaloOutVertexPions","pions in cone at calo but out at vertex",
                      InCaloTrackDirectionJetDRHisto_,OutVertexTrackImpactPointJetDRHisto_,dqm);
  bookTrackHistograms(&outCaloInVertexPionHistograms_,"OutCaloInVertexPions","pions out of cone at calo but in at vertex",
                      OutCaloTrackDirectionJetDRHisto_,InVertexTrackImpactPointJetDRHisto_,dqm);
  
  bookTrackHistograms(&allMuonHistograms_,"AllMuons","muon",NULL,NULL,dqm);
  bookTrackHistograms(&inCaloInVertexMuonHistograms_,"InCaloInVertexMuons","muons in cone at calo and vertex",
                      InCaloTrackDirectionJetDRHisto_,InVertexTrackImpactPointJetDRHisto_,dqm);
  bookTrackHistograms(&inCaloOutVertexMuonHistograms_,"InCaloOutVertexMuons","muons in cone at calo but out at vertex",
                      InCaloTrackDirectionJetDRHisto_,OutVertexTrackImpactPointJetDRHisto_,dqm);
  bookTrackHistograms(&outCaloInVertexMuonHistograms_,"OutCaloInVertexMuons","muons out of cone at calo but in at vertex",
                      OutCaloTrackDirectionJetDRHisto_,InVertexTrackImpactPointJetDRHisto_,dqm);
  
  bookTrackHistograms(&allElectronHistograms_,"AllElectrons","electron",NULL,NULL,dqm);
  bookTrackHistograms(&inCaloInVertexElectronHistograms_,"InCaloInVertexElectrons","electrons in cone at calo and vertex",
                      InCaloTrackDirectionJetDRHisto_,InVertexTrackImpactPointJetDRHisto_,dqm);
  bookTrackHistograms(&inCaloOutVertexElectronHistograms_,"InCaloOutVertexElectrons","electrons in cone at calo but out at vertex",
                      InCaloTrackDirectionJetDRHisto_,OutVertexTrackImpactPointJetDRHisto_,dqm);
  bookTrackHistograms(&outCaloInVertexElectronHistograms_,"OutCaloInVertexElectrons","electrons out of cone at calo but in at vertex",
                      OutCaloTrackDirectionJetDRHisto_,InVertexTrackImpactPointJetDRHisto_,dqm);
}

void JPTJetAnalyzer::bookTrackHistograms(TrackHistograms* histos, const std::string& tag, const std::string& titleTag,
                                           MonitorElement* trackDirectionJetDRHisto, MonitorElement* trackImpactPointJetDRHisto, DQMStore* dqm)
{
  histos->nTracksHisto = bookHistogram("n"+tag+"TracksPerJet","Number of "+titleTag+" tracks per jet","n Tracks",dqm);
  histos->ptHisto = bookHistogram(tag+"TrackPt",titleTag+" p_{T}","p_{T} /GeV",dqm);
  histos->phiHisto = bookHistogram(tag+"TrackPhi",titleTag+" track #phi","#phi",dqm);
  histos->etaHisto = bookHistogram(tag+"TrackEta",titleTag+" track #eta","#eta",dqm);
  histos->nHitsHisto = bookHistogram(tag+"TrackNHits",titleTag+" track N hits","N hits",dqm);
  histos->ptVsEtaHisto = bookProfile(tag+"TrackPtVsEta",titleTag+" track p_{T} vs #eta","#eta","p_{T} /GeV",dqm);
  histos->trackDirectionJetDRHisto = trackDirectionJetDRHisto;
  histos->trackImpactPointJetDRHisto = trackImpactPointJetDRHisto;
}

void JPTJetAnalyzer::fillTrackHistograms(TrackHistograms& allTracksHistos, TrackHistograms& inCaloInVertexHistos,
                                         TrackHistograms& inCaloOutVertexHistos, TrackHistograms& outCaloInVertexHistos,
                                         const jpt::MatchedTracks& tracks, const reco::CaloJet& rawJet)
{
  fillTrackHistograms(allTracksHistos,tracks.inVertexInCalo_,rawJet);
  fillTrackHistograms(allTracksHistos,tracks.outOfVertexInCalo_,rawJet);
  fillTrackHistograms(allTracksHistos,tracks.inVertexOutOfCalo_,rawJet);
  fillHistogram(allTracksHistos.nTracksHisto,tracks.inVertexInCalo_.size()+tracks.outOfVertexInCalo_.size()+tracks.inVertexOutOfCalo_.size());
  fillSiStripSoNForTracks(tracks.inVertexInCalo_);
  fillSiStripSoNForTracks(tracks.outOfVertexInCalo_);
  fillSiStripSoNForTracks(tracks.inVertexOutOfCalo_);
  fillTrackHistograms(inCaloInVertexHistos,tracks.inVertexInCalo_,rawJet);
  fillHistogram(inCaloInVertexHistos.nTracksHisto,tracks.inVertexInCalo_.size());
  fillTrackHistograms(inCaloOutVertexHistos,tracks.outOfVertexInCalo_,rawJet);
  fillHistogram(inCaloOutVertexHistos.nTracksHisto,tracks.outOfVertexInCalo_.size());
  fillTrackHistograms(outCaloInVertexHistos,tracks.inVertexOutOfCalo_,rawJet);
  fillHistogram(outCaloInVertexHistos.nTracksHisto,tracks.inVertexOutOfCalo_.size());
}

void JPTJetAnalyzer::fillTrackHistograms(TrackHistograms& histos, const reco::TrackRefVector& tracks, const reco::CaloJet& rawJet)
{
  const reco::TrackRefVector::const_iterator tracksEnd = tracks.end();
  for (reco::TrackRefVector::const_iterator iTrack = tracks.begin(); iTrack != tracksEnd; ++iTrack) {
    const reco::Track& track = **iTrack;
    const double pt = track.pt();
    const double phi = track.phi();
    const double eta = track.eta();
    const unsigned int nHits = track.found();
    fillHistogram(histos.ptHisto,pt);
    fillHistogram(histos.phiHisto,phi);
    fillHistogram(histos.etaHisto,eta);
    fillHistogram(histos.nHitsHisto,nHits);
    fillHistogram(histos.ptVsEtaHisto,eta,pt);
    const double trackDirectionJetDR = deltaR(rawJet,track);
    fillHistogram(histos.trackDirectionJetDRHisto,trackDirectionJetDR);
    const double impactPointJetDR = deltaR(rawJet,trackPropagator_->impactPoint(track));
    fillHistogram(histos.trackImpactPointJetDRHisto,impactPointJetDR);
  }
}

void JPTJetAnalyzer::fillSiStripSoNForTracks(const reco::TrackRefVector& tracks)
{
  const reco::TrackRefVector::const_iterator tracksEnd = tracks.end();
  for (reco::TrackRefVector::const_iterator iTrack = tracks.begin(); iTrack != tracksEnd; ++iTrack) {
    const trackingRecHit_iterator trackRecHitsEnd = (*iTrack)->recHitsEnd();
    for (trackingRecHit_iterator iHit = (*iTrack)->recHitsBegin(); iHit != trackRecHitsEnd; ++iHit) {
      fillSiStripHitSoN(**iHit);
    }
  }
}

void JPTJetAnalyzer::fillSiStripHitSoN(const TrackingRecHit& hit)
{
  //check it is an SiStrip hit
  DetId detId(hit.geographicalId());
  if (!( (detId.det() == DetId::Tracker) &&
        ( (detId.subdetId() == SiStripDetId::TIB) ||
          (detId.subdetId() == SiStripDetId::TID) ||
          (detId.subdetId() == SiStripDetId::TOB) ||
          (detId.subdetId() == SiStripDetId::TEC)
        )
     )) return;
  //try to determine the type of the hit
  const TrackingRecHit* pHit = &hit;
  const SiStripRecHit2D* pRecHit2D = dynamic_cast<const SiStripRecHit2D*>(pHit);
  const SiStripMatchedRecHit2D* pMatchedRecHit2D = dynamic_cast<const SiStripMatchedRecHit2D*>(pHit);
  const ProjectedSiStripRecHit2D* pProjctedRecHit2D = dynamic_cast<const ProjectedSiStripRecHit2D*>(pHit);
  //fill signal to noise for appropriate hit
  if (pMatchedRecHit2D) {
    fillSiStripHitSoNForSingleHit(*pMatchedRecHit2D->monoHit());
    fillSiStripHitSoNForSingleHit(*pMatchedRecHit2D->stereoHit());
  } else if (pProjctedRecHit2D) {
    fillSiStripHitSoNForSingleHit(pProjctedRecHit2D->originalHit());
  } else if (pRecHit2D) {
    fillSiStripHitSoNForSingleHit(*pRecHit2D);
  } else {
    edm::LogError(messageLoggerCatregory) << "Hit on det ID " << hit.geographicalId().rawId() << " cannot be converted to a strip hit";
  }
}

void JPTJetAnalyzer::fillSiStripHitSoNForSingleHit(const SiStripRecHit2D& hit)
{
  //get the cluster
  const SiStripCluster* cluster = NULL;
  const SiStripRecHit2D::ClusterRegionalRef& regionalClusterRef = hit.cluster_regional();
  const SiStripRecHit2D::ClusterRef& normalClusterRef = hit.cluster();
  if (regionalClusterRef.isNonnull()) {
    cluster = &*regionalClusterRef;
  } else if (normalClusterRef.isNonnull()) {
    cluster = &*normalClusterRef;
  } else {
    edm::LogError(messageLoggerCatregory) << "Unable to get cluster from SiStripRecHit2D with det ID " << hit.geographicalId().rawId();
    return;
  }
  //calculate signal to noise for cluster
  const double sOverN = (*sOverNCalculator_)(*cluster);
  //fill histogram
  fillHistogram(TrackSiStripHitStoNHisto_,sOverN);
}

double JPTJetAnalyzer::findPtFractionInCone(const reco::TrackRefVector& inConeTracks, const reco::TrackRefVector& outOfConeTracks)
{
  double totalPt = 0;
  double inConePt = 0;
  const reco::TrackRefVector::const_iterator inConeTracksEnd = inConeTracks.end();
  for (reco::TrackRefVector::const_iterator iInConeTrack = inConeTracks.begin(); iInConeTrack != inConeTracksEnd; ++iInConeTrack) {
    const double pt = (*iInConeTrack)->pt();
    totalPt += pt;
    inConePt += pt;
  }
  const reco::TrackRefVector::const_iterator outOfConeTracksEnd = outOfConeTracks.end();
  for (reco::TrackRefVector::const_iterator iOutOfConeTrack = outOfConeTracks.begin(); iOutOfConeTrack != outOfConeTracksEnd; ++iOutOfConeTrack) {
    const double pt = (*iOutOfConeTrack)->pt();
    totalPt += pt;
  }
  if (totalPt) return inConePt/totalPt;
  //return 0 if there are no tracks at all
  else return 0;
}



JPTJetAnalyzer::HistogramConfig::HistogramConfig()
  : enabled(false),
    nBins(0),
    min(0),
    max(0),
    nBinsY(0),
    minY(0),
    maxY(0)
{}

JPTJetAnalyzer::HistogramConfig::HistogramConfig(const unsigned int theNBins, const double theMin, const double theMax)
  : enabled(true),
    nBins(theNBins),
    min(theMin),
    max(theMax),
    nBinsY(0),
    minY(0),
    maxY(0)
{}

JPTJetAnalyzer::HistogramConfig::HistogramConfig(const unsigned int theNBinsX, const double theMinX, const double theMaxX,
                                                 const unsigned int theNBinsY, const double theMinY, const double theMaxY)
  : enabled(true),
    nBins(theNBinsX),
    min(theMinX),
    max(theMaxX),
    nBinsY(theNBinsY),
    minY(theMinY),
    maxY(theMinX)
{}

JPTJetAnalyzer::TrackHistograms::TrackHistograms()
  : ptHisto(NULL),
    phiHisto(NULL),
    etaHisto(NULL),
    nHitsHisto(NULL),
    ptVsEtaHisto(NULL),
    trackDirectionJetDRHisto(NULL),
    trackImpactPointJetDRHisto(NULL)
{}

JPTJetAnalyzer::TrackHistograms::TrackHistograms(MonitorElement* theNTracksHisto,
                                                 MonitorElement* thePtHisto, MonitorElement* thePhiHisto, MonitorElement* theEtaHisto,
                                                 MonitorElement* theNHitsHisto, MonitorElement* thePtVsEtaHisto,
                                                 MonitorElement* theTrackDirectionJetDRHisto, MonitorElement* theTrackImpactPointJetDRHisto)
  : nTracksHisto(theNTracksHisto),
    ptHisto(thePtHisto),
    phiHisto(thePhiHisto),
    etaHisto(theEtaHisto),
    nHitsHisto(theNHitsHisto),
    ptVsEtaHisto(thePtVsEtaHisto),
    trackDirectionJetDRHisto(theTrackDirectionJetDRHisto),
    trackImpactPointJetDRHisto(theTrackImpactPointJetDRHisto)
{}

namespace jptJetAnalysis {
  
  TrackPropagatorToCalo::TrackPropagatorToCalo()
    : magneticField_(NULL),
      propagator_(NULL),
      magneticFieldCacheId_(0),
      propagatorCacheId_(0)
    {}
  
  void TrackPropagatorToCalo::update(const edm::EventSetup& eventSetup)
  {
    //update magnetic filed if necessary
    const IdealMagneticFieldRecord& magneticFieldRecord = eventSetup.get<IdealMagneticFieldRecord>();
    const uint32_t newMagneticFieldCacheId = magneticFieldRecord.cacheIdentifier();
    if ((newMagneticFieldCacheId != magneticFieldCacheId_) || !magneticField_) {
      edm::ESHandle<MagneticField> magneticFieldHandle;
      magneticFieldRecord.get(magneticFieldHandle);
      magneticField_ = magneticFieldHandle.product();
      magneticFieldCacheId_ = newMagneticFieldCacheId;
    }
    //update propagator if necessary
    const TrackingComponentsRecord& trackingComponentsRecord = eventSetup.get<TrackingComponentsRecord>();
    const uint32_t newPropagatorCacheId = trackingComponentsRecord.cacheIdentifier();
    if ((propagatorCacheId_ != newPropagatorCacheId) || !propagator_) {
      edm::ESHandle<Propagator> propagatorHandle;
      trackingComponentsRecord.get("SteppingHelixPropagatorAlong",propagatorHandle);
      propagator_ = propagatorHandle.product();
      propagatorCacheId_ = newPropagatorCacheId;
    }
  }
  
  inline math::XYZPoint TrackPropagatorToCalo::impactPoint(const reco::Track& track) const
  {
    return JetTracksAssociationDRCalo::propagateTrackToCalorimeter(track,*magneticField_,*propagator_);
  }
  
  StripSignalOverNoiseCalculator::StripSignalOverNoiseCalculator(const std::string& theQualityLabel)
    : qualityLabel_(theQualityLabel),
      quality_(NULL),
      noise_(NULL),
      gain_(NULL),
      qualityCacheId_(0),
      noiseCacheId_(0),
      gainCacheId_(0)
    {}
  
  void StripSignalOverNoiseCalculator::update(const edm::EventSetup& eventSetup)
  {
    //update the quality if necessary
    const SiStripQualityRcd& qualityRecord = eventSetup.get<SiStripQualityRcd>();
    const uint32_t newQualityCacheId = qualityRecord.cacheIdentifier();
    if ((newQualityCacheId != qualityCacheId_) || !quality_) {
      edm::ESHandle<SiStripQuality> qualityHandle;
      qualityRecord.get(qualityLabel_,qualityHandle);
      quality_ = qualityHandle.product();
      qualityCacheId_ = newQualityCacheId;
    }
    //update the noise if necessary
    const SiStripNoisesRcd& noiseRecord = eventSetup.get<SiStripNoisesRcd>();
    const uint32_t newNoiseCacheId = noiseRecord.cacheIdentifier();
    if ((newNoiseCacheId != noiseCacheId_) || !noise_) {
      edm::ESHandle<SiStripNoises> noiseHandle;
      noiseRecord.get(noiseHandle);
      noise_ = noiseHandle.product();
      noiseCacheId_ = newNoiseCacheId;
    }
    //update the gain if necessary
    const SiStripGainRcd& gainRecord = eventSetup.get<SiStripGainRcd>();
    const uint32_t newGainCacheId = gainRecord.cacheIdentifier();
    if ((newGainCacheId != gainCacheId_) || !gain_) {
      edm::ESHandle<SiStripGain> gainHandle;
      gainRecord.get(gainHandle);
      gain_ = gainHandle.product();
      gainCacheId_ = newGainCacheId;
    }
  }
  
  double StripSignalOverNoiseCalculator::signalOverNoise(const SiStripCluster& cluster) const
  {
    const uint32_t detId = cluster.geographicalId();
    const uint16_t firstStrip = cluster.firstStrip();
    const SiStripQuality::Range& qualityRange = quality_->getRange(detId);
    const SiStripNoises::Range& noiseRange = noise_->getRange(detId);
    const SiStripApvGain::Range& gainRange = gain_->getRange(detId);
    double signal = 0;
    double noise2 = 0;
    unsigned int nNonZeroStrips = 0;
    const std::vector<uint8_t>& clusterAmplitudes = cluster.amplitudes();
    const std::vector<uint8_t>::const_iterator clusterAmplitudesEnd = clusterAmplitudes.end();
    const std::vector<uint8_t>::const_iterator clusterAmplitudesBegin = clusterAmplitudes.begin();
    for (std::vector<uint8_t>::const_iterator iAmp = clusterAmplitudesBegin; iAmp != clusterAmplitudesEnd; ++iAmp) {
      const uint8_t adc = *iAmp;
      const uint16_t strip = iAmp-clusterAmplitudesBegin+firstStrip;
      const bool stripBad = quality_->IsStripBad(qualityRange,strip);
      const double noise = noise_->getNoise(strip,noiseRange);
      const double gain = gain_->getStripGain(strip,gainRange);
      signal += adc;
      if (adc) ++nNonZeroStrips;
      const double noiseContrib = (stripBad ? 0 : noise/gain);
      noise2 += noiseContrib*noiseContrib;
    }
    const double noise = sqrt(noise2/nNonZeroStrips);
    if (noise) return signal/noise;
    else return 0;
  }
  
}
