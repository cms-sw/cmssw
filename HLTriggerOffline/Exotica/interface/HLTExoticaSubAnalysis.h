#ifndef HLTriggerOffline_Exotica_HLTExoticaSubAnalysis_H
#define HLTriggerOffline_Exotica_HLTExoticaSubAnalysis_H

/** \class HLTExoticaSubAnalysis
 *  Generate histograms for trigger efficiencies Exotica related
 *  Documentation available on the CMS TWiki:
 *  https://twiki.cern.ch/twiki/bin/view/CMS/EXOTICATriggerValidation
 *
 *  \author  Thiago R. Fernandez Perez Tomei
 *           Based and adapted from:
 *           J. Duarte Campderros code from HLTriggerOffline/Higgs
 *           J. Klukas, M. Vander Donckt and J. Alcaraz code
 *           from the HLTriggerOffline/Muon package.
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "HLTriggerOffline/Exotica/interface/HLTExoticaPlotter.h"

#include <cstring>
#include <map>
#include <set>
#include <vector>

/// Class to manage all object collections from a centralized place.
struct EVTColContainer;

/// This class is the main workhorse of the package.
/// It makes the histograms for one given analysis, taking care
/// of all HLT paths related to that analysis.
class HLTExoticaSubAnalysis {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  enum class Level { GEN = 98, RECO = 99 };

  HLTExoticaSubAnalysis(const edm::ParameterSet &pset,
                        const std::string &analysisname,
                        edm::ConsumesCollector &&consCollector);
  ~HLTExoticaSubAnalysis();
  void beginJob();
  void beginRun(const edm::Run &iRun, const edm::EventSetup &iEventSetup);
  void endRun();

  /// Method to book all relevant histograms in the DQMStore.
  /// Uses the IBooker interface for thread safety.
  /// Intended to be called from master object.
  void subAnalysisBookHistos(DQMStore::IBooker &iBooker, const edm::Run &iRun, const edm::EventSetup &iSetup);

  /// Method to fill all relevant histograms.
  /// Notice that we update the EVTColContaner to point to the collections we
  /// want.
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iEventSetup, EVTColContainer *cols);

private:
  /// Return the objects (muons,electrons,photons,...) needed by a HLT path.
  /// Will in general return: 0 for muon, 1 for electron, 2 for photon,
  /// 3 for PFMET, 4 for PFTau, 5 for Jet.
  /// Notice that this function is really based on a parsing of the name of
  /// the path; any incongruences there may lead to problems.
  const std::vector<unsigned int> getObjectsType(const std::string &hltpath) const;

  /// Creates the maps that map  which collection should come from which label
  void getNamesOfObjects(const edm::ParameterSet &anpset);
  /// Registers consumption of objects
  void registerConsumes(edm::ConsumesCollector &consCollector);
  /// Gets the collections themselves
  void getHandlesToObjects(const edm::Event &iEvent, EVTColContainer *col);
  /// Initializes the selectors of the objects based on which object it is
  void initSelector(const unsigned int &objtype);
  /// This function applies the selectors initialized previously to the objects,
  /// and matches the passing objects to HLT objects.
  void insertCandidates(const unsigned int &objtype,
                        const EVTColContainer *col,
                        std::vector<reco::LeafCandidate> *matches,
                        std::map<int, double> &theSumEt,
                        std::map<int, std::vector<const reco::Track *>> &trkObjs);

  /// The internal functions to book and fill histograms
  void bookHist(DQMStore::IBooker &iBooker,
                const std::string &source,
                const std::string &objType,
                const std::string &variable);
  void fillHist(const std::string &source, const std::string &objType, const std::string &variable, const float &value);

  /// Internal, working copy of the PSet passed from above.
  edm::ParameterSet _pset;

  /// The name of this sub-analysis
  std::string _analysisname;

  /// The minimum number of reco/gen candidates needed by the analysis
  unsigned int _minCandidates;

  /// The hlt paths to check for.
  std::vector<std::string> _hltPathsToCheck;
  /// The hlt paths found in the hltConfig
  std::set<std::string> _hltPaths;
  /// Relation between the short and long versions of the path
  std::map<std::string, std::string> _shortpath2long;

  /// The labels of the object collections to be used in this analysis.
  std::string _hltProcessName;
  edm::InputTag _genParticleLabel;
  edm::InputTag _trigResultsLabel;
  edm::InputTag _beamSpotLabel;
  std::map<unsigned int, edm::InputTag> _recLabels;
  /// And also the tokens to get the object collections
  edm::EDGetTokenT<reco::GenParticleCollection> _genParticleToken;
  edm::EDGetTokenT<edm::TriggerResults> _trigResultsToken;
  edm::EDGetTokenT<reco::BeamSpot> _bsToken;
  std::map<unsigned int, edm::EDGetToken> _tokens;

  /// Some kinematical parameters
  std::vector<double> _parametersEta;
  std::vector<double> _parametersPhi;
  std::vector<double> _parametersTurnOn;
  std::vector<double> _parametersTurnOnSumEt;
  std::vector<double> _parametersDxy;

  // flag to switch off
  bool _drop_pt2;
  bool _drop_pt3;

  /// gen/rec objects cuts
  std::map<unsigned int, std::string> _genCut;
  std::map<unsigned int, std::string> _recCut;
  /// gen/rec pt-leading objects cuts
  std::map<unsigned int, std::string> _genCut_leading;
  std::map<unsigned int, std::string> _recCut_leading;

  /// The concrete String selectors (use the string cuts introduced
  /// via the config python)
  std::map<unsigned int, StringCutObjectSelector<reco::GenParticle> *> _genSelectorMap;
  StringCutObjectSelector<reco::Muon> *_recMuonSelector;
  StringCutObjectSelector<reco::Track> *_recMuonTrkSelector;
  StringCutObjectSelector<reco::Track> *_recTrackSelector;
  StringCutObjectSelector<reco::GsfElectron> *_recElecSelector;
  StringCutObjectSelector<reco::MET> *_recMETSelector;
  StringCutObjectSelector<reco::PFMET> *_recPFMETSelector;
  StringCutObjectSelector<reco::PFMET> *_recPFMHTSelector;
  StringCutObjectSelector<reco::GenMET> *_genMETSelector;
  StringCutObjectSelector<reco::CaloMET> *_recCaloMETSelector;
  StringCutObjectSelector<reco::CaloMET> *_recCaloMHTSelector;
  StringCutObjectSelector<reco::PFTau> *_recPFTauSelector;
  StringCutObjectSelector<reco::Photon> *_recPhotonSelector;
  StringCutObjectSelector<reco::PFJet> *_recPFJetSelector;
  StringCutObjectSelector<reco::CaloJet> *_recCaloJetSelector;

  /// The plotters: managers of each hlt path where the plots are done
  std::vector<HLTExoticaPlotter> _plotters;

  /// counting HLT passed events
  std::map<std::string, int> _triggerCounter;

  /// Interface to the HLT information
  HLTConfigProvider _hltConfig;

  /// Structure of the MonitorElements
  std::map<std::string, MonitorElement *> _elements;
};

#endif
