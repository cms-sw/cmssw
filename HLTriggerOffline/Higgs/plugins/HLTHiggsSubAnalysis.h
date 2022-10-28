#ifndef HLTriggerOffline_Higgs_HLTHiggsSubAnalysis_H
#define HLTriggerOffline_Higgs_HLTHiggsSubAnalysis_H

/** \class HLTHiggsSubAnalysis
*  Generate histograms for trigger efficiencies Higgs related
*  Documentation available on the CMS TWiki:
*  https://twiki.cern.ch/twiki/bin/view/CMS/HiggsWGHLTValidate
*
*  \author  J. Duarte Campderros (based and adapted on J. Klukas,
*           M. Vander Donckt and J. Alcaraz code from the 
*           HLTriggerOffline/Muon package)
*
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "HLTHiggsPlotter.h"

#include <vector>
#include <set>
#include <map>
#include <cstring>

struct EVTColContainer;

class HLTHiggsSubAnalysis {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  enum { GEN, RECO };

  HLTHiggsSubAnalysis(const edm::ParameterSet &pset, const std::string &analysisname, edm::ConsumesCollector &&iC);
  ~HLTHiggsSubAnalysis();
  void beginJob();
  void beginRun(const edm::Run &iRun, const edm::EventSetup &iEventSetup);
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iEventSetup, EVTColContainer *cols);
  void bookHistograms(DQMStore::IBooker &);

  //! Extract what objects need this analysis
  const std::vector<unsigned int> getObjectsType(const std::string &hltpath) const;

private:
  void bookobjects(const edm::ParameterSet &anpset, edm::ConsumesCollector &iC);
  void initobjects(const edm::Event &iEvent, EVTColContainer *col);
  void InitSelector(const unsigned int &objtype);
  void initAndInsertJets(const edm::Event &iEvent, EVTColContainer *cols, std::vector<MatchStruct> *matches);
  void passJetCuts(std::vector<MatchStruct> *matches,
                   std::map<std::string, bool> &jetCutResult,
                   float &dEtaqq,
                   float &mqq,
                   float &dPhibb,
                   float &CSV1,
                   float &CSV2,
                   float &CSV3);
  void passOtherCuts(const std::vector<MatchStruct> &matches, std::map<std::string, bool> &jetCutResult);
  void insertcandidates(const unsigned int &objtype, const EVTColContainer *col, std::vector<MatchStruct> *matches);

  void bookHist(const std::string &source, const std::string &objType, const std::string &variable, DQMStore::IBooker &);
  void fillHist(const std::string &source, const std::string &objType, const std::string &variable, const float &value);

  edm::ParameterSet _pset;

  std::string _analysisname;

  //! The minimum number of reco/gen candidates needed by the analysis
  unsigned int _minCandidates;

  double _HtJetPtMin;
  double _HtJetEtaMax;

  bool _bookHtPlots;

  std::string _hltProcessName;
  std::string _histDirectory;

  //! the hlt paths with regular expressions
  std::vector<std::string> _hltPathsToCheck;
  //! the hlt paths found in the hltConfig
  std::set<std::string> _hltPaths;

  //! Relation between the short version of a path
  std::map<std::string, std::string> _shortpath2long;

  // The name of the object collections to be used in this
  // analysis.
  edm::EDGetTokenT<reco::GenParticleCollection> _genParticleLabel;
  edm::EDGetTokenT<reco::GenJetCollection> _genJetLabel;
  edm::EDGetTokenT<reco::PFJetCollection> _recoHtJetLabel;

  std::map<unsigned int, std::string> _recLabels;
  edm::EDGetTokenT<reco::MuonCollection> _recLabelsMuon;
  edm::EDGetTokenT<reco::GsfElectronCollection> _recLabelsElec;
  edm::EDGetTokenT<reco::PhotonCollection> _recLabelsPhoton;
  edm::EDGetTokenT<reco::CaloMETCollection> _recLabelsCaloMET;
  edm::EDGetTokenT<reco::PFMETCollection> _recLabelsPFMET;
  edm::EDGetTokenT<reco::PFTauCollection> _recLabelsPFTau;
  edm::EDGetTokenT<reco::PFJetCollection> _recLabelsPFJet;
  edm::EDGetTokenT<reco::JetTagCollection> _recTagPFJet;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo> > _puSummaryInfo;

  //! Some kinematical parameters
  std::vector<double> _parametersEta;
  std::vector<double> _parametersPhi;
  std::vector<double> _parametersPu;
  std::vector<double> _parametersHt;
  std::vector<double> _parametersTurnOn;
  edm::EDGetTokenT<edm::TriggerResults> _trigResultsTag;

  std::map<unsigned int, double> _cutMinPt;
  std::map<unsigned int, double> _cutMaxEta;
  std::map<unsigned int, unsigned int> _cutMotherId;     //TO BE DEPRECATED (HLTMATCH)
  std::map<unsigned int, std::vector<double> > _cutsDr;  // TO BE DEPRECATED (HLTMATCH)
  //! gen/rec objects cuts
  std::map<unsigned int, std::string> _genCut;
  std::map<unsigned int, std::string> _recCut;

  //! The concrete String selectors (use the string cuts introduced
  //! via the config python)
  std::map<unsigned int, StringCutObjectSelector<reco::GenParticle> *> _genSelectorMap;
  StringCutObjectSelector<reco::GenJet> *_genJetSelector;
  StringCutObjectSelector<reco::Muon> *_recMuonSelector;
  StringCutObjectSelector<reco::GsfElectron> *_recElecSelector;
  StringCutObjectSelector<reco::CaloMET> *_recCaloMETSelector;
  StringCutObjectSelector<reco::PFMET> *_recPFMETSelector;
  StringCutObjectSelector<reco::PFTau> *_recPFTauSelector;
  StringCutObjectSelector<reco::Photon> *_recPhotonSelector;
  StringCutObjectSelector<reco::PFJet> *_recPFJetSelector;
  StringCutObjectSelector<reco::Track> *_recTrackSelector;

  //N-1 cut values
  std::vector<double> _NminOneCuts;
  bool _useNminOneCuts;
  unsigned int NptPlots;

  // The plotters: managers of each hlt path where the plots are done
  std::vector<HLTHiggsPlotter> _analyzers;

  HLTConfigProvider _hltConfig;
  std::map<std::string, MonitorElement *> _elements;
};
#endif
