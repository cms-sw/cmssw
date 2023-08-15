#ifndef HLTriggerOffline_Egamma_EmDQM_H
#define HLTriggerOffline_Egamma_EmDQM_H

// Base Class Headers
#include "CommonTools/Utils/interface/PtComparator.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HepMC/GenParticle.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <boost/regex.hpp>

#include "TDirectory.h"
#include "TFile.h"
#include "TH1F.h"
#include <Math/VectorUtil.h>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class EmDQM;

template <class T>
class HistoFiller {
public:
  HistoFiller(EmDQM *d) : dqm(d){};
  ~HistoFiller(){};

  void fillHistos(edm::Handle<trigger::TriggerEventWithRefs> &,
                  const edm::Event &,
                  unsigned int,
                  unsigned int,
                  std::vector<reco::Particle> &,
                  bool &);
  // std::vector<edm::EDGetTokenT<edm::AssociationMap<edm::OneToValue< T ,
  // float>>>> isoNameTokens_;

private:
  EmDQM *dqm;
};

class EmDQM : public DQMOneEDAnalyzer<> {
public:
  friend class HistoFiller<reco::ElectronCollection>;
  friend class HistoFiller<reco::RecoEcalCandidateCollection>;
  friend class HistoFiller<l1extra::L1EmParticleCollection>;

  /// Constructor
  explicit EmDQM(const edm::ParameterSet &pset);

  /// Destructor
  ~EmDQM() override;

  // Operations

  void analyze(const edm::Event &event, const edm::EventSetup &) override;

  void dqmBeginRun(edm::Run const &, edm::EventSetup const &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void dqmEndRun(edm::Run const &, edm::EventSetup const &) override;

private:
  // interface to DQM framework
  std::string dirname_;

  HistoFiller<reco::ElectronCollection> *histoFillerEle;
  HistoFiller<reco::RecoEcalCandidateCollection> *histoFillerClu;
  HistoFiller<l1extra::L1EmParticleCollection> *histoFillerL1NonIso;
  HistoFiller<reco::RecoEcalCandidateCollection> *histoFillerPho;
  HistoFiller<l1extra::L1EmParticleCollection> *histoFillerL1Iso;

  // run in automatic configuration generation mode
  bool autoConfMode_;
  // global parameters
  edm::InputTag triggerObject_;
  unsigned int verbosity_;
  double genEtaAcc_;
  double genEtAcc_;
  bool isData_;
  double ptMax_;
  double ptMin_;
  double etaMax_;
  double phiMax_;
  unsigned int nbins_;
  double eta2DMax_;
  double phi2DMax_;
  unsigned int nbins2D_;
  unsigned int minEtForEtaEffPlot_;
  bool useHumanReadableHistTitles_;
  bool mcMatchedOnly_;
  bool noPhiPlots_;
  bool noIsolationPlots_;

  /** helper to check whether there were enough generator level
   *  electrons/photons (MC) or enough reco level electrons/photons
   *  to analyze this event.
   *
   *  @return if the event has enough of these candidates.
   */
  bool checkGeneratedParticlesRequirement(const edm::Event &event);

  /** similar to checkGeneratedParticlesRequirement(..) but for reconstructed
   *  particles. For the moment, there are some additional requirements in
   *  the MC version so we can't use the same code for both cases.
   */
  bool checkRecoParticlesRequirement(const edm::Event &event);

  /// The instance of the HLTConfigProvider as a data member
  HLTConfigProvider hltConfig_;

  // routines to build validation configuration from HLTConfiguration
  int countSubstring(const std::string &, const std::string &);
  std::vector<std::vector<std::string>> findEgammaPaths();
  std::vector<std::string> getFilterModules(const std::string &);
  double getPrimaryEtCut(const std::string &);
  edm::ParameterSet makePSetForL1SeedFilter(const std::string &);
  edm::ParameterSet makePSetForL1SeedToSuperClusterMatchFilter(const std::string &);
  edm::ParameterSet makePSetForEtFilter(const std::string &);
  edm::ParameterSet makePSetForOneOEMinusOneOPFilter(const std::string &);
  edm::ParameterSet makePSetForPixelMatchFilter(const std::string &);
  edm::ParameterSet makePSetForEgammaGenericFilter(const std::string &);
  edm::ParameterSet makePSetForEgammaGenericQuadraticFilter(const std::string &);
  edm::ParameterSet makePSetForElectronGenericFilter(const std::string &);
  edm::ParameterSet makePSetForEgammaDoubleEtDeltaPhiFilter(const std::string &);

  // set validation configuration parameters for a trigger path
  void SetVarsFromPSet(std::vector<edm::ParameterSet>::iterator);

  // generated parameter set for trigger path
  std::vector<edm::ParameterSet> paramSets;
  // input from generated parameter set
  unsigned int pathIndex;
  std::vector<edm::InputTag> theHLTCollectionLabels;
  unsigned int numOfHLTCollectionLabels;                // Will be size of above vector
  std::vector<std::string> theHLTCollectionHumanNames;  // Human-readable names for the collections
  edm::InputTag theL1Seed;
  std::vector<int> theHLTOutputTypes;
  std::vector<bool> plotiso;
  std::vector<std::vector<edm::InputTag>> isoNames;  // there has to be a better solution
  std::vector<std::pair<double, double>> plotBounds;
  std::vector<unsigned int> nCandCuts;
  // paramters for generator study
  unsigned int reqNum;
  int pdgGen;
  // plotting parameters
  double plotEtMin;
  double plotPtMin;
  double plotPtMax;

  /** collection which should be used for generator particles (MC)
   *  or reconstructed particles (data).
   *
   *  This collection is used for matching the HLT objects against (e.g. match
   * the HLT object to generated particles or reconstructed electrons/photons).
   */
  edm::InputTag gencutCollection_;

  /** number of generator level particles (electrons/photons) required (for MC)
   */
  unsigned int gencut_;

  /** which hltCollectionLabels were SEEN at least once */
  std::vector<std::set<std::string>> hltCollectionLabelsFoundPerPath;
  std::set<std::string> hltCollectionLabelsFound;

  /** which hltCollectionLabels were MISSED at least once */
  std::vector<std::set<std::string>> hltCollectionLabelsMissedPerPath;
  std::set<std::string> hltCollectionLabelsMissed;

  ////////////////////////////////////////////////////////////
  //          Create Histogram containers
  ////////////////////////////////////////////////////////////
  // Et & eta distributions
  std::vector<std::vector<MonitorElement *>> etahists;
  std::vector<std::vector<MonitorElement *>> phihists;
  std::vector<std::vector<MonitorElement *>> ethists;
  std::vector<std::vector<MonitorElement *>> etahistmatchs;
  std::vector<std::vector<MonitorElement *>> phihistmatchs;
  std::vector<std::vector<MonitorElement *>> ethistmatchs;
  std::vector<std::vector<MonitorElement *>> histEtOfHltObjMatchToGens;
  std::vector<std::vector<MonitorElement *>> histEtaOfHltObjMatchToGens;
  std::vector<std::vector<MonitorElement *>> histPhiOfHltObjMatchToGens;
  std::vector<std::vector<MonitorElement *>> etaphihists;
  std::vector<std::vector<MonitorElement *>> etaphihistmatchs;
  std::vector<std::vector<MonitorElement *>> histEtaPhiOfHltObjMatchToGens;
  // Plots of efficiency per step
  std::vector<MonitorElement *> totals;
  std::vector<MonitorElement *> totalmatchs;
  // generator histograms
  std::vector<MonitorElement *> etgens;
  std::vector<MonitorElement *> etagens;
  std::vector<MonitorElement *> phigens;
  std::vector<MonitorElement *> etaphigens;

  GreaterByPt<reco::Particle> pTComparator_;
  GreaterByPt<reco::GenParticle> pTGenComparator_;

  // tokens for data access
  edm::EDGetTokenT<edm::View<reco::Candidate>> genParticles_token;
  edm::EDGetTokenT<trigger::TriggerEventWithRefs> triggerObject_token;
  edm::EDGetTokenT<edm::TriggerResults> hltResults_token;
  edm::EDGetTokenT<edm::View<reco::Candidate>> gencutColl_fidWenu_token;
  edm::EDGetTokenT<edm::View<reco::Candidate>> gencutColl_fidZee_token;
  edm::EDGetTokenT<edm::View<reco::Candidate>> gencutColl_fidTripleEle_token;
  edm::EDGetTokenT<edm::View<reco::Candidate>> gencutColl_fidGammaJet_token;
  edm::EDGetTokenT<edm::View<reco::Candidate>> gencutColl_fidDiGamma_token;
  edm::EDGetTokenT<edm::View<reco::Candidate>> gencutColl_manualConf_token;

  // static variables
  //
  // trigger types considered
  static const unsigned TYPE_SINGLE_ELE = 0;
  static const unsigned TYPE_DOUBLE_ELE = 1;
  static const unsigned TYPE_SINGLE_PHOTON = 2;
  static const unsigned TYPE_DOUBLE_PHOTON = 3;
  static const unsigned TYPE_TRIPLE_ELE = 4;

  // verbosity levels
  static const unsigned OUTPUT_SILENT = 0;
  static const unsigned OUTPUT_ERRORS = 1;
  static const unsigned OUTPUT_WARNINGS = 2;
  static const unsigned OUTPUT_ALL = 3;
};
#endif
