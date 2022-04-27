#ifndef HLTriggerOffline_Egamma_EmDQMReco_H
#define HLTriggerOffline_Egamma_EmDQMReco_H

// Base Class Headers
#include "CommonTools/Utils/interface/PtComparator.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HepMC/GenParticle.h"
#include "TDirectory.h"

#include <memory>
#include <vector>

class EmDQMReco;

template <class T>
class HistoFillerReco {
public:
  HistoFillerReco(EmDQMReco *d) : dqm(d){};
  ~HistoFillerReco(){};

  void fillHistos(edm::Handle<trigger::TriggerEventWithRefs> &triggerObj,
                  const edm::Event &iEvent,
                  unsigned int n,
                  std::vector<reco::Particle> &sortedReco,
                  bool plotReco,
                  bool plotMonpath);
  std::vector<edm::EDGetTokenT<edm::AssociationMap<edm::OneToValue<T, float>>>> isoNameTokens_;

private:
  EmDQMReco *dqm;
};

class EmDQMReco : public DQMEDAnalyzer {
  //----------------------------------------

  /** a class managing a set of MonitorElements for quantities of a fourvector
   *  we want to histogram.
   */
  class FourVectorMonitorElements {
  public:
    /** @param histogramNameTemplate should be a format string (like used in
     * printf(..) for the histogram NAME where the first %s is replaced with 
     * et,eta or phi.
     *
     *  @param histogramTitleTemplate should be a format string (see
     * histogramNameTemplate) for the histogram TITLE where the first %s is
     * replaced with et,eta or phi.
     */
    FourVectorMonitorElements(EmDQMReco *_parent,
                              DQMStore::IBooker &iBooker,
                              const std::string &histogramNameTemplate,
                              const std::string &histogramTitleTemplate);

    void fill(const math::XYZTLorentzVector &momentum);

  private:
    /** for accessing the histogramming parameters */
    EmDQMReco *parent;

    /** DQM objects (histograms) for the variables to be plotted */
    MonitorElement *etMonitorElement;
    MonitorElement *etaMonitorElement;
    MonitorElement *phiMonitorElement;
  };
  //----------------------------------------

public:
  friend class HistoFillerReco<reco::ElectronCollection>;
  friend class HistoFillerReco<reco::RecoEcalCandidateCollection>;
  friend class HistoFillerReco<l1extra::L1EmParticleCollection>;

  /// Constructor
  explicit EmDQMReco(const edm::ParameterSet &pset);

  /// Destructor
  ~EmDQMReco() override;

  // Operations
  void analyze(const edm::Event &event, const edm::EventSetup &) override;
  void dqmBeginRun(const edm::Run &, const edm::EventSetup &) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  // Input from cfg file

  /** the HLT collections to be looked at */
  std::vector<edm::InputTag> theHLTCollectionLabels;

  unsigned int numOfHLTCollectionLabels;  // Will be size of above vector

  bool useHumanReadableHistTitles;
  std::vector<std::string> theHLTCollectionHumanNames;  // Human-readable names for the collections
  // edm::InputTag theL1Seed;
  std::vector<int> theHLTOutputTypes;
  std::vector<bool> plotiso;
  std::vector<std::vector<edm::InputTag>> isoNames;  // there has to be a better solution
  std::vector<std::pair<double, double>> plotBounds;
  std::string theHltName;
  HLTConfigProvider hltConfig_;
  bool isHltConfigInitialized_;

  ////////////////////////////////////////////////////////////
  //          Read from configuration file                  //
  ////////////////////////////////////////////////////////////
  // parameters for generator study
  unsigned int reqNum;
  int pdgGen;
  double recoEtaAcc;
  double recoEtAcc;
  // plotting paramters
  double plotEtaMax;
  double plotPtMin;
  double plotPtMax;
  double plotPhiMax;

  /** number of bins to use for ALL plots (?) */
  unsigned int plotBins;

  // preselction cuts
  // edm::InputTag recocutCollection_;
  unsigned int recocut_;

  /** events which fire this trigger are filled into {et,eta,phi}recomonpath
   */
  std::string triggerNameRecoMonPath;

  /** process name for the trigger results for events to be filled
   *  into {et,eta,phi}recomonpath
   */
  std::string processNameRecoMonPath;

  /** input tag for the reconstructed electron collection
   *  (with respect to which the HLT efficiencies are calculated ?)
   */
  edm::EDGetTokenT<reco::GsfElectronCollection> recoElectronsInput;
  edm::EDGetTokenT<std::vector<reco::SuperCluster>> recoObjectsEBT;
  edm::EDGetTokenT<std::vector<reco::SuperCluster>> recoObjectsEET;
  edm::EDGetTokenT<edm::TriggerResults> hltResultsT;
  edm::EDGetTokenT<trigger::TriggerEventWithRefs> triggerObjT;
  ////////////////////////////////////////////////////////////
  //          Create Histograms                             //
  ////////////////////////////////////////////////////////////
  /** \label Et, eta and phi distributions (RECO) for the different
   *  HLT modules to be looked at. */
  /** @{ */
  // std::vector<MonitorElement*> etahist;
  //  std::vector<MonitorElement*> ethist;
  //  std::vector<MonitorElement*> phiHist;

  std::vector<std::unique_ptr<FourVectorMonitorElements>> standardHist;

  //  std::vector<MonitorElement*> etahistmatchreco;
  //  std::vector<MonitorElement*> ethistmatchreco;
  //  std::vector<MonitorElement*> phiHistMatchReco;
  std::vector<std::unique_ptr<FourVectorMonitorElements>> histMatchReco;

  //  std::vector<MonitorElement*> etahistmatchrecomonpath;
  //  std::vector<MonitorElement*> ethistmatchrecomonpath;
  //  std::vector<MonitorElement*> phiHistMatchRecoMonPath;
  std::vector<std::unique_ptr<FourVectorMonitorElements>> histMatchRecoMonPath;

  //  std::vector<MonitorElement*> histEtOfHltObjMatchToReco;
  //  std::vector<MonitorElement*> histEtaOfHltObjMatchToReco;
  //  std::vector<MonitorElement*> histPhiOfHltObjMatchToReco;
  std::vector<std::unique_ptr<FourVectorMonitorElements>> histHltObjMatchToReco;

  /** @} */

  /** \label Isolation distributions */
  /** @{ */
  std::vector<MonitorElement *> etahistiso;
  std::vector<MonitorElement *> ethistiso;
  std::vector<MonitorElement *> phiHistIso;

  std::vector<MonitorElement *> etahistisomatchreco;
  std::vector<MonitorElement *> ethistisomatchreco;
  std::vector<MonitorElement *> phiHistIsoMatchReco;

  std::vector<MonitorElement *> histEtIsoOfHltObjMatchToReco;
  std::vector<MonitorElement *> histEtaIsoOfHltObjMatchToReco;
  std::vector<MonitorElement *> histPhiIsoOfHltObjMatchToReco;
  /** @} */

  /** Plots of efficiency per step (note that these are NOT
   *  filled with four vector quantities but rather event counts) */
  MonitorElement *totalreco;
  MonitorElement *totalmatchreco;

  /** \name reco histograms */
  /** @{ */
  //  MonitorElement* etreco;
  //  MonitorElement* etareco;
  //  MonitorElement* phiReco;
  std::unique_ptr<FourVectorMonitorElements> histReco;

  //  MonitorElement* etrecomonpath;
  //  MonitorElement* etarecomonpath;
  //  MonitorElement* phiRecoMonPath;
  std::unique_ptr<FourVectorMonitorElements> histRecoMonpath;

  //  MonitorElement* etahistmonpath;
  //  MonitorElement* ethistmonpath;
  //  MonitorElement* phiHistMonPath;
  std::unique_ptr<FourVectorMonitorElements> histMonpath;
  /** @} */

  int eventnum;
  // int prescale;

  // interface to DQM framework
  std::string dirname_;

  HistoFillerReco<reco::ElectronCollection> *histoFillerEle;
  HistoFillerReco<reco::RecoEcalCandidateCollection> *histoFillerClu;
  HistoFillerReco<l1extra::L1EmParticleCollection> *histoFillerL1NonIso;
  HistoFillerReco<reco::RecoEcalCandidateCollection> *histoFillerPho;
  HistoFillerReco<l1extra::L1EmParticleCollection> *histoFillerL1Iso;

  // template <class T> void
  // fillHistos(edm::Handle<trigger::TriggerEventWithRefs>&,const edm::Event&
  // ,unsigned int, std::vector<reco::Particle>&, bool, bool);
  GreaterByPt<reco::Particle> pTComparator_;
  GreaterByPt<reco::GsfElectron> pTRecoComparator_;

  //----------------------------------------
};
#endif
