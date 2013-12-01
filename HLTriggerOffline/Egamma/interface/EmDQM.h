#ifndef HLTriggerOffline_Egamma_EmDQM_H
#define HLTriggerOffline_Egamma_EmDQM_H


// Base Class Headers
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>

#include "TDirectory.h"
#include "TFile.h"
#include "TH1F.h"
#include <memory>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <Math/VectorUtil.h>

class EmDQM : public edm::EDAnalyzer{
public:
  /// Constructor
  explicit EmDQM(const edm::ParameterSet& pset);

  /// Destructor
  ~EmDQM();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup&);
  void beginJob();
  void endJob();

  void beginRun(edm::Run const&, edm::EventSetup const&);
  void endRun(edm::Run const&, edm::EventSetup const&);

private:
  const edm::ParameterSet& pset;
  std::vector<edm::ParameterSet> paramSets;

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
  bool checkRecoParticlesRequirement(const edm::Event & event);

  edm::InputTag triggerObject_;
  /// The instance of the HLTConfigProvider as a data member
  HLTConfigProvider hltConfig_;
  HLTConfigProvider hltConf_;

      std::vector<std::vector<std::string> > findEgammaPaths();
      std::vector<std::string> getFilterModules(const std::string&);
      double getPrimaryEtCut(const std::string&);

      edm::ParameterSet makePSetForL1SeedFilter(const std::string&);
      edm::ParameterSet makePSetForL1SeedToSuperClusterMatchFilter(const std::string&);
      edm::ParameterSet makePSetForEtFilter(const std::string&);
      edm::ParameterSet makePSetForOneOEMinusOneOPFilter(const std::string&);
      edm::ParameterSet makePSetForPixelMatchFilter(const std::string&);
      edm::ParameterSet makePSetForEgammaGenericFilter(const std::string&);
      edm::ParameterSet makePSetForEgammaGenericQuadraticFilter(const std::string&);
      edm::ParameterSet makePSetForElectronGenericFilter(const std::string&);
      edm::ParameterSet makePSetForEgammaDoubleEtDeltaPhiFilter(const std::string&);

      void SetVarsFromPSet(std::vector<edm::ParameterSet>::iterator);

  // Input from cfg file
  unsigned int pathIndex;
  std::vector<edm::InputTag> theHLTCollectionLabels;  
  unsigned int numOfHLTCollectionLabels;  // Will be size of above vector
  bool useHumanReadableHistTitles;
  bool mcMatchedOnly;
  bool noPhiPlots;
  bool noIsolationPlots;
  std::vector<std::string> theHLTCollectionHumanNames; // Human-readable names for the collections
  edm::InputTag theL1Seed;
  std::vector<int> theHLTOutputTypes;
  std::vector<bool> plotiso;
  std::vector<std::vector<edm::InputTag> > isoNames; // there has to be a better solution
  std::vector<std::pair<double,double> > plotBounds; 
  std::vector<unsigned int> nCandCuts;

  static const unsigned TYPE_SINGLE_ELE = 0;
  static const unsigned TYPE_DOUBLE_ELE = 1;
  static const unsigned TYPE_SINGLE_PHOTON = 2;
  static const unsigned TYPE_DOUBLE_PHOTON = 3;
  static const unsigned TYPE_TRIPLE_ELE = 4;

  unsigned verbosity_;
  // verbosity levels
  static const unsigned OUTPUT_SILENT = 0;
  static const unsigned OUTPUT_ERRORS = 1;
  static const unsigned OUTPUT_WARNINGS = 2;
  static const unsigned OUTPUT_ALL = 3;

  ////////////////////////////////////////////////////////////
  //          Read from configuration file                  //
  ////////////////////////////////////////////////////////////
  // paramters for generator study
  unsigned int reqNum;
  int   pdgGen;
  double genEtaAcc;
  double genEtAcc;
  // plotting paramters
  double plotEtMin;
  double plotEtaMax;
  double plotPhiMax;
  double plotPtMin ;
  double plotPtMax ;
  unsigned int plotBins ;
  unsigned int plotMinEtForEtaEffPlot;
  // preselction cuts

  /** collection which should be used for generator particles (MC)
   *  or reconstructed particles (data).
   *
   *  This collection is used for matching the HLT objects against (e.g. match the HLT
   *  object to generated particles or reconstructed electrons/photons).
   */
  edm::InputTag gencutCollection_;

  /** number of generator level particles (electrons/photons) required (for MC) */
  unsigned int gencut_;

  /** which hltCollectionLabels were SEEN at least once */
  std::vector<std::set<std::string> > hltCollectionLabelsFoundPerPath;
  std::set<std::string> hltCollectionLabelsFound;

  /** which hltCollectionLabels were MISSED at least once */
  std::vector<std::set<std::string> > hltCollectionLabelsMissedPerPath;
  std::set<std::string> hltCollectionLabelsMissed;


  ////////////////////////////////////////////////////////////
  //          Create Histograms                             //
  ////////////////////////////////////////////////////////////
  // Et & eta distributions
  std::vector<std::vector<MonitorElement*> > etahists;
  std::vector<std::vector<MonitorElement*> > phihists;
  std::vector<std::vector<MonitorElement*> > ethists;
  std::vector<std::vector<MonitorElement*> > etahistmatchs;
  std::vector<std::vector<MonitorElement*> > phihistmatchs;
  std::vector<std::vector<MonitorElement*> > ethistmatchs;
  std::vector<std::vector<MonitorElement*> > histEtOfHltObjMatchToGens;
  std::vector<std::vector<MonitorElement*> > histEtaOfHltObjMatchToGens;
  std::vector<std::vector<MonitorElement*> > histPhiOfHltObjMatchToGens;
  // Isolation distributions
  std::vector<std::vector<MonitorElement*> > etahistisos;
  std::vector<std::vector<MonitorElement*> > phihistisos;
  std::vector<std::vector<MonitorElement*> > ethistisos;
  std::vector<std::vector<MonitorElement*> > etahistisomatchs;
  std::vector<std::vector<MonitorElement*> > phihistisomatchs;
  std::vector<std::vector<MonitorElement*> > ethistisomatchs;
  std::vector<std::vector<MonitorElement*> > histEtIsoOfHltObjMatchToGens; 
  std::vector<std::vector<MonitorElement*> > histEtaIsoOfHltObjMatchToGens;
  std::vector<std::vector<MonitorElement*> > histPhiIsoOfHltObjMatchToGens;
  // Plots of efficiency per step
  std::vector<MonitorElement*> totals;
  std::vector<MonitorElement*> totalmatchs;
  //generator histograms
  std::vector<MonitorElement*> etgens;
  std::vector<MonitorElement*> etagens;
  std::vector<MonitorElement*> phigens;

  // interface to DQM framework
  DQMStore * dbe;
  std::string dirname_;

  template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& ,const edm::Event& ,unsigned int, unsigned int, std::vector<reco::Particle>&, bool & );
  GreaterByPt<reco::Particle> pTComparator_;
  GreaterByPt<reco::GenParticle> pTGenComparator_;

  // tokens for data access
  edm::EDGetTokenT<edm::View<reco::Candidate> > genParticles_token;
  edm::EDGetTokenT<trigger::TriggerEventWithRefs> triggerObject_token;
  edm::EDGetTokenT<edm::View<reco::Candidate> > gencutCollection_token;
};
#endif
