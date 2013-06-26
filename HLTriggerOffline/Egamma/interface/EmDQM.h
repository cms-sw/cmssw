#ifndef HLTriggerOffline_Egamma_EmDQM_H
#define HLTriggerOffline_Egamma_EmDQM_H


// Base Class Headers
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include <vector>
#include "TDirectory.h"
#include "HepMC/GenParticle.h"
#include "CommonTools/Utils/interface/PtComparator.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Common/interface/TriggerResults.h"

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

  /// The instance of the HLTConfigProvider as a data member
  HLTConfigProvider hltConf_;

  // Input from cfg file
  edm::InputTag triggerobjwithrefs;
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
  std::string theHltName;
  unsigned verbosity;
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
  std::set<std::string> hltCollectionLabelsFound;

  /** which hltCollectionLabels were MISSED at least once */
  std::set<std::string> hltCollectionLabelsMissed;


  ////////////////////////////////////////////////////////////
  //          Create Histograms                             //
  ////////////////////////////////////////////////////////////
  // Et & eta distributions
  std::vector<MonitorElement*> etahist;
  std::vector<MonitorElement*> phihist;
  std::vector<MonitorElement*> ethist;
  std::vector<MonitorElement*> etahistmatch;
  std::vector<MonitorElement*> phihistmatch;
  std::vector<MonitorElement*> ethistmatch;
  std::vector<MonitorElement*> histEtOfHltObjMatchToGen;
  std::vector<MonitorElement*> histEtaOfHltObjMatchToGen;
  std::vector<MonitorElement*> histPhiOfHltObjMatchToGen;
  // Isolation distributions
  std::vector<MonitorElement*> etahistiso;
  std::vector<MonitorElement*> phihistiso;
  std::vector<MonitorElement*> ethistiso;
  std::vector<MonitorElement*> etahistisomatch;
  std::vector<MonitorElement*> phihistisomatch;
  std::vector<MonitorElement*> ethistisomatch;
  std::vector<MonitorElement*> histEtIsoOfHltObjMatchToGen; 
  std::vector<MonitorElement*> histEtaIsoOfHltObjMatchToGen;
  std::vector<MonitorElement*> histPhiIsoOfHltObjMatchToGen;
  // Plots of efficiency per step
  MonitorElement* total;
  MonitorElement* totalmatch;
  //generator histograms
  MonitorElement* etgen;
  MonitorElement* etagen;
  MonitorElement* phigen;

  // interface to DQM framework
  DQMStore * dbe;
  std::string dirname_;

  template <class T> void fillHistos(edm::Handle<trigger::TriggerEventWithRefs>& ,const edm::Event& ,unsigned int, std::vector<reco::Particle>&, bool & );
  GreaterByPt<reco::Particle> pTComparator_;
  GreaterByPt<reco::GenParticle> pTGenComparator_;
  
};
#endif
