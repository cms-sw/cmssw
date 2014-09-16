#ifndef TOPDQMHELPERS
#define TOPDQMHELPERS

#include <string>
#include <vector>
#include <iostream>
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include <DataFormats/METReco/interface/PFMET.h>
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
using namespace std;
/**
   \fn      accept TopDQMHelpers.h "DQM/Physics/interface/TopDQMHelpers.h"

   \brief   Helper function to determine trigger accepts.

   Helper function to determine trigger accept for given TriggerResults and
   a given TriggerPath(s).
*/

inline bool accept(const edm::Event& event,
                   const edm::TriggerResults& triggerTable,
                   const std::string& triggerPath) {
  bool passed = false;
  const edm::TriggerNames& triggerNames = event.triggerNames(triggerTable);
  for (unsigned int i = 0; i < triggerNames.triggerNames().size(); ++i) {
    if (triggerNames.triggerNames()[i] == triggerPath) {
      if (triggerTable.accept(i)) {
        passed = true;
        break;
      }
    }
  }
  return passed;
}

inline bool accept(const edm::Event& event,
                   const edm::TriggerResults& triggerTable,
                   const std::vector<std::string>& triggerPaths) {
  bool passed = false;
  for (unsigned int j = 0; j < triggerPaths.size(); ++j) {
    if (accept(event, triggerTable, triggerPaths[j])) {
      passed = true;
      break;
    }
  }
  return passed;
}

#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/Framework/interface/EventSetup.h"

/**
   \class   Calculate TopDQMHelpers.h "DQM/Physics/interface/TopDQMHelpers.h"

   \brief   Helper class for the calculation of a top and a W boson mass estime.

   Helper class for the calculation of a top and a W boson mass estimate. The
   core implementation originates from the plugin TtSemiLepHypMaxSumPtWMass
   in TopQuarkAnalysis/TopJetCombination package. It may be extended to include
   b tag information.
*/

class Calculate {
 public:
  /// default constructor
  Calculate(int maxNJets, double wMass);
  /// default destructor
  ~Calculate() {};

  /// calculate W boson mass estimate
  double massWBoson(const std::vector<reco::Jet>& jets);
  /// calculate t-quark mass estimate
  double massTopQuark(const std::vector<reco::Jet>& jets);
  /// calculate b-tagged t-quark mass estimate
  // double massBTopQuark(const std::vector<reco::Jet>& jets, std::vector<bool>
  // bjet);
  double massBTopQuark(const std::vector<reco::Jet>& jets,
                       std::vector<double> VbtagWP, double btagWP_);

  /// calculate W boson transverse mass estimate
  double tmassWBoson(reco::RecoCandidate* lep, const reco::MET& met,
                     const reco::Jet& b);
  /// calculate top quark transverse mass estimate
  double tmassTopQuark(reco::RecoCandidate* lep, const reco::MET& met,
                       const reco::Jet& b);

 private:
  /// do the calculation; this is called only once per event by the first
  /// function call to return a mass estimate. The once calculated values
  /// are cached afterwards
  void operator()(const std::vector<reco::Jet>& jets);
  /// do the calculation of the t-quark mass with one b-jet
  void operator2(const std::vector<reco::Jet>&, std::vector<double>, double);
  /// do the calculation of the transverse top and W masses
  void operator()(const reco::Jet& bJet, reco::RecoCandidate* lepton,
                  const reco::MET& met);

 private:
  /// indicate failed associations
  bool failed_;
  /// max. number of jets to be considered
  int maxNJets_;
  /// paramater of the w boson mass
  double wMass_;
  /// cache of w boson mass estimate
  double massWBoson_;
  /// cache of top quark mass estimate
  double massTopQuark_;
  /// cache of b-tagged top quark mass estimate
  double massBTopQuark_;
  /// cache of W boson transverse mass estimate
  double tmassWBoson_;
  /// cache of top quark transverse mass estimate
  double tmassTopQuark_;
};

#include "DataFormats/JetReco/interface/JetID.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

/**
   \class   SelectionStep TopDQMHelpers.h
   "DQM/Physics/interface/TopDQMHelpers.h"

   \brief   Templated helper class to allow a selection on a certain object
   collection.

   Templated helper class to allow a selection on a certain object collection,
   which may be monitored by a separate class afterwards. The class wraps and
   slightly extends the features of the StringCutParser to allow also to apply
   event based selections, according to a minimal or maximal number of elements
   in the collection after the object selection has been applied. It takes an
   edm::ParameterSet in the constructor, which should contain the following
   elements:
    - src          : the input collection (mandatory).
    - select       : the selection string (mandatory).
    - min          : whether there is a min value on which to reject the whole
   event after
                     the selection (optional).
    - max          : whether there is a max value on which to reject the whole
   event after
                     the selection (optional).
    - electronId   : input tag of an electronId association map and cut value
                     (optional).
    - jetCorrector : label of jet corrector (optional).
    - jetBTagger   : parameters defining the btag algorithm and working point of
   choice
                     (optional).
    - jetID        : parameters defining the jetID value map and selection
   (optional).


   The parameters _src_ and _select_ are mandatory. The parameters _min_ and
   _max_ are optional. The parameters _electronId_ and _jetCorrector_ are
   optional. They are added to keep the possibility to apply selections on id'ed
   electrons or on corrected jets.  They may be omitted in the PSet for
   simplification reasons if not needed at any time.  They are not effiective
   for other object collections but electrons or jets.  If none of the two
   parameters _min_ or _max_ is found in the event the select function returns
   true if at least one object fullfilled the requirements.

   The class has one template value, which is the object collection to apply the
   selection on. This has to be parsed to the StringCutParser class. The
   function select is overrided for jets to circumvent problems with the
   template specialisation. Note that for MET not type1 or muon corrections are
   supported on reco candidates.
*/

template <typename Object>
class SelectionStep {
 public:
  /// default constructor
  SelectionStep(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC);
  /// default destructor
  ~SelectionStep() {};

  /// apply selection
  bool select(const edm::Event& event);
  bool select(const edm::Event& event, const std::string& type);
  /// apply selection override for jets
  bool select(const edm::Event& event, const edm::EventSetup& setup);
  bool selectVertex(const edm::Event& event);

 private:
  /// input collection
  edm::EDGetTokenT<edm::View<Object> > src_;
  /// min/max for object multiplicity
  int min_, max_;
  /// electronId label as extra selection type
  edm::EDGetTokenT<edm::ValueMap<float> > electronId_;
  /// electronId pattern we expect the following pattern:
  ///  0: fails
  ///  1: passes electron ID only
  ///  2: passes electron Isolation only
  ///  3: passes electron ID and Isolation only
  ///  4: passes conversion rejection
  ///  5: passes conversion rejection and ID
  ///  6: passes conversion rejection and Isolation
  ///  7: passes the whole selection
  /// As described on
  /// https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
  double eidCutValue_;
  /// jet corrector as extra selection type
  std::string jetCorrector_;
  /// choice for b-tag as extra selection type
  edm::EDGetTokenT<reco::JetTagCollection> btagLabel_;
  /// choice of b-tag working point as extra selection type
  double btagWorkingPoint_;
  /// jetID as an extra selection type
  edm::EDGetTokenT<reco::JetIDValueMap> jetIDLabel_;

  edm::EDGetTokenT<edm::View<reco::Vertex> > pvs_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > gsfEs_;
  /// string cut selector
  StringCutObjectSelector<Object> select_;
  /// selection string on the jetID
  StringCutObjectSelector<reco::JetID>* jetIDSelect_;
};

/// default constructor
template <typename Object>
SelectionStep<Object>::SelectionStep(const edm::ParameterSet& cfg,
                                     edm::ConsumesCollector&& iC)
    : select_(cfg.getParameter<std::string>("select")), jetIDSelect_(0) {

  src_ =
      iC.consumes<edm::View<Object> >(cfg.getParameter<edm::InputTag>("src"));
  // exist otherwise they are initialized with -1
  cfg.exists("min") ? min_ = cfg.getParameter<int>("min") : min_ = -1;
  cfg.exists("max") ? max_ = cfg.getParameter<int>("max") : max_ = -1;
  std::string mygSF = "gedGsfElectrons";
  gsfEs_ = iC.consumes<edm::View<reco::GsfElectron> >(
      cfg.getUntrackedParameter<edm::InputTag>("myGSF", mygSF));
  if (cfg.existsAs<edm::ParameterSet>("electronId")) {
    edm::ParameterSet elecId =
        cfg.getParameter<edm::ParameterSet>("electronId");
    electronId_ = iC.consumes<edm::ValueMap<float> >(
        elecId.getParameter<edm::InputTag>("src"));
    eidCutValue_ = elecId.getParameter<double>("cutValue");
  }
  if (cfg.exists("jetCorrector")) {
    jetCorrector_ = cfg.getParameter<std::string>("jetCorrector");
  }
  if (cfg.existsAs<edm::ParameterSet>("jetBTagger")) {
    edm::ParameterSet jetBTagger =
        cfg.getParameter<edm::ParameterSet>("jetBTagger");
    btagLabel_ = iC.consumes<reco::JetTagCollection>(
        jetBTagger.getParameter<edm::InputTag>("label"));
    btagWorkingPoint_ = jetBTagger.getParameter<double>("workingPoint");
  }
  if (cfg.existsAs<edm::ParameterSet>("jetID")) {
    edm::ParameterSet jetID = cfg.getParameter<edm::ParameterSet>("jetID");
    jetIDLabel_ = iC.consumes<reco::JetIDValueMap>(
        jetID.getParameter<edm::InputTag>("label"));
    jetIDSelect_ = new StringCutObjectSelector<reco::JetID>(
        jetID.getParameter<std::string>("select"));
  }
}

/// apply selection
template <typename Object>
bool SelectionStep<Object>::select(const edm::Event& event) {

  // fetch input collection
  edm::Handle<edm::View<Object> > src;
  if (!event.getByToken(src_, src)) return false;

  // load electronId value map if configured such
  edm::Handle<edm::ValueMap<float> > electronId;
  if (!electronId_.isUninitialized()) {
    if (!event.getByToken(electronId_, electronId)) return false;
  }

  // determine multiplicity of selected objects
  int n = 0;
  for (typename edm::View<Object>::const_iterator obj = src->begin();
       obj != src->end(); ++obj) {
    // special treatment for electrons
    if (dynamic_cast<const reco::GsfElectron*>(&*obj)) {
      unsigned int idx = obj - src->begin();
      if (electronId_.isUninitialized()
              ? true
              : ((double)(*electronId)[src->refAt(idx)] >= eidCutValue_)) {
        if (select_(*obj)) ++n;
      }
    }
    // normal treatment
    else {
      if (select_(*obj)) ++n;
    }
  }
  bool accept =
      (min_ >= 0 ? n >= min_ : true) && (max_ >= 0 ? n <= max_ : true);
  return (min_ < 0 && max_ < 0) ? (n > 0) : accept;
}

/// apply selection with special treatment for PFCandidates
template <typename Object>
bool SelectionStep<Object>::select(const edm::Event& event,
                                   const std::string& type) {
  // fetch input collection
  edm::Handle<edm::View<Object> > src;
  if (!event.getByToken(src_, src)) return false;

  // special for gsfElectron
  edm::Handle<edm::View<reco::GsfElectron> > elecs_gsf;

  // load electronId value map if configured such
  edm::Handle<edm::ValueMap<float> > electronId;
  if (!electronId_.isUninitialized()) {
    if (!event.getByToken(electronId_, electronId)) return false;
  }

  // determine multiplicity of selected objects
  int n = 0;
  for (typename edm::View<Object>::const_iterator obj = src->begin();
       obj != src->end(); ++obj) {

    // special treatment for PF candidates
    if (dynamic_cast<const reco::PFCandidate*>(&*obj)) {
      reco::PFCandidate objtmp = dynamic_cast<const reco::PFCandidate&>(*obj);

      if (objtmp.muonRef().isNonnull() && type == "muon") {
        if (select_(*obj)) {
          ++n;
        }
      } else if (objtmp.gsfElectronRef().isNonnull() && type == "electron") {
        if (select_(*obj)) {
          if (electronId_.isUninitialized()) {
            ++n;
          } else if (((double)(*electronId)[obj->gsfElectronRef()] >=
                      eidCutValue_)) {
            ++n;
          }
        }
        //        idx_gsf++;
      }
    }

    // special treatment for electrons
    else if (dynamic_cast<const reco::GsfElectron*>(&*obj)) {
      unsigned int idx = obj - src->begin();
      if (electronId_.isUninitialized()
              ? true
              : ((double)(*electronId)[src->refAt(idx)] >= eidCutValue_)) {
        if (select_(*obj)) ++n;
      }
    }

    // normal treatment
    else {
      if (select_(*obj)) ++n;
    }
  }
  bool accept =
      (min_ >= 0 ? n >= min_ : true) && (max_ >= 0 ? n <= max_ : true);
  return (min_ < 0 && max_ < 0) ? (n > 0) : accept;
}

template <typename Object>
bool SelectionStep<Object>::selectVertex(const edm::Event& event) {
  // fetch input collection
  edm::Handle<edm::View<Object> > src;
  if (!event.getByToken(src_, src)) return false;

  // load electronId value map if configured such
  edm::Handle<edm::ValueMap<float> > electronId;
  if (!electronId_.isUninitialized()) {
    if (!event.getByToken(electronId_, electronId)) return false;
  }

  // determine multiplicity of selected objects
  int n = 0;
  for (typename edm::View<Object>::const_iterator obj = src->begin();
       obj != src->end(); ++obj) {

    if (select_(*obj)) ++n;
  }
  bool accept =
      (min_ >= 0 ? n >= min_ : true) && (max_ >= 0 ? n <= max_ : true);
  return (min_ < 0 && max_ < 0) ? (n > 0) : accept;
}

/// apply selection (w/o using the template class Object), override for jets
template <typename Object>
bool SelectionStep<Object>::select(const edm::Event& event,
                                   const edm::EventSetup& setup) {
  // fetch input collection
  edm::Handle<edm::View<Object> > src;
  if (!event.getByToken(src_, src)) return false;

  // load btag collection if configured such
  // NOTE that the JetTagCollection needs an
  // edm::View to reco::Jets; we have to add
  // another Handle bjets for this purpose
  edm::Handle<edm::View<reco::Jet> > bjets;
  edm::Handle<reco::JetTagCollection> btagger;
  edm::Handle<edm::View<reco::Vertex> > pvertex;
  if (!btagLabel_.isUninitialized()) {
    if (!event.getByToken(src_, bjets)) return false;
    if (!event.getByToken(btagLabel_, btagger)) return false;
    if (!event.getByToken(pvs_, pvertex)) return false;
  }

  // load jetID value map if configured such
  edm::Handle<reco::JetIDValueMap> jetID;
  if (jetIDSelect_) {
    if (!event.getByToken(jetIDLabel_, jetID)) return false;
  }

  // load jet corrector if configured such
  const JetCorrector* corrector = 0;
  if (!jetCorrector_.empty()) {
    // check whether a jet correcto is in the event setup or not
    if (setup.find(edm::eventsetup::EventSetupRecordKey::makeKey<
            JetCorrectionsRecord>())) {
      corrector = JetCorrector::getJetCorrector(jetCorrector_, setup);
    } else {
      edm::LogVerbatim("TopDQMHelpers")
          << "\n"
          << "---------------------------------------------------------------\n"
          << " No JetCorrectionsRecord available from EventSetup:\n"
          << "  - Jets will not be corrected.\n"
          << "  - If you want to change this add the following lines to your "
             "cfg file:\n"
          << "\n"
          << "  ## load jet corrections\n"
          << "  process.load(\""
             "JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff"
             "\")\n"
          << "  process.prefer(\"ak5CaloL2L3\")\n"
          << "---------------------------------------------------------------"
             "\n";
    }
  }
  // determine multiplicity of selected objects
  int n = 0;
  for (typename edm::View<Object>::const_iterator obj = src->begin();
       obj != src->end(); ++obj) {
    // check for chosen btag discriminator to be above the
    // corresponding working point if configured such
    unsigned int idx = obj - src->begin();
    if (btagLabel_.isUninitialized() ? true : (*btagger)[bjets->refAt(idx)] >
                                                  btagWorkingPoint_) {
      bool passedJetID = true;
      // check jetID for calo jets
      if (jetIDSelect_ &&
          dynamic_cast<const reco::CaloJet*>(src->refAt(idx).get())) {
        passedJetID = (*jetIDSelect_)((*jetID)[src->refAt(idx)]);
      }
      if (passedJetID) {
        // scale jet energy if configured such
        Object jet = *obj;
        jet.scaleEnergy(corrector ? corrector->correction(*obj) : 1.);
        if (select_(jet)) ++n;
      }
    }
  }
  bool accept =
      (min_ >= 0 ? n >= min_ : true) && (max_ >= 0 ? n <= max_ : true);
  return (min_ < 0 && max_ < 0) ? (n > 0) : accept;
}

#endif

/* Local Variables: */
/* show-trailing-whitespace: t */
/* truncate-lines: t */
/* End: */
