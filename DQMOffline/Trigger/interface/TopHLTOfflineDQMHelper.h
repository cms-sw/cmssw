#ifndef TOPHLTOFFLINEDQMHELPER
#define TOPHLTOFFLINEDQMHELPER

#include <string>
#include <vector>
//#include <math>
#include "TString.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

/*Originally from DQM/Physics package, written by Roger Wolf and Jeremy Andrea*/
/**
   \fn      TopHLTOfflineDQMHelper.h 
   
   \brief   Helper function to determine trigger accepts.
   
   Helper function to determine trigger accept for given TriggerResults and 
   a given TriggerPath(s).
*/

inline bool 
acceptHLT(const edm::Event& event, const edm::TriggerResults& triggerTable, const std::string& triggerPath)
{
  bool passed=false;
  const edm::TriggerNames& triggerNames = event.triggerNames(triggerTable);
  for(unsigned int i=0; i<triggerNames.triggerNames().size(); ++i){
    TString name = triggerNames.triggerNames()[i].c_str();
    //std::cout << name << " " << triggerTable.accept(i) << std::endl;
    if(name.Contains(TString(triggerPath.c_str()), TString::kIgnoreCase)) {
      if(triggerTable.accept(i)){
	      passed=true;
	      break;
      }
    }
  }
  return passed;
}

inline bool 
acceptHLT(const edm::Event& event, const edm::TriggerResults& triggerTable, const std::vector<std::string>& triggerPaths)
{
  bool passed=false;
  for(unsigned int j=0; j<triggerPaths.size(); ++j){
    if(acceptHLT(event, triggerTable, triggerPaths[j])){
      passed=true;
      break;
    }
  }
  return passed;
}


#include "DataFormats/JetReco/interface/Jet.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/METReco/interface/CaloMET.h"
/**
   \class   Calculate TopHLTOfflineDQMHelper.h 

   \brief   Helper class for the calculation of a top and a W boson mass estime.
   
   Helper class for the calculation of a top and a W boson mass estimate. The
   core implementation originates from the plugin TtSemiLepHypMaxSumPtWMass 
   in TopQuarkAnalysis/TopJetCombination package. It may be extended to include
   b tag information.
*/

class CalculateHLT {
 public:
  /// default constructor
  CalculateHLT(int maxNJets, double wMass);
  /// default destructor
  ~CalculateHLT(){};
     
  /// calculate W boson mass estimate
  double massWBoson(const std::vector<reco::Jet>& jets);
  /// calculate top quark mass estimate
  double massTopQuark(const std::vector<reco::Jet>& jets); 
  /// calculate W boson transverse mass estimate
/*  double tmassWBoson(const T& mu, const reco::CaloMET& met, const reco::Jet& b);
  /// calculate top quark transverse mass estimate
  double tmassTopQuark(const T& mu, const reco::CaloMET& met, const reco::Jet& b);
  /// calculate mlb estimate
  double masslb(const T& mu, const reco::CaloMET& met, const reco::Jet& b);*/

  /// calculate W boson transverse mass estimate
  double tmassWBoson(reco::RecoCandidate* mu, const reco::MET& met, const reco::Jet& b);
  /// calculate top quark transverse mass estimate
  double tmassTopQuark(reco::RecoCandidate* mu, const reco::MET& met, const reco::Jet& b);
  /// calculate mlb estimate
  double masslb(reco::RecoCandidate* mu, const reco::MET& met, const reco::Jet& b);
  
 private:
  /// do the calculation; this is called only once per event by the first 
  /// function call to return a mass estimate. The once calculated values 
  /// are cached afterwards
  void operator()(const std::vector<reco::Jet>& jets);
  void operator()(const reco::Jet& bJet, reco::RecoCandidate* lepton, const reco::MET& met);
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
  /// cache of W boson transverse mass estimate
  double tmassWBoson_;
  /// cache of top quark transverse mass estimate
  double tmassTopQuark_;
  /// cache of mlb estimate
  double mlb_;


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
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

/**
   \class   SelectionStep TopHLTOfflineDQMHelper.h 
   
   \brief   Templated helper class to allow a selection on a certain object collection.
   
   Templated helper class to allow a selection on a certain object collection, which may 
   be monitored by a separate class afterwards. The class wraps and slightly extends the 
   features of the StringCutParser to allow also to apply event based selections, according 
   to a minimal or maximal number of elements in the collection after the object selection 
   has been applied. It takes an edm::ParameterSet in the constructor, which should contain
   the following elements:
   
    - src          : the input collection (mandatory).
    - select       : the selection string (mandatory).
    - min          : whether there is a min value on which to reject the whole event after 
                     the selection (optional).
    - max          : whether there is a max value on which to reject the whole event after 
                     the selection (optional).
    - electronId   : input tag of an electronId association map and selection pattern 
                     (optional). 
    - jetCorrector : label of jet corrector (optional).
    - jetBTagger   : parameters defining the btag algorithm and working point of choice
                     (optional).
    - jetID        : parameters defining the jetID value map and selection (optional).


   The parameters _src_ and _select_ are mandatory. The parameters _min_ and _max_ are 
   optional. The parameters _electronId_ and _jetCorrector_ are optional. They are added 
   to keep the possibility to apply selections on id'ed electrons or on corrected jets. 
   They may be omitted in the PSet for simplification reasons if not needed at any time. 
   They are not effiective for other object collections but electrons or jets. If none
   of the two parameters _min_ or _max_ is found in the event the select function returns 
   true if at least one object fullfilled the requirements.

   The class has one template value, which is the object collection to apply the selection 
   on. This has to be parsed to the StringCutParser class. The function select is overrided 
   for jets to circumvent problems with the template specialisation. Note that for MET not 
   type1 or muon corrections are supported on reco candidates.
*/
class SelectionStepHLTBase {
  public:
    virtual bool select(const edm::Event& event) {
      return false;
    };

    virtual bool select(const edm::Event& event, const edm::EventSetup& setup) {
      assert(false);
      return false;
    };
};

template <typename Object> 
class SelectionStepHLT: public SelectionStepHLTBase {
public:
  /// default constructor
  SelectionStepHLT(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC);
  /// default destructor
  virtual ~SelectionStepHLT(){};

  /// apply selection
  virtual bool select(const edm::Event& event);
  /// apply selection override for jets
  virtual bool select(const edm::Event& event, const edm::EventSetup& setup); 
  bool selectVertex(const edm::Event& event);
private:
  /// input collection
  edm::EDGetTokenT< edm::View<Object> > src_;
  /// min/max for object multiplicity
  int min_, max_; 
  /// electronId label as extra selection type
  edm::EDGetTokenT< edm::ValueMap<float> > electronId_;
  /// electronId pattern we expect the following pattern:
  ///  0: fails
  ///  1: passes electron ID only
  ///  2: passes electron Isolation only
  ///  3: passes electron ID and Isolation only
  ///  4: passes conversion rejection
  ///  5: passes conversion rejection and ID
  ///  6: passes conversion rejection and Isolation
  ///  7: passes the whole selection
  /// As described on https://twiki.cern.ch/twiki/bin/view/CMS/SimpleCutBasedEleID
  int eidPattern_;
  /// jet corrector as extra selection type
  std::string jetCorrector_;
  /// choice for b-tag as extra selection type
  edm::EDGetTokenT< reco::JetTagCollection > btagLabel_;
  /// choice of b-tag working point as extra selection type
  double btagWorkingPoint_;
  /// jetID as an extra selection type 
  edm::EDGetTokenT< reco::JetIDValueMap > jetIDLabel_;

  edm::EDGetTokenT< edm::View<reco::Vertex> > pvs_; 

  /// string cut selector
  StringCutObjectSelector<Object> select_;
  /// selection string on the jetID
  StringCutObjectSelector<reco::JetID>* jetIDSelect_;
};

/// default constructor
template <typename Object> 
SelectionStepHLT<Object>::SelectionStepHLT(const edm::ParameterSet& cfg, edm::ConsumesCollector&& iC) :
  src_( iC.consumes< edm::View<Object> >(cfg.getParameter<edm::InputTag>( "src"   ))),
  select_( cfg.getParameter<std::string>("select")),
  jetIDSelect_(0)
{
  // construct min/max if the corresponding params
  // exist otherwise they are initialized with -1
  cfg.exists("min") ? min_= cfg.getParameter<int>("min") : min_= -1;
  cfg.exists("max") ? max_= cfg.getParameter<int>("max") : max_= -1;
  // read electron extras if they exist
  if(cfg.existsAs<edm::ParameterSet>("electronId")){ 
    edm::ParameterSet elecId=cfg.getParameter<edm::ParameterSet>("electronId");
    electronId_= iC.consumes< edm::ValueMap<float> >(elecId.getParameter<edm::InputTag>("src"));
    eidPattern_= elecId.getParameter<int>("pattern");
  }
  // read jet corrector label if it exists
  if(cfg.exists("jetCorrector")){ jetCorrector_= cfg.getParameter<std::string>("jetCorrector"); }
  // read btag information if it exists
  if(cfg.existsAs<edm::ParameterSet>("jetBTagger")){
    edm::ParameterSet jetBTagger=cfg.getParameter<edm::ParameterSet>("jetBTagger");
    btagLabel_= iC.consumes< reco::JetTagCollection >(jetBTagger.getParameter<edm::InputTag>("label"));
    btagWorkingPoint_=jetBTagger.getParameter<double>("workingPoint");
  }
  // read jetID information if it exists
  if(cfg.existsAs<edm::ParameterSet>("jetID")){
    edm::ParameterSet jetID=cfg.getParameter<edm::ParameterSet>("jetID");
    jetIDLabel_ = iC.consumes< reco::JetIDValueMap >(jetID.getParameter<edm::InputTag>("label"));
    jetIDSelect_= new StringCutObjectSelector<reco::JetID>(jetID.getParameter<std::string>("select"));
  }
}

/// apply selection
template <typename Object> 
bool SelectionStepHLT<Object>::select(const edm::Event& event)
{
  // fetch input collection
  edm::Handle<edm::View<Object> > src; 
  if( !event.getByToken(src_, src) ) return false;

  // load electronId value map if configured such
  edm::Handle<edm::ValueMap<float> > electronId;
  if(!electronId_.isUninitialized()) {
    if( !event.getByToken(electronId_, electronId) ) return false;
  }

  // determine multiplicity of selected objects
  int n=0;
  for(typename edm::View<Object>::const_iterator obj=src->begin(); obj!=src->end(); ++obj){
    // special treatment for electrons
    if(dynamic_cast<const reco::GsfElectron*>(&*obj)){
      unsigned int idx = obj-src->begin();
      if( electronId_.isUninitialized() ? true : ((int)(*electronId)[src->refAt(idx)] & eidPattern_) ){   
	if(select_(*obj))++n;
      }
    }
    // normal treatment
    else{
      if(select_(*obj))++n;
    }
  }
  bool accept=(min_>=0 ? n>=min_:true) && (max_>=0 ? n<=max_:true);
  return (min_<0 && max_<0) ? (n>0):accept;
}
template <typename Object> 
bool SelectionStepHLT<Object>::selectVertex(const edm::Event& event)
{
  // fetch input collection
  edm::Handle<edm::View<Object> > src; 
  if( !event.getByToken(src_, src) ) return false;

  // load electronId value map if configured such
  edm::Handle<edm::ValueMap<float> > electronId;
  if(!electronId_.isUninitialized()) {
    if( !event.getByToken(electronId_, electronId) ) return false;
  }

  // determine multiplicity of selected objects
  int n=0;
  for(typename edm::View<Object>::const_iterator obj=src->begin(); obj!=src->end(); ++obj){
   
      if(select_(*obj))++n;
  }
  bool accept=(min_>=0 ? n>=min_:true) && (max_>=0 ? n<=max_:true);
  return (min_<0 && max_<0) ? (n>0):accept;
}

template <typename Object> 
bool SelectionStepHLT<Object>::select(const edm::Event& event, const edm::EventSetup& setup)
{
  throw cms::Exception("SelectionStepHLT") << "you fail" << std::endl;
  return false;
}
#endif
