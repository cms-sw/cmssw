#ifndef RecoJets_JetProducers_plugins_MyVirtualJetProducer_h
#define RecoJets_JetProducers_plugins_MyVirtualJetProducer_h


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/BasicJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "RecoJets/JetProducers/interface/PileUpSubtractor.h"

#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/ClusterSequenceArea.hh"
#include "fastjet/PseudoJet.hh"
#include "fastjet/GhostedAreaSpec.hh"

#include <vector>
#include <boost/shared_ptr.hpp>
#include "TNtuple.h"

class MyVirtualJetProducer : public edm::EDProducer
{
protected:
  //
  // typedefs & structs
  //
  struct JetType {
    enum Type {
      BasicJet,
      GenJet,
      CaloJet,
      PFJet,
      TrackJet,
      LastJetType  // no real type, technical
    };
    static const char *names[];
    static Type byName(const std::string &name);
  };

  JetType::Type jetTypeE;

  inline bool makeCaloJet(const JetType::Type &fTag) {
    return fTag == JetType::CaloJet;
  }
  inline bool makePFJet(const JetType::Type &fTag) {
    return fTag == JetType::PFJet;
  }
  inline bool makeGenJet(const JetType::Type &fTag) {
    return fTag == JetType::GenJet;
  }
  inline bool makeTrackJet(const JetType::Type &fTag) {
    return fTag == JetType::TrackJet;
  }
  inline bool makeBasicJet(const JetType::Type &fTag) {
    return fTag == JetType::BasicJet;
  }


  //
  // construction/destruction
  //
public:
  explicit MyVirtualJetProducer(const edm::ParameterSet& iConfig);
  virtual ~MyVirtualJetProducer();

  // typedefs
  typedef boost::shared_ptr<fastjet::ClusterSequence>        ClusterSequencePtr;
  typedef boost::shared_ptr<fastjet::JetDefinition::Plugin>  PluginPtr;
  typedef boost::shared_ptr<fastjet::JetDefinition>          JetDefPtr;
  typedef boost::shared_ptr<fastjet::GhostedAreaSpec>        GhostedAreaSpecPtr;
  typedef boost::shared_ptr<fastjet::RangeDefinition>        RangeDefPtr;

  TNtuple* ntuple;

  //
  // member functions
  //
public:
  virtual void  produce(edm::Event& iEvent, const edm::EventSetup& iSetup);
  std::string   jetType() const { return jetType_; }

protected:

  //
  // Internal methods for jet production.
  // The user can either use the defaults, or override all of these methods.
  //

  // This method creates the "produces" statement in the constructor.
  // The default is to produce a single jet collection as per the user's request
  // (Calo,PF,Basic, or Gen).
  virtual void makeProduces( std::string s, std::string tag = "" );

  // This method inputs the constituents from "inputs" and modifies
  // fjInputs.
  virtual void inputTowers();

  // This checks if the tower is anomalous (if a calo tower).
  virtual bool isAnomalousTower(reco::CandidatePtr input);

  // This will copy the fastjet constituents to the jet itself.
  virtual void copyConstituents(const std::vector<fastjet::PseudoJet>&fjConstituents,
				reco::Jet* jet);

  // This will run the actual algorithm. This method is pure virtual and
  // has no default.
  virtual void runAlgorithm( edm::Event& iEvent, const edm::EventSetup& iSetup) = 0;

  // Do the offset correction.
  // Only runs if "doPUOffsetCorrection_" is true.
  void offsetCorrectJets(std::vector<fastjet::PseudoJet> & orphanInput);

  // This will write the jets to the event.
  // The default is to write out the single jet collection in the default "produces"
  // statement.
  // This is a function template that can be called for the five types
  // CaloJet, PFJet, GenJet, TrackJet, BasicJet. This is not suitable
  // for compound jets.
  // Note: The "output" method is virtual and can be overriden.
  // The default behavior is to call the function template "writeJets".
  virtual void output(  edm::Event & iEvent, edm::EventSetup const& iSetup );
  template< typename T >
    void writeJets( edm::Event & iEvent, edm::EventSetup const& iSetup );

  // This method copies the constituents from the fjConstituents method
  // to an output of CandidatePtr's.
  virtual std::vector<reco::CandidatePtr>
    getConstituents(const std::vector<fastjet::PseudoJet>&fjConstituents);



  //
  // member data
  //
protected:
  std::string           moduleLabel_;               // label for this module
  edm::InputTag         src_;                       // input constituent source
  edm::InputTag         srcPVs_;                    // primary vertex source
  std::string           jetType_;                   // type of jet (Calo,PF,Basic,Gen)
  std::string           jetAlgorithm_;              // the jet algorithm to use
  double                rParam_;                    // the R parameter to use
  double                inputEtMin_;                // minimum et of input constituents
  double                inputEMin_;                 // minimum e of input constituents
  double                jetPtMin_;                  // minimum jet pt
  bool                  doPVCorrection_;            // correct to primary vertex?

  // for restricting inputs due to processing time
  bool                  restrictInputs_;            // restrict inputs to first "maxInputs" inputs.
  unsigned int          maxInputs_;                 // maximum number of inputs.

  // for fastjet jet area calculation
  bool                  doAreaFastjet_;             // calculate area w/ fastjet?
  // for fastjet rho calculation
  bool                  doRhoFastjet_;              // calculate rho w/ fastjet?
  bool                  doFastJetNonUniform_;       // choice of eta-dependent PU calculation

  // for pileup offset correction
  bool                  doPUOffsetCorr_;            // add the pileup calculation from offset correction?
  std::string           puSubtractorName_;

  // anomalous cell cuts
  unsigned int          maxBadEcalCells_;           // maximum number of bad ECAL cells
  unsigned int          maxRecoveredEcalCells_;     // maximum number of recovered ECAL cells
  unsigned int          maxProblematicEcalCells_;   // maximum number of problematic ECAL cells
  unsigned int          maxBadHcalCells_;           // maximum number of bad HCAL cells
  unsigned int          maxRecoveredHcalCells_;	    // maximum number of recovered HCAL cells
  unsigned int          maxProblematicHcalCells_;   // maximum number of problematic HCAL cells

  std::vector<edm::Ptr<reco::Candidate> > inputs_;  // input candidates [View, PtrVector and CandCollection have limitations]
  reco::Particle::Point           vertex_;          // Primary vertex
  ClusterSequencePtr              fjClusterSeq_;    // fastjet cluster sequence
  JetDefPtr                       fjJetDefinition_; // fastjet jet definition
  PluginPtr                       fjPlugin_;        // fastjet plugin
  GhostedAreaSpecPtr               fjActiveArea_;    // fastjet active area definition
  RangeDefPtr                     fjRangeDef_;      // range definition
  std::vector<fastjet::PseudoJet> fjInputs_;        // fastjet inputs
  std::vector<fastjet::PseudoJet> fjJets_;          // fastjet jets

  // Parameters of the eta-dependent rho calculation
  std::vector<double>             puCenters_;
  double                          puWidth_;

  std::string                     jetCollInstanceName_;       // instance name for output jet collection

public:
  boost::shared_ptr<PileUpSubtractor>  subtractor_;
  const CaloGeometry *geo;

};





#endif
