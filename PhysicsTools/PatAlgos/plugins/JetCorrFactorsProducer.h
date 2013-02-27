#ifndef PhysicsTools_PatAlgos_JetCorrFactorsProducer_h
#define PhysicsTools_PatAlgos_JetCorrFactorsProducer_h

/**
  \class    pat::JetCorrFactorsProducer JetCorrFactorsProducer.h "PhysicsTools/PatAlgos/interface/JetCorrFactorsProducer.h"
  \brief    Produces a ValueMap between JetCorrFactors and the to the originating reco jets

   The JetCorrFactorsProducer produces a set of correction factors, defined in the class pat::JetCorrFactors. This vector 
   is linked to the originating reco jets through an edm::ValueMap. The initializing parameters of the module can be found 
   in the recoLayer1/jetCorrFactors_cfi.py of the PatAlgos package. In the standard PAT workflow the module has to be run 
   before the creation of the pat::Jet. The edm::ValueMap will then be embedded into the pat::Jet. 

   Jets corrected up to a given correction level can then be accessed via the pat::Jet member function correctedJet. For 
   more details have a look into the class description of the pat::Jet.

   ATTENTION: available options for flavor corrections are 
    * L5Flavor_gJ        L7Parton_gJ         gluon   from dijets
    * L5Flavor_qJ/_qT    L7Parton_qJ/_qT     quark   from dijets/top
    * L5Flavor_cJ/_cT    L7Parton_cJ/_cT     charm   from dijets/top
    * L5Flavor_bJ/_bT    L7Parton_bJ/_bT     beauty  from dijets/top
    *                    L7Parton_jJ/_tT     mixture from dijets/top

   where mixture refers to the flavor mixture as determined from the MC sample the flavor dependent corrections have been
   derived from. 'J' and 'T' stand for a typical dijet (ttbar) sample. 

   L1Offset corrections require the collection of _offlinePrimaryVertices_, which are supposed to be added as an additional 
   optional parameter _primaryVertices_ in the jetCorrFactors_cfi.py file.  

   L1FastJet corrections, which are an alternative to the standard L1Offset correction as recommended by the JetMET PAG the 
   energy density parameter _rho_ is supposed to be added as an additional optional parameter _rho_ in the 
   jetCorrFactors_cfi.py file.  

   NOTE:
    * the mixed mode (mc input mixture from dijets/ttbar) only exists for parton level corrections.
    * jJ and tT are not covered in this implementation of the JetCorrFactorsProducer
    * there are no gluon corrections available from the top sample on the L7Parton level.
*/

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"


namespace pat {

  class JetCorrFactorsProducer : public edm::EDProducer {
  public:
    /// value map for JetCorrFactors (to be written into the event)
    typedef edm::ValueMap<pat::JetCorrFactors> JetCorrFactorsMap;
    /// map of correction levels to different flavors
    typedef std::map<JetCorrFactors::Flavor, std::vector<std::string> > FlavorCorrLevelMap;

  public:
    /// default constructor
    explicit JetCorrFactorsProducer(const edm::ParameterSet& cfg);
    /// default destructor
    ~JetCorrFactorsProducer() {};
    /// everything that needs to be done per event
    virtual void produce(edm::Event& event, const edm::EventSetup& setup) override;
    /// description of configuration file parameters
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
    
  private:
    /// return true if the jec levels contain at least one flavor dependent correction level
    bool flavorDependent() const { return (levels_.size()>1); }; 
    /// return the jec parameters as input to the FactorizedJetCorrector for different flavors
    std::vector<JetCorrectorParameters> params(const JetCorrectorParametersCollection& parameters, const std::vector<std::string>& levels) const;
    /// return an expanded version of correction levels for different flavors; the result should
    /// be of type ['L2Relative', 'L3Absolute', 'L5FLavor_gJ', 'L7Parton_gJ']; L7Parton_gT will 
    /// result in an empty string as this correction level is not available
    std::vector<std::string> expand(const std::vector<std::string>& levels, const JetCorrFactors::Flavor& flavor);
    /// evaluate jet correction factor up to a given level
    float evaluate(edm::View<reco::Jet>::const_iterator& jet, boost::shared_ptr<FactorizedJetCorrector>& corrector, boost::shared_ptr<FactorizedJetCorrector>& extraJPTOffset, int level);
    /// determines the number of valid primary vertices for the standard L1Offset correction of JetMET
    int numberOf(const edm::Handle<std::vector<reco::Vertex> >& primaryVertices);
    /// map jet algorithm to payload in DB
    std::string payload();
    
  private:
    /// use electromagnetic fraction for jet energy corrections or not (will only have an effect for jets CaloJets)
    bool emf_;
    /// input jet collection
    edm::InputTag src_;
    /// type of flavor dependent JEC factors (only 'J' and 'T' are allowed)
    std::string type_;
    /// label of jec factors
    std::string label_;
    /// label of payload
    std::string payload_;
    /// label of additional L1Offset corrector for JPT jets; for format reasons this string is
    /// kept in a vector of strings
    std::vector<std::string> extraJPTOffset_;
    /// label for L1Offset primaryVertex collection
    edm::InputTag primaryVertices_;
    /// label for L1FastJet energy density parameter rho
    edm::InputTag rho_;
    /// use the NPV and rho with the JEC? (used for L1Offset/L1FastJet and L1FastJet, resp.)
    bool useNPV_;
    bool useRho_;
    /// jec levels for different flavors. In the default configuration 
    /// this map would look like this:
    /// GLUON  : 'L2Relative', 'L3Absolute', 'L5FLavor_jg', L7Parton_jg'
    /// UDS    : 'L2Relative', 'L3Absolute', 'L5FLavor_jq', L7Parton_jq'
    /// CHARM  : 'L2Relative', 'L3Absolute', 'L5FLavor_jc', L7Parton_jc'
    /// BOTTOM : 'L2Relative', 'L3Absolute', 'L5FLavor_jb', L7Parton_jb'
    /// or just like this:
    /// NONE   : 'L2Relative', 'L3Absolute', 'L2L3Residual'
    /// per definition the vectors for all elements in this map should
    /// have the same size 
    FlavorCorrLevelMap levels_;
  };

  inline int 
  JetCorrFactorsProducer::numberOf(const edm::Handle<std::vector<reco::Vertex> >& primaryVertices)
  {
    int npv=0;
    for(std::vector<reco::Vertex>::const_iterator pv=primaryVertices->begin(); pv!=primaryVertices->end(); ++pv){
      if(pv->ndof()>=4) ++npv;
    }
    return npv;
  }
}

#endif
