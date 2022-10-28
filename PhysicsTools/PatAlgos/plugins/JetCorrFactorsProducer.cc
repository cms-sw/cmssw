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

#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

namespace pat {

  class JetCorrFactorsProducer : public edm::stream::EDProducer<> {
  public:
    /// value map for JetCorrFactors (to be written into the event)
    typedef edm::ValueMap<pat::JetCorrFactors> JetCorrFactorsMap;
    /// map of correction levels to different flavors
    typedef std::map<JetCorrFactors::Flavor, std::vector<std::string> > FlavorCorrLevelMap;

  public:
    /// default constructor
    explicit JetCorrFactorsProducer(const edm::ParameterSet& cfg);
    /// default destructor
    ~JetCorrFactorsProducer() override{};
    /// everything that needs to be done per event
    void produce(edm::Event& event, const edm::EventSetup& setup) override;
    /// description of configuration file parameters
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    /// return true if the jec levels contain at least one flavor dependent correction level
    bool flavorDependent() const { return (levels_.size() > 1); };
    /// return the jec parameters as input to the FactorizedJetCorrector for different flavors
    std::vector<JetCorrectorParameters> params(const JetCorrectorParametersCollection& parameters,
                                               const std::vector<std::string>& levels) const;
    /// return an expanded version of correction levels for different flavors; the result should
    /// be of type ['L2Relative', 'L3Absolute', 'L5FLavor_gJ', 'L7Parton_gJ']; L7Parton_gT will
    /// result in an empty string as this correction level is not available
    std::vector<std::string> expand(const std::vector<std::string>& levels, const JetCorrFactors::Flavor& flavor);
    /// evaluate jet correction factor up to a given level
    float evaluate(edm::View<reco::Jet>::const_iterator& jet, const JetCorrFactors::Flavor& flavor, int level);
    /// determines the number of valid primary vertices for the standard L1Offset correction of JetMET
    int numberOf(const edm::Handle<std::vector<reco::Vertex> >& primaryVertices);
    /// map jet algorithm to payload in DB

  private:
    /// use electromagnetic fraction for jet energy corrections or not (will only have an effect for jets CaloJets)
    bool emf_;
    /// input jet collection
    edm::EDGetTokenT<edm::View<reco::Jet> > srcToken_;
    /// type of flavor dependent JEC factors (only 'J' and 'T' are allowed)
    std::string type_;
    /// label of jec factors
    std::string label_;
    /// label of additional L1Offset corrector for JPT jets; for format reasons this string is
    /// kept in a vector of strings
    std::vector<std::string> extraJPTOffset_;
    /// label for L1Offset primaryVertex collection
    edm::InputTag primaryVertices_;
    edm::EDGetTokenT<std::vector<reco::Vertex> > primaryVerticesToken_;
    /// label for L1FastJet energy density parameter rho
    edm::InputTag rho_;
    edm::EDGetTokenT<double> rhoToken_;
    const edm::ESGetToken<JetCorrectorParametersCollection, JetCorrectionsRecord> parametersToken_;
    edm::ESWatcher<JetCorrectionsRecord> parametersWatcher_;
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
    /// cache container for jet corrections
    std::map<JetCorrFactors::Flavor, std::unique_ptr<FactorizedJetCorrector> > correctors_;
    /// cache container for JPTOffset jet corrections
    std::unique_ptr<FactorizedJetCorrector> extraJPTOffsetCorrector_;
  };

  inline int JetCorrFactorsProducer::numberOf(const edm::Handle<std::vector<reco::Vertex> >& primaryVertices) {
    int npv = 0;
    for (auto const& pv : *primaryVertices) {
      if (pv.ndof() >= 4)
        ++npv;
    }
    return npv;
  }
}  // namespace pat

using namespace pat;

JetCorrFactorsProducer::JetCorrFactorsProducer(const edm::ParameterSet& cfg)
    : emf_(cfg.getParameter<bool>("emf")),
      srcToken_(consumes(cfg.getParameter<edm::InputTag>("src"))),
      type_(cfg.getParameter<std::string>("flavorType")),
      label_(cfg.getParameter<std::string>("@module_label")),
      parametersToken_{esConsumes(edm::ESInputTag("", cfg.getParameter<std::string>("payload")))},
      useNPV_(cfg.getParameter<bool>("useNPV")),
      useRho_(cfg.getParameter<bool>("useRho")) {
  std::vector<std::string> levels = cfg.getParameter<std::vector<std::string> >("levels");
  // fill the std::map for levels_, which might be flavor dependent or not;
  // flavor dependency is determined from the fact whether the std::string
  // L5Flavor or L7Parton can be found in levels; if flavor dependent four
  // vectors of strings will be filled into the map corresponding to GLUON,
  // UDS, CHARM and BOTTOM (according to JetCorrFactors::Flavor), 'L5Flavor'
  // and 'L7Parton' will be expanded accordingly; if not levels_ is filled
  // with only one vector of strings according to NONE. This vector will be
  // equivalent to the original vector of strings.
  if (std::find(levels.begin(), levels.end(), "L5Flavor") != levels.end() ||
      std::find(levels.begin(), levels.end(), "L7Parton") != levels.end()) {
    levels_[JetCorrFactors::GLUON] = expand(levels, JetCorrFactors::GLUON);
    levels_[JetCorrFactors::UDS] = expand(levels, JetCorrFactors::UDS);
    levels_[JetCorrFactors::CHARM] = expand(levels, JetCorrFactors::CHARM);
    levels_[JetCorrFactors::BOTTOM] = expand(levels, JetCorrFactors::BOTTOM);
  } else {
    levels_[JetCorrFactors::NONE] = levels;
  }
  // if the std::string L1JPTOffset can be found in levels an additional
  // parameter extraJPTOffset is needed, which should pass on the the usual
  // L1Offset correction, which is an additional input to the L1JPTOffset
  // corrector
  if (std::find(levels.begin(), levels.end(), "L1JPTOffset") != levels.end()) {
    extraJPTOffset_.push_back(cfg.getParameter<std::string>("extraJPTOffset"));
  }
  // if the std::string L1Offset can be found in levels an additional para-
  // meter primaryVertices is needed, which should pass on the offline pri-
  // mary vertex collection. The size of this collection is needed for the
  // L1Offset correction.
  if (useNPV_) {
    primaryVertices_ = cfg.getParameter<edm::InputTag>("primaryVertices");
    primaryVerticesToken_ = mayConsume<std::vector<reco::Vertex> >(primaryVertices_);
  }
  // if the std::string L1FastJet can be found in levels an additional
  // parameter rho is needed, which should pass on the energy density
  // parameter for the corresponding jet collection.
  if (useRho_) {
    if ((!extraJPTOffset_.empty() && extraJPTOffset_.front() == std::string("L1FastJet")) ||
        std::find(levels.begin(), levels.end(), "L1FastJet") != levels.end()) {
      rho_ = cfg.getParameter<edm::InputTag>("rho");
      rhoToken_ = mayConsume<double>(rho_);
    } else {
      edm::LogInfo message("Parameter rho not used");
      message << "Module is configured to use the parameter rho, but rho is only used     \n"
              << "for L1FastJet corrections. The configuration of levels does not contain \n"
              << "L1FastJet corrections though, so rho will not be used by this module.   \n";
    }
  }
  produces<JetCorrFactorsMap>();
}

std::vector<std::string> JetCorrFactorsProducer::expand(const std::vector<std::string>& levels,
                                                        const JetCorrFactors::Flavor& flavor) {
  std::vector<std::string> expand;
  for (std::vector<std::string>::const_iterator level = levels.begin(); level != levels.end(); ++level) {
    if ((*level) == "L5Flavor" || (*level) == "L7Parton") {
      if (flavor == JetCorrFactors::GLUON) {
        if (*level == "L7Parton" && type_ == "T") {
          edm::LogWarning message("L7Parton::GLUON not available");
          message << "Jet energy corrections requested for level: L7Parton and type: 'T'. \n"
                  << "For this combination there is no GLUON correction available. The    \n"
                  << "correction for this flavor type will be taken from 'J'.";
        }
        expand.push_back(std::string(*level).append("_").append("g").append("J"));
      }
      if (flavor == JetCorrFactors::UDS)
        expand.push_back(std::string(*level).append("_").append("q").append(type_));
      if (flavor == JetCorrFactors::CHARM)
        expand.push_back(std::string(*level).append("_").append("c").append(type_));
      if (flavor == JetCorrFactors::BOTTOM)
        expand.push_back(std::string(*level).append("_").append("b").append(type_));
    } else {
      expand.push_back(*level);
    }
  }
  return expand;
}

std::vector<JetCorrectorParameters> JetCorrFactorsProducer::params(const JetCorrectorParametersCollection& parameters,
                                                                   const std::vector<std::string>& levels) const {
  std::vector<JetCorrectorParameters> params;
  for (std::vector<std::string>::const_iterator level = levels.begin(); level != levels.end(); ++level) {
    const JetCorrectorParameters& ip = parameters[*level];  //ip.printScreen();
    params.push_back(ip);
  }
  return params;
}

float JetCorrFactorsProducer::evaluate(edm::View<reco::Jet>::const_iterator& jet,
                                       const JetCorrFactors::Flavor& flavor,
                                       int level) {
  std::unique_ptr<FactorizedJetCorrector>& corrector = correctors_.find(flavor)->second;
  // add parameters for JPT corrections
  const reco::JPTJet* jpt = dynamic_cast<reco::JPTJet const*>(&*jet);
  if (jpt) {
    TLorentzVector p4;
    p4.SetPtEtaPhiE(jpt->getCaloJetRef()->pt(),
                    jpt->getCaloJetRef()->eta(),
                    jpt->getCaloJetRef()->phi(),
                    jpt->getCaloJetRef()->energy());
    if (extraJPTOffsetCorrector_) {
      extraJPTOffsetCorrector_->setJPTrawP4(p4);
      corrector->setJPTrawOff(extraJPTOffsetCorrector_->getSubCorrections()[0]);
    }
    corrector->setJPTrawP4(p4);
  }
  //For PAT jets undo previous jet energy corrections
  const Jet* patjet = dynamic_cast<Jet const*>(&*jet);
  if (patjet) {
    corrector->setJetEta(patjet->correctedP4(0).eta());
    corrector->setJetPt(patjet->correctedP4(0).pt());
    corrector->setJetPhi(patjet->correctedP4(0).phi());
    corrector->setJetE(patjet->correctedP4(0).energy());
  } else {
    corrector->setJetEta(jet->eta());
    corrector->setJetPt(jet->pt());
    corrector->setJetPhi(jet->phi());
    corrector->setJetE(jet->energy());
  }
  if (emf_ && dynamic_cast<const reco::CaloJet*>(&*jet)) {
    corrector->setJetEMF(dynamic_cast<const reco::CaloJet*>(&*jet)->emEnergyFraction());
  }
  return corrector->getSubCorrections()[level];
}

void JetCorrFactorsProducer::produce(edm::Event& event, const edm::EventSetup& setup) {
  // get jet collection from the event
  edm::Handle<edm::View<reco::Jet> > jets;
  event.getByToken(srcToken_, jets);

  // get primary vertices for L1Offset correction level if needed
  edm::Handle<std::vector<reco::Vertex> > primaryVertices;
  if (!primaryVertices_.label().empty())
    event.getByToken(primaryVerticesToken_, primaryVertices);

  // get parameter rho for L1FastJet correction level if needed
  edm::Handle<double> rho;
  if (!rho_.label().empty())
    event.getByToken(rhoToken_, rho);

  if (parametersWatcher_.check(setup)) {
    // retreive parameters from the DB
    auto const& parameters = setup.getData(parametersToken_);
    // initialize jet correctors
    for (auto const& flavor : levels_) {
      correctors_[flavor.first] = std::make_unique<FactorizedJetCorrector>(params(parameters, flavor.second));
    }
    // initialize extra jet corrector for jpt if needed
    if (!extraJPTOffset_.empty()) {
      extraJPTOffsetCorrector_ = std::make_unique<FactorizedJetCorrector>(params(parameters, extraJPTOffset_));
    }
  }

  // fill the jetCorrFactors
  std::vector<JetCorrFactors> jcfs;
  for (edm::View<reco::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); ++jet) {
    // the JetCorrFactors::CorrectionFactor is a std::pair<std::string, std::vector<float> >
    // the string corresponds to the label of the correction level, the vector contains four
    // floats if flavor dependent and one float else. Per construction jet energy corrections
    // will be flavor independent up to the first flavor dependent correction and flavor de-
    // pendent afterwards. The first correction level is predefined with label 'Uncorrected'.
    // Per definition it is flavor independent. The correction factor is 1.
    std::vector<JetCorrFactors::CorrectionFactor> jec;
    jec.emplace_back("Uncorrected", std::vector<float>(1, 1));

    // pick the first element in the map (which could be the only one) and loop all jec
    // levels listed for that element. If this is not the only element all jec levels, which
    // are flavor independent will give the same correction factors until the first flavor
    // dependent correction level is reached. So the first element is still a good choice.
    FlavorCorrLevelMap::const_iterator corrLevel = levels_.begin();
    if (corrLevel == levels_.end()) {
      throw cms::Exception("No JECFactors")
          << "You request to create a jetCorrFactors object with no JEC Levels indicated. \n"
          << "This makes no sense, either you should correct this or drop the module from \n"
          << "the sequence.";
    }
    for (unsigned int idx = 0; idx < corrLevel->second.size(); ++idx) {
      std::vector<float> factors;
      if (corrLevel->second[idx].find("L5Flavor") != std::string::npos ||
          corrLevel->second[idx].find("L7Parton") != std::string::npos) {
        for (FlavorCorrLevelMap::const_iterator flavor = corrLevel; flavor != levels_.end(); ++flavor) {
          if (!primaryVertices_.label().empty()) {
            // if primaryVerticesToken_ has a value the number of primary vertices needs to be
            // specified
            correctors_.find(flavor->first)->second->setNPV(numberOf(primaryVertices));
          }
          if (!rho_.label().empty()) {
            // if rhoToken_ has a value the energy density parameter rho and the jet area need
            //  to be specified
            correctors_.find(flavor->first)->second->setRho(*rho);
            correctors_.find(flavor->first)->second->setJetA(jet->jetArea());
          }
          factors.push_back(evaluate(jet, flavor->first, idx));
        }
      } else {
        if (!primaryVertices_.label().empty()) {
          // if primaryVerticesToken_ has a value the number of primary vertices needs to be
          // specified
          correctors_.find(corrLevel->first)->second->setNPV(numberOf(primaryVertices));
        }
        if (!rho_.label().empty()) {
          // if rhoToken_ has a value the energy density parameter rho and the jet area need
          // to be specified
          correctors_.find(corrLevel->first)->second->setRho(*rho);
          correctors_.find(corrLevel->first)->second->setJetA(jet->jetArea());
        }
        factors.push_back(evaluate(jet, corrLevel->first, idx));
      }
      // push back the set of JetCorrFactors: the first entry corresponds to the label
      // of the correction level, which is taken from the first element in levels_. For
      // L5Flavor and L7Parton the part including the first '_' indicating the flavor
      // of the first element in levels_ is chopped of from the label to avoid confusion
      // of the correction levels. The second parameter corresponds to the set of jec
      // factors, which might be flavor dependent or not. In the default configuration
      // the CorrectionFactor will look like this: 'Uncorrected': 1 ; 'L2Relative': x ;
      // 'L3Absolute': x ; 'L5Flavor': v, x, y, z ; 'L7Parton': v, x, y, z
      jec.emplace_back((corrLevel->second[idx]).substr(0, (corrLevel->second[idx]).find('_')), factors);
    }
    // create the actual object with the scale factors we want the valuemap to refer to
    // label_ corresponds to the label of the module instance
    jcfs.emplace_back(label_, jec);
  }
  // build the value map
  auto jetCorrsMap = std::make_unique<JetCorrFactorsMap>();
  JetCorrFactorsMap::Filler filler(*jetCorrsMap);
  // jets and jetCorrs have their indices aligned by construction
  filler.insert(jets, jcfs.begin(), jcfs.end());
  filler.fill();  // do the actual filling
  // put our produced stuff in the event
  event.put(std::move(jetCorrsMap));
}

void JetCorrFactorsProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.add<bool>("emf", false);
  iDesc.add<std::string>("flavorType", "J");
  iDesc.add<edm::InputTag>("src", edm::InputTag("ak5CaloJets"));
  iDesc.add<std::string>("payload", "AK5Calo");
  iDesc.add<bool>("useNPV", true);
  iDesc.add<edm::InputTag>("primaryVertices", edm::InputTag("offlinePrimaryVertices"));
  iDesc.add<bool>("useRho", true);
  iDesc.add<edm::InputTag>("rho", edm::InputTag("fixedGridRhoFastjetAllCalo"));
  iDesc.add<std::string>("extraJPTOffset", "L1Offset");

  iDesc.add<std::vector<std::string> >("levels",
                                       {
                                           "L1Offset",
                                           "L2Relative",
                                           "L3Absolute",
                                           "L2L3Residual",
                                           "L5Flavor",
                                           "L7Parton",
                                       });
  descriptions.add("JetCorrFactorsProducer", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(JetCorrFactorsProducer);
