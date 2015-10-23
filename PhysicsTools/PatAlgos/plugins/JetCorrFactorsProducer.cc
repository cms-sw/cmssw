#include <vector>
#include <string>
#include <iostream>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/PatAlgos/plugins/JetCorrFactorsProducer.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"

#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"


using namespace pat;

JetCorrFactorsProducer::JetCorrFactorsProducer(const edm::ParameterSet& cfg):
  emf_(cfg.getParameter<bool>( "emf" )),
  srcToken_(consumes<edm::View<reco::Jet> >(cfg.getParameter<edm::InputTag>( "src" ))),
  type_ (cfg.getParameter<std::string>("flavorType")),
  label_(cfg.getParameter<std::string>( "@module_label" )),
  payload_( cfg.getParameter<std::string>("payload") ),
  useNPV_(cfg.getParameter<bool>("useNPV")),
  useRho_(cfg.getParameter<bool>("useRho")),
  cacheId_(0)
{
  std::vector<std::string> levels = cfg.getParameter<std::vector<std::string> >("levels");
  // fill the std::map for levels_, which might be flavor dependent or not;
  // flavor dependency is determined from the fact whether the std::string
  // L5Flavor or L7Parton can be found in levels; if flavor dependent four
  // vectors of strings will be filled into the map corresponding to GLUON,
  // UDS, CHARM and BOTTOM (according to JetCorrFactors::Flavor), 'L5Flavor'
  // and 'L7Parton' will be expanded accordingly; if not levels_ is filled
  // with only one vector of strings according to NONE. This vector will be
  // equivalent to the original vector of strings.
  if(std::find(levels.begin(), levels.end(), "L5Flavor")!=levels.end() || std::find(levels.begin(), levels.end(), "L7Parton")!=levels.end()){
    levels_[JetCorrFactors::GLUON ] = expand(levels, JetCorrFactors::GLUON );
    levels_[JetCorrFactors::UDS   ] = expand(levels, JetCorrFactors::UDS   );
    levels_[JetCorrFactors::CHARM ] = expand(levels, JetCorrFactors::CHARM );
    levels_[JetCorrFactors::BOTTOM] = expand(levels, JetCorrFactors::BOTTOM);
  }
  else{
    levels_[JetCorrFactors::NONE  ] = levels;
  }
  // if the std::string L1JPTOffset can be found in levels an additional
  // parameter extraJPTOffset is needed, which should pass on the the usual
  // L1Offset correction, which is an additional input to the L1JPTOffset
  // corrector
  if(std::find(levels.begin(), levels.end(), "L1JPTOffset")!=levels.end()){
    if(cfg.existsAs<std::string>("extraJPTOffset")){
      extraJPTOffset_.push_back(cfg.getParameter<std::string>("extraJPTOffset"));
    }
    else{
      throw cms::Exception("No parameter extraJPTOffset specified")
	<< "The configured correction levels contain a L1JPTOffset correction, which re- \n"
	<< "quires the additional parameter extraJPTOffset or type std::string. This     \n"
	<< "string should correspond to the L1Offset corrections that should be applied  \n"
	<< "together with the JPTL1Offset corrections. These corrections can be of type  \n"
	<< "L1Offset or L1FastJet.                                                       \n";
    }
  }
  // if the std::string L1Offset can be found in levels an additional para-
  // meter primaryVertices is needed, which should pass on the offline pri-
  // mary vertex collection. The size of this collection is needed for the
  // L1Offset correction.
  if(useNPV_){
    if(cfg.existsAs<edm::InputTag>("primaryVertices")){
      primaryVertices_=cfg.getParameter<edm::InputTag>("primaryVertices");
      primaryVerticesToken_=mayConsume<std::vector<reco::Vertex> >(primaryVertices_);
    }
    else{
      throw cms::Exception("No primaryVertices specified")
	<< "The configured correction levels contain an L1Offset or L1FastJet correction, \n"
	<< "which requires the number of offlinePrimaryVertices. Please specify this col- \n"
	<< "lection as additional optional parameter primaryVertices of type edm::InputTag\n"
	<< "in the jetCorrFactors module.                                                 \n";
    }
  }
  // if the std::string L1FastJet can be found in levels an additional
  // parameter rho is needed, which should pass on the energy density
  // parameter for the corresponding jet collection.
  if(useRho_){
    if((!extraJPTOffset_.empty() && extraJPTOffset_.front()==std::string("L1FastJet")) || std::find(levels.begin(), levels.end(), "L1FastJet")!=levels.end()){
      if(cfg.existsAs<edm::InputTag>("rho")){
	rho_=cfg.getParameter<edm::InputTag>("rho");
	rhoToken_=mayConsume<double>(rho_);
      }
      else{
	throw cms::Exception("No parameter rho specified")
	  << "The configured correction levels contain a L1FastJet correction, which re- \n"
	  << "quires the energy density parameter rho. Please specify this collection as \n"
	  << "additional optional parameter rho of type edm::InputTag in the jetCorrFac- \n"
	  << "tors module.                                                               \n";
      }
    }
    else{
      edm::LogWarning message( "Parameter rho not used" );
      message << "Module is configured to use the parameter rho, but rho is only used     \n"
	      << "for L1FastJet corrections. The configuration of levels does not contain \n"
	      << "L1FastJet corrections though, so rho will not be used by this module.   \n";
    }
  }
  produces<JetCorrFactorsMap>();
}

std::vector<std::string>
JetCorrFactorsProducer::expand(const std::vector<std::string>& levels, const JetCorrFactors::Flavor& flavor)
{
  std::vector<std::string> expand;
  for(std::vector<std::string>::const_iterator level=levels.begin(); level!=levels.end(); ++level){
    if((*level)=="L5Flavor" || (*level)=="L7Parton"){
      if(flavor==JetCorrFactors::GLUON ){
	if(*level=="L7Parton" && type_=="T"){
	  edm::LogWarning message( "L7Parton::GLUON not available" );
	  message << "Jet energy corrections requested for level: L7Parton and type: 'T'. \n"
		  << "For this combination there is no GLUON correction available. The    \n"
		  << "correction for this flavor type will be taken from 'J'.";
	}
	expand.push_back(std::string(*level).append("_").append("g").append("J"));
      }
      if(flavor==JetCorrFactors::UDS   ) expand.push_back(std::string(*level).append("_").append("q").append(type_));
      if(flavor==JetCorrFactors::CHARM ) expand.push_back(std::string(*level).append("_").append("c").append(type_));
      if(flavor==JetCorrFactors::BOTTOM) expand.push_back(std::string(*level).append("_").append("b").append(type_));
    }
    else{
      expand.push_back(*level);
    }
  }
  return expand;
}

std::vector<JetCorrectorParameters>
JetCorrFactorsProducer::params(const JetCorrectorParametersCollection& parameters, const std::vector<std::string>& levels) const
{
  std::vector<JetCorrectorParameters> params;
  for(std::vector<std::string>::const_iterator level=levels.begin(); level!=levels.end(); ++level){
    const JetCorrectorParameters& ip = parameters[*level]; //ip.printScreen();
    params.push_back(ip);
  }
  return params;
}

float
JetCorrFactorsProducer::evaluate(edm::View<reco::Jet>::const_iterator& jet, const JetCorrFactors::Flavor& flavor, int level)
{
  std::unique_ptr<FactorizedJetCorrector>& corrector = correctors_.find(flavor)->second;
  // add parameters for JPT corrections
  const reco::JPTJet* jpt = dynamic_cast<reco::JPTJet const *>( &*jet );
  if( jpt ){
    TLorentzVector p4; p4.SetPtEtaPhiE(jpt->getCaloJetRef()->pt(), jpt->getCaloJetRef()->eta(), jpt->getCaloJetRef()->phi(), jpt->getCaloJetRef()->energy());
    if( extraJPTOffsetCorrector_ ){
      extraJPTOffsetCorrector_->setJPTrawP4(p4);
      corrector->setJPTrawOff(extraJPTOffsetCorrector_->getSubCorrections()[0]);
    }
    corrector->setJPTrawP4(p4);
  }
  //For PAT jets undo previous jet energy corrections
  const Jet* patjet = dynamic_cast<Jet const *>( &*jet );
  if( patjet ){
    corrector->setJetEta(patjet->correctedP4(0).eta()); corrector->setJetPt(patjet->correctedP4(0).pt()); corrector->setJetE(patjet->correctedP4(0).energy());
  } else {
    corrector->setJetEta(jet->eta()); corrector->setJetPt(jet->pt()); corrector->setJetE(jet->energy());
  }
  if( emf_ && dynamic_cast<const reco::CaloJet*>(&*jet)){
    corrector->setJetEMF(dynamic_cast<const reco::CaloJet*>(&*jet)->emEnergyFraction());
  }
  return corrector->getSubCorrections()[level];
}

std::string
JetCorrFactorsProducer::payload()
{
  return payload_;
}

void
JetCorrFactorsProducer::produce(edm::Event& event, const edm::EventSetup& setup)
{
  // get jet collection from the event
  edm::Handle<edm::View<reco::Jet> > jets;
  event.getByToken(srcToken_, jets);

  // get primary vertices for L1Offset correction level if needed
  edm::Handle<std::vector<reco::Vertex> > primaryVertices;
  if(!primaryVertices_.label().empty()) event.getByToken(primaryVerticesToken_, primaryVertices);

  // get parameter rho for L1FastJet correction level if needed
  edm::Handle<double> rho;
  if(!rho_.label().empty()) event.getByToken(rhoToken_, rho);

  auto const& rec = setup.get<JetCorrectionsRecord>();
  if (cacheId_ != rec.cacheIdentifier()) {
    // retreive parameters from the DB
    edm::ESHandle<JetCorrectorParametersCollection> parameters;
    setup.get<JetCorrectionsRecord>().get(payload(), parameters);
    // initialize jet correctors
    for(FlavorCorrLevelMap::const_iterator flavor=levels_.begin(); flavor!=levels_.end(); ++flavor){
      correctors_[flavor->first].reset( new FactorizedJetCorrector(params(*parameters, flavor->second)) );
    }
    // initialize extra jet corrector for jpt if needed
    if(!extraJPTOffset_.empty()){
      extraJPTOffsetCorrector_.reset( new FactorizedJetCorrector(params(*parameters, extraJPTOffset_)) );
    }
    cacheId_ = rec.cacheIdentifier();
  }

  // fill the jetCorrFactors
  std::vector<JetCorrFactors> jcfs;
  for(edm::View<reco::Jet>::const_iterator jet = jets->begin(); jet!=jets->end(); ++jet){
    // the JetCorrFactors::CorrectionFactor is a std::pair<std::string, std::vector<float> >
    // the string corresponds to the label of the correction level, the vector contains four
    // floats if flavor dependent and one float else. Per construction jet energy corrections
    // will be flavor independent up to the first flavor dependent correction and flavor de-
    // pendent afterwards. The first correction level is predefined with label 'Uncorrected'.
    // Per definition it is flavor independent. The correction factor is 1.
    std::vector<JetCorrFactors::CorrectionFactor> jec;
    jec.push_back(std::make_pair<std::string, std::vector<float> >(std::string("Uncorrected"), std::vector<float>(1, 1)));

    // pick the first element in the map (which could be the only one) and loop all jec
    // levels listed for that element. If this is not the only element all jec levels, which
    // are flavor independent will give the same correction factors until the first flavor
    // dependent correction level is reached. So the first element is still a good choice.
    FlavorCorrLevelMap::const_iterator corrLevel=levels_.begin();
    if(corrLevel==levels_.end()){
      throw cms::Exception("No JECFactors") << "You request to create a jetCorrFactors object with no JEC Levels indicated. \n"
					    << "This makes no sense, either you should correct this or drop the module from \n"
					    << "the sequence.";
    }
    for(unsigned int idx=0; idx<corrLevel->second.size(); ++idx){
      bool flavorDependent=false;
      std::vector<float> factors;
      if(flavorDependent ||
	 corrLevel->second[idx].find("L5Flavor")!=std::string::npos ||
	 corrLevel->second[idx].find("L7Parton")!=std::string::npos){
	flavorDependent=true;
	// after the first encounter all subsequent correction levels are flavor dependent
	for(FlavorCorrLevelMap::const_iterator flavor=corrLevel; flavor!=levels_.end(); ++flavor){
	  if(!primaryVertices_.label().empty()){
	    // if primaryVerticesToken_ has a value the number of primary vertices needs to be
	    // specified
	    correctors_.find(flavor->first)->second->setNPV(numberOf(primaryVertices));
	  }
	  if(!rho_.label().empty()){
	    // if rhoToken_ has a value the energy density parameter rho and the jet area need
	    //  to be specified
	    correctors_.find(flavor->first)->second->setRho(*rho);
	    correctors_.find(flavor->first)->second->setJetA(jet->jetArea());
	  }
	  factors.push_back(evaluate(jet, flavor->first, idx));
	}
      }
      else{
	if(!primaryVertices_.label().empty()){
	  // if primaryVerticesToken_ has a value the number of primary vertices needs to be
	  // specified
	  correctors_.find(corrLevel->first)->second->setNPV(numberOf(primaryVertices));
	}
	if(!rho_.label().empty()){
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
      jec.push_back(std::make_pair((corrLevel->second[idx]).substr(0, (corrLevel->second[idx]).find("_")), factors));
    }
    // create the actual object with the scale factors we want the valuemap to refer to
    // label_ corresponds to the label of the module instance
    JetCorrFactors corrFactors(label_, jec);
    jcfs.push_back(corrFactors);
  }
  // build the value map
  std::auto_ptr<JetCorrFactorsMap> jetCorrsMap(new JetCorrFactorsMap());
  JetCorrFactorsMap::Filler filler(*jetCorrsMap);
  // jets and jetCorrs have their indices aligned by construction
  filler.insert(jets, jcfs.begin(), jcfs.end());
  filler.fill(); // do the actual filling
  // put our produced stuff in the event
  event.put(jetCorrsMap);
}

void
JetCorrFactorsProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
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

  std::vector<std::string> levels;
  levels.push_back(std::string("L1Offset"  ));
  levels.push_back(std::string("L2Relative"));
  levels.push_back(std::string("L3Absolute"));
  levels.push_back(std::string("L2L3Residual"));
  levels.push_back(std::string("L5Flavor"  ));
  levels.push_back(std::string("L7Parton"  ));
  iDesc.add<std::vector<std::string> >("levels", levels);
  descriptions.add("JetCorrFactorsProducer", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(JetCorrFactorsProducer);
