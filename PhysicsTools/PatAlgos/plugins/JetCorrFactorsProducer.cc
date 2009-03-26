//
// $Id: JetCorrFactorsProducer.cc,v 1.4 2009/02/19 16:18:08 rwolf Exp $
//

#include "PhysicsTools/PatAlgos/plugins/JetCorrFactorsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <memory>


using namespace pat;


JetCorrFactorsProducer::JetCorrFactorsProducer(const edm::ParameterSet& iConfig) :
  useEMF_ (iConfig.getParameter<bool>( "useEMF" )), 
  jetsSrc_(iConfig.getParameter<edm::InputTag>( "jetSource" )),
  moduleLabel_(iConfig.getParameter<std::string>( "@module_label" ))
{
  // configure constructor strings for CombinedJetCorrector
  // if there is no corrector defined the string should be 
  // 'none'
  configure(std::string("L1"), iConfig.getParameter<std::string>( "L1Offset"   ) );
  configure(std::string("L2"), iConfig.getParameter<std::string>( "L2Relative" ) );
  configure(std::string("L3"), iConfig.getParameter<std::string>( "L3Absolute" ) );
  configure(std::string("L4"), iConfig.getParameter<std::string>( "L4EMF"      ) );
  configure(std::string("L5"), iConfig.getParameter<std::string>( "L5Flavor"   ) );
  configure(std::string("L6"), iConfig.getParameter<std::string>( "L6UE"       ) );
  configure(std::string("L7"), iConfig.getParameter<std::string>( "L7Parton"   ) );
  
  // define CombinedJetCorrectors

  // flavor & parton
  if(levels_.find("L5")!=std::string::npos && levels_.find("L7")!=std::string::npos){
    // available options: see below
    jetCorrector_    = new CombinedJetCorrector(levels_, tags_, "Flavor:gJ & Parton:gJ");
    jetCorrectorGlu_ = new CombinedJetCorrector(levels_, tags_, "Flavor:gJ & Parton:gJ");
    jetCorrectorUds_ = new CombinedJetCorrector(levels_, tags_, "Flavor:qJ & Parton:qJ");
    jetCorrectorC_   = new CombinedJetCorrector(levels_, tags_, "Flavor:cJ & Parton:cJ");
    jetCorrectorB_   = new CombinedJetCorrector(levels_, tags_, "Flavor:bJ & Parton:bJ");
  }
  // flavor
  else if(levels_.find("L5")!=std::string::npos){
    // available options are: 
    // Flavor:gJ   gluon
    // Flavor:qJ   uds
    // Flavor:cJ   charm
    // Flavor:bJ   beauty
    jetCorrector_    = new CombinedJetCorrector(levels_, tags_, "Flavor:gJ");
    jetCorrectorGlu_ = new CombinedJetCorrector(levels_, tags_, "Flavor:gJ");
    jetCorrectorUds_ = new CombinedJetCorrector(levels_, tags_, "Flavor:qJ");
    jetCorrectorC_   = new CombinedJetCorrector(levels_, tags_, "Flavor:cJ"  );
    jetCorrectorB_   = new CombinedJetCorrector(levels_, tags_, "Flavor:bJ"  );
  }
  // parton
  else if(levels_.find("L7")!=std::string::npos){
    // available options are: 
    // Parton:gJ/--  gluon  from dijets
    // Parton:qJ/qT  uds    from dijets/top
    // Parton:cJ/cT  charm  from dijets/top
    // Parton:bJ/bT  beauty from dijets/top
    // Parton:jJ/tT  mc input mixture from dijets/top
    // be aware that from the top sample there is no
    // gluon correction available down to parton level
    jetCorrector_    = new CombinedJetCorrector(levels_, tags_, "Parton:jJ");
    jetCorrectorGlu_ = new CombinedJetCorrector(levels_, tags_, "Parton:gJ");
    jetCorrectorUds_ = new CombinedJetCorrector(levels_, tags_, "Parton:qJ");
    jetCorrectorC_   = new CombinedJetCorrector(levels_, tags_, "Parton:cJ");
    jetCorrectorB_   = new CombinedJetCorrector(levels_, tags_, "Parton:bJ");
  }
  // common
  else{
    jetCorrector_ = new CombinedJetCorrector(levels_, tags_);
  }
  // produces valuemap of jet correction factors
  produces<JetCorrFactorsMap>();
}

JetCorrFactorsProducer::~JetCorrFactorsProducer() 
{
}

void 
JetCorrFactorsProducer::configure(std::string level, std::string tag)
{
  if( !tag.compare("none")==0 ){
    // take care to add the deliminator when the string is non-empty
    if( !tags_  .empty() && !(tags_  .rfind(":")==tags_  .size()) ) tags_   += ":";
    if( !levels_.empty() && !(levels_.rfind(":")==levels_.size()) ) levels_ += ":";
    // add tag and level
    tags_   += tag;
    levels_ += level;
  }
}

double 
JetCorrFactorsProducer::evaluate(edm::View<reco::Jet>::const_iterator& jet, CombinedJetCorrector* corrector, int& idx)
{
  // get the jet energy correction factors depending on whether emf should be used or not;
  double correction;
  if( !useEMF_ ){
    correction = (corrector->getSubCorrections(jet->pt(), jet->eta()))[idx];
  }
  else{
    // to have the emf accessible to the corrector the jet needs to 
    // be a CaloJet
    correction = (corrector->getSubCorrections(jet->pt(), jet->eta()))[idx];
  }
  return correction;
}

void 
JetCorrFactorsProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) 
{
  // get jet collection from the event
  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByLabel(jetsSrc_, jets);

  std::vector<JetCorrFactors> jetCorrs;
  for (edm::View<reco::Jet>::const_iterator jet = jets->begin(); jet != jets->end(); jet++) {
    // loop over jets and retrieve the correction factors
    float l1=-1, l2=-1, l3=-1, l4=-1;
    JetCorrFactors::FlavourCorrections l5, l6, l7;
    // get jet correction factors
    // from CombinedJetCorrectors 
    int levelIdx = -1;
    // --------------------------------------------
    // floavor independend jet correctors
    //

    // L1Offset
    levels_.find("L1")!=std::string::npos ? l1 = evaluate(jet, jetCorrector_, ++levelIdx) : l1= 1; 
    // L2Relative
    levels_.find("L2")!=std::string::npos ? l2 = evaluate(jet, jetCorrector_, ++levelIdx) : l2=l1; 
    // L3Absolute
    levels_.find("L3")!=std::string::npos ? l3 = evaluate(jet, jetCorrector_, ++levelIdx) : l3=l2; 
    // L4EMF
    levels_.find("L4")!=std::string::npos ? l4 = evaluate(jet, jetCorrector_, ++levelIdx) : l4=l3; 

    // --------------------------------------------
    // flavor dependend   jet correctors
    //

    // L5Flavor
    if(levels_.find("L5")!=std::string::npos){
      ++levelIdx;
      l5.uds = evaluate(jet, jetCorrectorUds_, levelIdx);
      l5.g   = evaluate(jet, jetCorrectorGlu_, levelIdx);
      l5.c   = evaluate(jet, jetCorrectorC_  , levelIdx);
      l5.b   = evaluate(jet, jetCorrectorB_  , levelIdx);
    }
    else{
      l5.uds = l4;
      l5.g   = l4;
      l5.c   = l4;
      l5.b   = l4;
    }
    // L6UE
    if(levels_.find("L6")!=std::string::npos){
      ++levelIdx;
      l6.uds = evaluate(jet, jetCorrectorUds_, levelIdx);
      l6.g   = evaluate(jet, jetCorrectorGlu_, levelIdx);
      l6.c   = evaluate(jet, jetCorrectorC_  , levelIdx);
      l6.b   = evaluate(jet, jetCorrectorB_  , levelIdx);
    }
    else{
      l6.uds = l5.uds;
      l6.g   = l5.g;
      l6.c   = l5.c;
      l6.b   = l5.b;
    }
    // L7Parton
    if(levels_.find("L7")!=std::string::npos){
      ++levelIdx;
      l7.uds = evaluate(jet, jetCorrectorUds_, levelIdx);
      l7.g   = evaluate(jet, jetCorrectorGlu_, levelIdx);
      l7.c   = evaluate(jet, jetCorrectorC_,   levelIdx);
      l7.b   = evaluate(jet, jetCorrectorB_,   levelIdx);
    }
    else{
      l7.uds = l6.uds;
      l7.g   = l6.g;
      l7.c   = l6.c;
      l7.b   = l6.b;
    }
    // create the actual object with scalefactos we want the valuemap to refer to
    JetCorrFactors aJetCorr( moduleLabel_, l1, l2, l3, l4, l5, l6, l7 );
    jetCorrs.push_back(aJetCorr);
  }

  // build the value map
  std::auto_ptr<JetCorrFactorsMap> jetCorrsMap(new JetCorrFactorsMap());
  JetCorrFactorsMap::Filler filler(*jetCorrsMap);
  // jets and jetCorrs have their indices aligned by construction
  filler.insert(jets, jetCorrs.begin(), jetCorrs.end());
  filler.fill(); // do the actual filling

  // put our produced stuff in the event
  iEvent.put(jetCorrsMap);
}


#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(JetCorrFactorsProducer);
