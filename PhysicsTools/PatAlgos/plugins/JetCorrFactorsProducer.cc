//
// $Id: JetCorrFactorsProducer.cc,v 1.3 2008/11/04 14:12:58 auterman Exp $
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

  // flavor
  if(levels_.find("L5")!=std::string::npos && 
     levels_.find("L7")!=std::string::npos){
    // available options: see below
    jetCorrector_    = new CombinedJetCorrector(levels_, tags_, "Flavor:g   & Parton:jJ");
    jetCorrectorGlu_ = new CombinedJetCorrector(levels_, tags_, "Flavor:g   & Parton:gJ");
    jetCorrectorUds_ = new CombinedJetCorrector(levels_, tags_, "Flavor:uds & Parton:qJ");
    jetCorrectorC_   = new CombinedJetCorrector(levels_, tags_, "Flavor:c   & Parton:cJ");
    jetCorrectorB_   = new CombinedJetCorrector(levels_, tags_, "Flavor:b   & Parton:bJ");
  }
  // flavor
  else if(levels_.find("L5")!=std::string::npos){
    // available options are: 
    // Flavor:g    gluon
    // Flavor:uds  uds
    // Flavor:c    charm
    // Flavor:b    beauty
    jetCorrector_    = new CombinedJetCorrector(levels_, tags_, "Flavor:g");
    jetCorrectorGlu_ = new CombinedJetCorrector(levels_, tags_, "Flavor:g");
    jetCorrectorUds_ = new CombinedJetCorrector(levels_, tags_, "Flavor:uds");
    jetCorrectorC_   = new CombinedJetCorrector(levels_, tags_, "Flavor:c"  );
    jetCorrectorB_   = new CombinedJetCorrector(levels_, tags_, "Flavor:b"  );
  }
  // parton
  else if(levels_.find("L7")!=std::string::npos){
    // available options are: 
    // Parton:gJ/gT  gluon  from dijets/top
    // Parton:qJ/qT  uds    from dijets/top
    // Parton:cJ/cT  charm  from dijets/top
    // Parton:bJ/bT  beauty from dijets/top
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
    // to have the emf accessibla to the corrector the jet needs to 
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
    float l1=-1, l2=-1, l3=-1, l4=-1, l6=-1;
    JetCorrFactors::FlavourCorrections l5, l7;
    
    // get jet correction factors
    // from CombinedJetCorrectors 
    int levelCounter = -1;
    if(levels_.find("L1")!=std::string::npos){
      // L1Offset
      l1 = evaluate(jet, jetCorrector_, ++levelCounter);
    }
    if(levels_.find("L2")!=std::string::npos){
      // L2Relative
      l2 = evaluate(jet, jetCorrector_, ++levelCounter);
    }
    if(levels_.find("L3")!=std::string::npos){
      // L3Absolute
      l3 = evaluate(jet, jetCorrector_, ++levelCounter);
    }
    if(levels_.find("L4")!=std::string::npos){
      // L4EMF
      l4 = evaluate(jet, jetCorrector_, ++levelCounter);
    }
    if(levels_.find("L5")!=std::string::npos){
      // L5Flavor
      ++levelCounter;
      l5.uds = evaluate(jet, jetCorrectorUds_, levelCounter);
      l5.g   = evaluate(jet, jetCorrectorGlu_, levelCounter);
      l5.c   = evaluate(jet, jetCorrectorC_  , levelCounter);
      l5.b   = evaluate(jet, jetCorrectorB_  , levelCounter);
    }
    if(levels_.find("L6")!=std::string::npos){
      // L6UE
      l6 = evaluate(jet, jetCorrector_, ++levelCounter);
    }
    if(levels_.find("L7")!=std::string::npos){
      // L7Parton
      ++levelCounter;
      l7.uds = evaluate(jet, jetCorrectorUds_, levelCounter);
      l7.g   = evaluate(jet, jetCorrectorGlu_, levelCounter);
      l7.c   = evaluate(jet, jetCorrectorC_,   levelCounter);
      l7.b   = evaluate(jet, jetCorrectorB_,   levelCounter);
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
