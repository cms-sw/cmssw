//
// $Id: JetCorrFactorsProducer.cc,v 1.11 2009/10/12 19:43:32 rwolf Exp $
//

#include "PhysicsTools/PatAlgos/plugins/JetCorrFactorsProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterDescription.h"


#include <vector>
#include <memory>


using namespace pat;


JetCorrFactorsProducer::JetCorrFactorsProducer(const edm::ParameterSet& iConfig) :
  useEMF_ (iConfig.getParameter<bool>( "useEMF" )), 
  jetsSrc_(iConfig.getParameter<edm::InputTag>( "jetSource" )),
  jetCorrSet_(iConfig.getParameter<std::string>( "corrSample" )),
  moduleLabel_(iConfig.getParameter<std::string>( "@module_label" )),
  jetCorrector_(0), jetCorrectorGlu_(0), jetCorrectorUds_(0), 
  jetCorrectorC_(0), jetCorrectorB_(0) 
{
  // configure constructor strings for CombinedJetCorrector
  // if there is no corrector defined the string should be 
  // 'none'
  edm::ParameterSet corrLevels = iConfig.getParameter<edm::ParameterSet>( "corrLevels");
  configure(std::string("L1"), corrLevels.getParameter<std::string>( "L1Offset"   ) );
  configure(std::string("L2"), corrLevels.getParameter<std::string>( "L2Relative" ) );
  configure(std::string("L3"), corrLevels.getParameter<std::string>( "L3Absolute" ) );
  configure(std::string("L4"), corrLevels.getParameter<std::string>( "L4EMF"      ) );
  configure(std::string("L5"), corrLevels.getParameter<std::string>( "L5Flavor"   ) );
  configure(std::string("L6"), corrLevels.getParameter<std::string>( "L6UE"       ) );
  configure(std::string("L7"), corrLevels.getParameter<std::string>( "L7Parton"   ) );

  CorrType corr=kPlain;
  // determine correction type for the flavor dependend corrections
  if(levels_.find("L5")!=std::string::npos && 
     levels_.find("L7")!=std::string::npos){
    // flavor & partons conbined
    corr=kCombined;
  } 
  else if(levels_.find("L5")!=std::string::npos){
    // flavor only
    corr=kFlavor;
  }
  else if(levels_.find("L7")!=std::string::npos){
    // parton only
    corr=kParton;
  }

  SampleType type=kNone;
  // determine sample type for the flavor dependend corrections
  if     ( iConfig.getParameter<std::string>( "sampleType" ).compare("dijet")==0){
    type = kDijet;
  }
  else if( iConfig.getParameter<std::string>( "sampleType" ).compare("ttbar")==0){
    type = kTtbar;
  }
  else{
    throw cms::Exception("InvalidRequest") 
      << "you ask for a sample type for jet energy corrections which does not exist \n";  
  }

  if( corr==kPlain ){
    // plain jet corrector w/o flavor dependend corrections
    jetCorrector_ = new CombinedJetCorrector(levels_, tags_);
  }
  else{
    // special treatment in case of flavor dep. corrections
    if( type==kTtbar ){
      // ATTENTION: there is no gluon corrections from ttbar
      //  * for kParton   the default will be set to kMixed 
      //  * for kFlavor   the default will be set to kQuark
      //  * for kCombined the default will be set to kQuark
      //  * the gluon corrector remains uninitialized (after
      //    all the gluon correction do not exist...)
      if( corr==kParton ){
	jetCorrector_    = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kMixed ));
      }
      else{
	jetCorrector_    = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kQuark ));
      }
    }
    if( type==kDijet ){
      // ATTENTION: 
      //  * for kParton   the default will be set to kMixed 
      //  * for kFlavor   the default will be set to kGluon
      //  * for kCombined the default will be set to lGluon
      //  * the gluon corrector is initialized here
      if( corr==kParton ){
	jetCorrector_    = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kMixed ));
	jetCorrectorGlu_ = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kGluon ));
      }
      else{
	jetCorrector_    = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kGluon ));
	jetCorrectorGlu_ = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kGluon ));
      }
    }
    jetCorrectorUds_     = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kQuark ));
    jetCorrectorC_       = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kCharm ));
    jetCorrectorB_       = new CombinedJetCorrector(levels_, tags_, flavorTag(corr, type, kBeauty));
  }
  
  jtuncrtl1_ = new JetCorrectionUncertainty( moduleLabel_+"L1.txt" );
  jtuncrtl2_ = new JetCorrectionUncertainty( moduleLabel_+"L2.txt" );
  jtuncrtl3_ = new JetCorrectionUncertainty( moduleLabel_+"L3.txt" );
  jtuncrtl4_ = new JetCorrectionUncertainty( moduleLabel_+"L4.txt" );
  jtuncrtl5_ = new JetCorrectionUncertainty( moduleLabel_+"L5.txt" );
  jtuncrtl6_ = new JetCorrectionUncertainty( moduleLabel_+"L6.txt" );
  jtuncrtl7_ = new JetCorrectionUncertainty( moduleLabel_+"L7.txt" );
  
  // produces valuemap of jet correction factors
  produces<JetCorrFactorsMap>();
}

JetCorrFactorsProducer::~JetCorrFactorsProducer() 
{
  delete jtuncrtl1_;
  delete jtuncrtl2_;
  delete jtuncrtl3_;
  delete jtuncrtl4_;
  delete jtuncrtl5_;
  delete jtuncrtl6_;
  delete jtuncrtl7_;
}

void 
JetCorrFactorsProducer::configure(std::string level, std::string tag)
{
  if( !tag.compare("none")==0 ){
    // add the correction set to the jet corrector tag, if necessary
    // at the moment that's true for level=='L2' and level=='L3'
    if(level=="L2" || level=="L3"){
      tag = jetCorrSet_+"_"+tag;
    }
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
    correction = (corrector->getSubCorrections(jet->pt(), jet->eta(), jet->energy()))[idx];
  }
  else{
    // to have the emf accessible to the corrector the jet needs to 
    // be a CaloJet
    correction = (corrector->getSubCorrections(jet->pt(), jet->eta(), jet->energy()))[idx];
  }
  return correction;
}

std::string
JetCorrFactorsProducer::flavorTag(CorrType correction, SampleType sample, FlavorType flavor)
{
  // ATTENTION: available options are
  //  * Flavor:gJ    Parton:gJ     gluon   from dijets
  //  * Flavor:qJ/qT Parton:qJ/qT  quark   from dijets/top
  //  * Flavor:cJ/cT Parton:cJ/cT  charm   from dijets/top
  //  * Flavor:bJ/bT Parton:bJ/bT  beauty  from dijets/top
  //  *              Parton:jJ/tT  mixture from dijets/top
  //
  // NOTE:
  //  * the mixed mode (mc input mixture from dijets/ttbar) 
  //    only exists for parton level corrections
  //  * there are no gluon corrections available from the 
  //    top sample neighter on the level of flavor nor on 
  //    the level of parton level corrections

  std::string flavtag;
  switch( flavor ){
  case kGluon : flavtag += "g"; break;
  case kQuark : flavtag += "q"; break;
  case kCharm : flavtag += "c"; break;
  case kBeauty: flavtag += "b"; break;
  case kMixed : 
    if( sample==kDijet ){
      flavtag += "Parton:jJ";
    }
    if( sample==kTtbar ){
      flavtag += "Parton:tT";
    } 
    // kMixed only makes sense for kParton;
    // therefor other options are ignored
    return flavtag;    
  }
  switch( sample ){
  case kDijet: flavtag += "J"; break;
  case kTtbar: flavtag += "T"; break;
  case kNone :                 break;
  }
  std::string tag;
  switch( correction ){
  case kPlain :
    break;
  case kFlavor: 
    tag  = "Flavor:"; 
    tag += flavtag;
    break;
  case kParton: 
    tag += "Parton:"; 
    tag += flavtag;
    break;
  case kCombined:
    tag  = "Flavor:"; 
    tag += flavtag; 
    tag += " & ";
    tag += "Parton:"; 
    tag += flavtag;
    break;
  }
  return tag;
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
      // check whether the corrector is available or not; if not fall back by to
      // the last available correction level 
      l5.uds = jetCorrectorUds_ ? evaluate(jet, jetCorrectorUds_, levelIdx) : l4;
      l5.g   = jetCorrectorGlu_ ? evaluate(jet, jetCorrectorGlu_, levelIdx) : l4;
      l5.c   = jetCorrectorC_   ? evaluate(jet, jetCorrectorC_  , levelIdx) : l4;
      l5.b   = jetCorrectorB_   ? evaluate(jet, jetCorrectorB_  , levelIdx) : l4;
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
      // check whether the corrector is available or not; if not fall back by to
      // the last available correction level 
      l6.uds = jetCorrectorUds_ ? evaluate(jet, jetCorrectorUds_, levelIdx) : l5.uds;
      l6.g   = jetCorrectorGlu_ ? evaluate(jet, jetCorrectorGlu_, levelIdx) : l5.g;
      l6.c   = jetCorrectorC_   ? evaluate(jet, jetCorrectorC_  , levelIdx) : l5.c;
      l6.b   = jetCorrectorC_   ? evaluate(jet, jetCorrectorB_  , levelIdx) : l5.b;
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
      // check whether the corrector is available or not; if not fall back by to
      // the last available correction level 
      l7.uds = jetCorrectorUds_ ? evaluate(jet, jetCorrectorUds_, levelIdx) : l6.uds;
      l7.g   = jetCorrectorGlu_ ? evaluate(jet, jetCorrectorGlu_, levelIdx) : l6.g;
      l7.c   = jetCorrectorC_   ? evaluate(jet, jetCorrectorC_,   levelIdx) : l6.c;
      l7.b   = jetCorrectorB_   ? evaluate(jet, jetCorrectorB_,   levelIdx) : l6.b;
    }
    else{
      l7.uds = l6.uds;
      l7.g   = l6.g;
      l7.c   = l6.c;
      l7.b   = l6.b;
    }
    // jet correction uncertainties
    std::vector<float> uncert(14);
    uncert[ 0]=jtuncrtl1_->uncertaintyPtEta( jet->pt(), jet->eta(), "UP");   //L1 + 1 sigma
    uncert[ 1]=jtuncrtl1_->uncertaintyPtEta( jet->pt(), jet->eta(), "DOWN"); //L1 - 1 sigma
    uncert[ 2]=jtuncrtl2_->uncertaintyPtEta( jet->pt(), jet->eta(), "UP");   //L2 + 1 sigma
    uncert[ 3]=jtuncrtl2_->uncertaintyPtEta( jet->pt(), jet->eta(), "DOWN"); //L2 - 1 sigma
    uncert[ 4]=jtuncrtl3_->uncertaintyPtEta( jet->pt(), jet->eta(), "UP");   //L3 + 1 sigma
    uncert[ 5]=jtuncrtl3_->uncertaintyPtEta( jet->pt(), jet->eta(), "DOWN"); //L3 - 1 sigma
    uncert[ 6]=jtuncrtl4_->uncertaintyPtEta( jet->pt(), jet->eta(), "UP");   //L4 + 1 sigma
    uncert[ 7]=jtuncrtl4_->uncertaintyPtEta( jet->pt(), jet->eta(), "DOWN"); //L4 - 1 sigma
    uncert[ 8]=jtuncrtl5_->uncertaintyPtEta( jet->pt(), jet->eta(), "UP");   //L5 + 1 sigma
    uncert[ 9]=jtuncrtl5_->uncertaintyPtEta( jet->pt(), jet->eta(), "DOWN"); //L5 - 1 sigma
    uncert[10]=jtuncrtl6_->uncertaintyPtEta( jet->pt(), jet->eta(), "UP");   //L6 + 1 sigma
    uncert[11]=jtuncrtl6_->uncertaintyPtEta( jet->pt(), jet->eta(), "DOWN"); //L6 - 1 sigma
    uncert[12]=jtuncrtl7_->uncertaintyPtEta( jet->pt(), jet->eta(), "UP");   //L7 + 1 sigma
    uncert[13]=jtuncrtl7_->uncertaintyPtEta( jet->pt(), jet->eta(), "DOWN"); //L7 - 1 sigma
    
    // create the actual object with scalefactors we want the valuemap to refer to
    JetCorrFactors aJetCorr( moduleLabel_, l1, l2, l3, l4, l5, l6, l7, uncert );
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

// ParameterSet description for module
void
JetCorrFactorsProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions)
{
  edm::ParameterSetDescription iDesc;
  iDesc.add<bool>("useEMF", false);
  iDesc.add<edm::InputTag>("jetSource", edm::InputTag("ak5CaloJets")); 
  iDesc.add<std::string>("corrSample", "Summer09");
  iDesc.add<std::string>("sampleType", "dijet");

  edm::ParameterSetDescription corrLevels;
  corrLevels.add<std::string>("L1Offset", "none");
  corrLevels.add<std::string>("L2Relative", "L2Relative_IC5Calo");
  corrLevels.add<std::string>("L3Absolute", "L2Relative_IC5Calo");
  corrLevels.add<std::string>("L4EMF", "none");
  corrLevels.add<std::string>("L5Flavor", "L5Flavor_IC5");
  corrLevels.add<std::string>("L6UE", "none");
  corrLevels.add<std::string>("L7Parton", "L7Parton_IC5");
  iDesc.add("corrLevels", corrLevels);

  descriptions.add("JetCorrFactorsProducer", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(JetCorrFactorsProducer);
