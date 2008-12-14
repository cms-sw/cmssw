//
// $Id: JetCorrFactorsProducer.cc,v 1.2 2008/03/10 14:38:57 lowette Exp $
//

#include "PhysicsTools/PatAlgos/plugins/JetCorrFactorsProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/View.h"

#include "JetMETCorrections/Objects/interface/JetCorrector.h"
#include "DataFormats/JetReco/interface/Jet.h"

#include <vector>
#include <memory>


using namespace pat;


JetCorrFactorsProducer::JetCorrFactorsProducer(const edm::ParameterSet& iConfig) {
  // initialize the configurables
  jetsSrc_ = iConfig.getParameter<edm::InputTag>( "jetSource" );

  L1JetCorrService_    = iConfig.getParameter<std::string>( "L1JetCorrector" );
  L2JetCorrService_    = iConfig.getParameter<std::string>( "L2JetCorrector" );
  L3JetCorrService_    = iConfig.getParameter<std::string>( "L3JetCorrector" );
  L4JetCorrService_    = iConfig.getParameter<std::string>( "L4JetCorrector" );
  L6JetCorrService_    = iConfig.getParameter<std::string>( "L6JetCorrector" );
  L5udsJetCorrService_ = iConfig.getParameter<std::string>( "L5udsJetCorrector" );
  L5gluJetCorrService_ = iConfig.getParameter<std::string>( "L5gluonJetCorrector" );
  L5cJetCorrService_   = iConfig.getParameter<std::string>( "L5cJetCorrector" );
  L5bJetCorrService_   = iConfig.getParameter<std::string>( "L5bJetCorrector" );
  L7udsJetCorrService_ = iConfig.getParameter<std::string>( "L7udsJetCorrector" );
  L7gluJetCorrService_ = iConfig.getParameter<std::string>( "L7gluonJetCorrector" );
  L7cJetCorrService_   = iConfig.getParameter<std::string>( "L7cJetCorrector" );
  L7bJetCorrService_   = iConfig.getParameter<std::string>( "L7bJetCorrector" );
  
  bl1_    = (L1JetCorrService_.compare("none")==0)    ? false : true;
  bl2_    = (L2JetCorrService_.compare("none")==0)    ? false : true;
  bl3_    = (L3JetCorrService_.compare("none")==0)    ? false : true;
  bl4_    = (L4JetCorrService_.compare("none")==0)    ? false : true;
  bl6_    = (L6JetCorrService_.compare("none")==0)    ? false : true;
  bl5uds_ = (L5udsJetCorrService_.compare("none")==0) ? false : true;
  bl5g_   = (L5gluJetCorrService_.compare("none")==0) ? false : true;
  bl5c_   = (L5cJetCorrService_.compare("none")==0)   ? false : true;
  bl5b_   = (L5bJetCorrService_.compare("none")==0)   ? false : true;
  bl7uds_ = (L7udsJetCorrService_.compare("none")==0) ? false : true;
  bl7g_   = (L7gluJetCorrService_.compare("none")==0) ? false : true;
  bl7c_   = (L7cJetCorrService_.compare("none")==0)   ? false : true;
  bl7b_   = (L7bJetCorrService_.compare("none")==0)   ? false : true;

  // produces valuemap of jet correction factors
  produces<JetCorrFactorsMap>();

}


JetCorrFactorsProducer::~JetCorrFactorsProducer() {
}


void JetCorrFactorsProducer::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
  // define the jet correctors - FIXME: make configurable
  const JetCorrector *L1JetCorr=0, *L2JetCorr=0, *L3JetCorr=0, *L4JetCorr=0, *L6JetCorr=0,
                     *L5udsJetCorr=0,*L5gluJetCorr=0,*L5cJetCorr=0,*L5bJetCorr=0,
		     *L7udsJetCorr=0,*L7gluJetCorr=0,*L7cJetCorr=0,*L7bJetCorr=0;

  if (bl1_)    L1JetCorr     = JetCorrector::getJetCorrector(L1JetCorrService_, iSetup);
  if (bl2_)    L2JetCorr     = JetCorrector::getJetCorrector(L2JetCorrService_, iSetup);
  if (bl3_)    L3JetCorr     = JetCorrector::getJetCorrector(L3JetCorrService_, iSetup);
  if (bl4_)    L4JetCorr     = JetCorrector::getJetCorrector(L4JetCorrService_, iSetup);
  if (bl6_)    L6JetCorr     = JetCorrector::getJetCorrector(L6JetCorrService_, iSetup);
  if (bl5uds_) L5udsJetCorr  = JetCorrector::getJetCorrector(L5udsJetCorrService_, iSetup);
  if (bl5g_)   L5gluJetCorr  = JetCorrector::getJetCorrector(L5gluJetCorrService_, iSetup);
  if (bl5c_)   L5cJetCorr    = JetCorrector::getJetCorrector(L5cJetCorrService_, iSetup);
  if (bl5b_)   L5bJetCorr    = JetCorrector::getJetCorrector(L5bJetCorrService_, iSetup);
  if (bl7uds_) L7udsJetCorr  = JetCorrector::getJetCorrector(L7udsJetCorrService_, iSetup);
  if (bl7g_)   L7gluJetCorr  = JetCorrector::getJetCorrector(L7gluJetCorrService_, iSetup);
  if (bl7c_)   L7cJetCorr    = JetCorrector::getJetCorrector(L7cJetCorrService_, iSetup);
  if (bl7b_)   L7bJetCorr    = JetCorrector::getJetCorrector(L7bJetCorrService_, iSetup);

  edm::Handle<edm::View<reco::Jet> > jets;
  iEvent.getByLabel(jetsSrc_, jets);

  // loop over jets and retrieve the correction factors
  std::vector<JetCorrFactors> jetCorrs;
  for (edm::View<reco::Jet>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++) {
    // retrieve the energy correction factors
    float l1=-1, l2=-1, l3=-1, l4=-1, l6=-1;
    JetCorrFactors::FlavourCorrections l5, l7; 
    if (bl1_)	 l1 =	  L1JetCorr->correction( *itJet );
    if (bl2_)	 l2 =	  L2JetCorr->correction( *itJet );  
    if (bl3_)	 l3 =	  L3JetCorr->correction( *itJet );
    if (bl4_)	 l4 =	  L4JetCorr->correction( *itJet );
    if (bl6_)	 l6 =	  L6JetCorr->correction( *itJet );
    if (bl5uds_) l5.uds = L5udsJetCorr->correction( *itJet );
    if (bl5g_)   l5.g =   L5gluJetCorr->correction( *itJet );
    if (bl5c_)   l5.c =   L5cJetCorr->correction( *itJet );
    if (bl5b_)   l5.b =   L5bJetCorr->correction( *itJet );
    if (bl7uds_) l7.uds = L7udsJetCorr->correction( *itJet );
    if (bl7g_)   l7.g =   L7gluJetCorr->correction( *itJet );
    if (bl7c_)   l7.c =   L7cJetCorr->correction( *itJet );
    if (bl7b_)   l7.b =   L7bJetCorr->correction( *itJet );

    // create the actual object with scalefactos we want the valuemap to refer to
    JetCorrFactors aJetCorr( l1, l2, l3, l4, l5, l6, l7 );

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
