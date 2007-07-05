//
// $Id: TopJet.cc,v 1.11 2007/06/30 14:42:54 gpetrucc Exp $
//


#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"


/// default constructor
TopJet::TopJet() :
  TopObject<TopJetType>(TopJetType(reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), reco::CaloJet::Specific(), reco::Jet::Constituents())),
  jetFlavour_(0), lrPhysicsJetLRval_(-999.), lrPhysicsJetProb_(-1),
  jetCharge_(0.0), associatedTracks_() {
}


/// constructor from a TopJetType
TopJet::TopJet(TopJetType aJet) :
  TopObject<TopJetType>(aJet),
  jetFlavour_(0), lrPhysicsJetLRval_(-999.), lrPhysicsJetProb_(-1) {
}


/// destructor
TopJet::~TopJet() {
}


/// return the matched generated parton
reco::Particle TopJet::getGenParton() const {
  return (genParton_.size() > 0 ?
    genParton_.front() :
    reco::Particle(0, reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0))
  );
}


/// return the matched generated jet
reco::GenJet TopJet::getGenJet() const {
  return (genJet_.size() > 0 ?
    genJet_.front() :
    reco::GenJet(reco::Particle::LorentzVector(0, 0, 0, 0), reco::Particle::Point(0,0,0), reco::GenJet::Specific(), reco::Jet::Constituents())
  );
}


/// return the associated non-calibrated jet
TopJetType TopJet::getRecJet() const {
  TopJetType recJet;
  if (recJet_.size() > 0) {
    recJet = *this;
    recJet.setP4(recJet_.front());
  }
  return recJet;
}


/// return the flavour of the parton underlying the jet
int TopJet::getPartonFlavour() const {
  return jetFlavour_;
}


/// get b discriminant from label name
double TopJet::getBDiscriminator(std::string theLabel) const {
  double discriminator = -10.;
  if (theLabel == "default") theLabel = "trackCountingJetTags";
  for(unsigned int i=0; i!=pairDiscriVector_.size(); i++){
    if(pairDiscriVector_[i].first == theLabel){
      discriminator = pairDiscriVector_[i].second;
    }
  }
  return discriminator;
}


/// get JetTagRef from labael name
reco::JetTagRef TopJet::getBJetTagRef(std::string theLabel) const {
  reco::JetTagRef theJetTagRef ;
  //if(pairDiscriJetTagRef.size() == 0){
  //  cout << "no JetTagRef found" << endl;
  //}
  for(unsigned int i=0; i!=pairJetTagRefVector_.size(); i++){
    if(pairJetTagRefVector_[i].first == theLabel){
       theJetTagRef= pairJetTagRefVector_[i].second;
    }
  } 
  return theJetTagRef;
}


/// write out all the labels present - FIXME: should use the message logger
void TopJet::dumpBTagLabels() const {
  if(pairDiscriVector_.size() == 0){
    std::cout << "no Label found" << std::endl;
  }
  for(unsigned int i=0; i!=pairDiscriVector_.size(); i++){
    std::cout << "Label : " << pairDiscriVector_[i].first << std::endl;
  }
}


/// get the value of the i'th jet cleaning variable
double TopJet::getLRPhysicsJetVar(unsigned int i) const {
  return (i < lrPhysicsJetVarVal_.size() ? lrPhysicsJetVarVal_[i].first  : 0);
}


/// get the likelihood ratio corresponding to the i'th jet cleaning variable
double TopJet::getLRPhysicsJetVal(unsigned int i) const {
  return (i < lrPhysicsJetVarVal_.size() ? lrPhysicsJetVarVal_[i].second : 1);
}


/// get the overall jet cleaning likelihood ratio
double TopJet::getLRPhysicsJetLRval() const {
  return lrPhysicsJetLRval_;
}


/// get the overall jet cleaning probability
double TopJet::getLRPhysicsJetProb() const {
  return lrPhysicsJetProb_;
}


/// method to set the matched parton
void TopJet::setGenParton(const reco::Particle & gp) {
  genParton_.clear();
  genParton_.push_back(gp);
}


/// method to set the matched generated jet
void TopJet::setGenJet(const reco::GenJet & gj) {
  genJet_.clear();
  genJet_.push_back(gj);
}


/// method to set the uncalibrated jet
void TopJet::setRecJet(const TopJetType & rj) {
  recJet_.clear();
  recJet_.push_back(rj.p4());
}


/// method to set the flavour of the parton underlying the jet
void TopJet::setPartonFlavour(int jetFl) {
  jetFlavour_ = jetFl;
}


/// method to add a algolabel-discriminator pair
void TopJet::addBDiscriminatorPair(std::pair<std::string, double> & thePair) {
  pairDiscriVector_.push_back(thePair);
}


/// method to add a algolabel-jettagref pair
void TopJet::addBJetTagRefPair(std::pair<std::string, reco::JetTagRef> & thePair) {
  pairJetTagRefVector_.push_back(thePair);
}


/// method to set all jet cleaning variable + LR pairs
void TopJet::setLRPhysicsJetVarVal(const std::vector<std::pair<double, double> > & varValVec) {
  for (size_t i = 0; i<varValVec.size(); i++) lrPhysicsJetVarVal_.push_back(varValVec[i]);
}


/// method to set the combined jet cleaning likelihood ratio value
void TopJet::setLRPhysicsJetLRval(double clr) {
  lrPhysicsJetLRval_ = clr;
}


/// method to set the jet cleaning probability
void TopJet::setLRPhysicsJetProb(double plr) {
  lrPhysicsJetProb_  = plr;
}

/// method to return the JetCharge computed when creating the TopJet
float TopJet::getJetCharge() const {
  return jetCharge_;
}

/// method to return a vector of refs to the tracks associated to this jet
const reco::TrackRefVector & TopJet::getAssociatedTracks() const {
  return associatedTracks_;
}

