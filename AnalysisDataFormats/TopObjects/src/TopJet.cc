// -*- C++ -*-
//
// Package:     TopJet
// Class  :     TopJet
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Wed May 10 11:48:25 CEST 2006
// $Id: TopJet.cc,v 1.5 2007/06/15 23:58:05 lowette Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"

TopJet::TopJet(){ 
  lrPhysicsJetLRval_ = -999.;
  lrPhysicsJetProb_  = -999.;
}

TopJet::TopJet(JetType aJet): TopObject<JetType>(aJet) {
  lrPhysicsJetLRval_ = -999.;
  lrPhysicsJetProb_  = -999.;
}

TopJet::~TopJet(){ }


reco::Particle	TopJet::getGenJet() const	  	{ return genJet_; }
JetType		TopJet::getRecJet() const	  	{ return recJet_; }
TopParticle	TopJet::getFitJet() const	  	{ return fitJet_; }
double   	TopJet::getBDiscriminator() const 	{ return bDiscr_; }
double          TopJet::getLRPhysicsJetVar(unsigned int i) const { return (i < lrPhysicsJetVarVal_.size() ? lrPhysicsJetVarVal_[i].first  : 0); } 
double          TopJet::getLRPhysicsJetVal(unsigned int i) const { return (i < lrPhysicsJetVarVal_.size() ? lrPhysicsJetVarVal_[i].second : 1); }
double          TopJet::getLRPhysicsJetLRval() const 	{ return lrPhysicsJetLRval_; }
double          TopJet::getLRPhysicsJetProb() const 	{ return lrPhysicsJetProb_; }
double          TopJet::getPartonFlavour() const        { return jetFlavour_;}

void 		TopJet::setGenJet(reco::Particle gj)  	{ genJet_ = gj; }
void 		TopJet::setRecJet(JetType rj)     	{ recJet_ = rj; }
void 		TopJet::setFitJet(TopParticle fj) 	{ fitJet_ = fj; }
void 		TopJet::setBDiscriminator(double b)	{ bDiscr_ = b; }
void            TopJet::addBDiscriminatorPair(std::pair<std::string, double>  thepair ) { pairDiscriVector_.push_back(thepair); }
void            TopJet::addBJetTagRefPair(std::pair<std::string, reco::JetTagRef>  thepair2 ){ pairJetTagRefVector_.push_back(thepair2); }
void            TopJet::setPartonFlavour(int jetFl)      { jetFlavour_ = jetFl;}
void 		TopJet::setLRPhysicsJetVarVal(std::vector<std::pair<double, double> > varval) { for(size_t i = 0; i<varval.size(); i++) lrPhysicsJetVarVal_.push_back(varval[i]); }
void 		TopJet::setLRPhysicsJetLRval(double clr) { lrPhysicsJetLRval_ = clr;}
void 		TopJet::setLRPhysicsJetProb(double plr)  { lrPhysicsJetProb_  = plr;}






//----------------------------------------------------------------------------
//get b discriminant from label name
double          TopJet::getBDiscriminator(std::string theLabel) const {
   
  double discriminator = -999.;
  for(unsigned int i=0; i!=pairDiscriVector_.size(); i++){
    if(pairDiscriVector_[i].first == theLabel){
      discriminator = pairDiscriVector_[i].second;
    }
  }
  
  return discriminator;

}






//----------------------------------------------------------------------------
//get JetTagRef from labael name
reco::JetTagRef          TopJet::getBJetTagRef(std::string theLabel) const {
  
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

//----------------------------------------------------------------------------
//get all the labels present
void            TopJet::dumpBTagLabels() const{

 if(pairDiscriVector_.size() == 0){
   std::cout << "no Label found" << std::endl;
 }
 for(unsigned int i=0; i!=pairDiscriVector_.size(); i++){
   std::cout << "Label : " << pairDiscriVector_[i].first << std::endl;
 }
 
 
 

}
