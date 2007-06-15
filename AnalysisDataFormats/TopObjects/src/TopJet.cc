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
// $Id: TopJet.cc,v 1.4 2007/06/15 16:47:46 heyninck Exp $
//

// system include files

// user include files
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"

TopJet::TopJet(){ 
  lrPhysicsJetLRval	= -999.;
  lrPhysicsJetProb	= -999.;
}

TopJet::TopJet(JetType aJet): TopObject<JetType>(aJet) {
  lrPhysicsJetLRval	= -999.;
  lrPhysicsJetProb	= -999.;
}

TopJet::~TopJet(){ }

void 		TopJet::setGenJet(reco::Particle gj)  	{ genJet = gj; }
void 		TopJet::setRecJet(JetType rj)     	{ recJet = rj; }
void 		TopJet::setFitJet(TopParticle fj) 	{ fitJet = fj; }
void 		TopJet::setBdiscriminant(double b)	{ bdiscr = b; }
void 		TopJet::setLRPhysicsJetVarVal(std::vector<std::pair<double, double> > varval) { for(size_t i = 0; i<varval.size(); i++) lrPhysicsJetVarVal.push_back(varval[i]); }
void 		TopJet::setLRPhysicsJetLRval(double clr) {lrPhysicsJetLRval = clr;}
void 		TopJet::setLRPhysicsJetProb(double plr)  {lrPhysicsJetProb  = plr;}

void            TopJet::addBdiscriminantPair( std::pair<std::string, double>  thepair ){ pairDiscriVector.push_back(thepair);}
void            TopJet::addBJetTagRefPair( std::pair<std::string, reco::JetTagRef>  thepair2 ){ pairDiscriJetTagRef.push_back(thepair2);}
void            TopJet::setQuarkFlavour(int jetf){jetFlavour = jetf;}



reco::Particle	TopJet::getGenJet() const	  	{ return genJet; }
JetType		TopJet::getRecJet() const	  	{ return recJet; }
TopParticle	TopJet::getFitJet() const	  	{ return fitJet; }
double   	TopJet::getBdiscriminant() const 	{ return bdiscr; }
double          TopJet::getLRPhysicsJetVar(unsigned int i) const { return (i < lrPhysicsJetVarVal.size() ? lrPhysicsJetVarVal[i].first  : 0); } 
double          TopJet::getLRPhysicsJetVal(unsigned int i) const { return (i < lrPhysicsJetVarVal.size() ? lrPhysicsJetVarVal[i].second : 1); }
double          TopJet::getLRPhysicsJetLRval() const 	{ return lrPhysicsJetLRval; }
double          TopJet::getLRPhysicsJetProb() const 	{ return lrPhysicsJetProb; }
double          TopJet::getQuarkFlavour() const         { return jetFlavour;}





//----------------------------------------------------------------------------
//get b discriminant from label name
double          TopJet::getBdiscriminantFromPair(std::string theLabel) const{
   
  double discriminator = -999.;
  for(unsigned int i=0; i!=pairDiscriVector.size(); i++){
    if(pairDiscriVector[i].first == theLabel){
      discriminator = pairDiscriVector[i].second;
    }
  }
  
  return discriminator;

}






//----------------------------------------------------------------------------
//get JetTagRef from labael name
reco::JetTagRef          TopJet::getBJetTagRefFromPair(std::string theLabel) const{
  
  reco::JetTagRef theJetTagRef ;
  
  //if(pairDiscriJetTagRef.size() == 0){
  //  cout << "no JetTagRef found" << endl;
  //}
  for(unsigned int i=0; i!=pairDiscriJetTagRef.size(); i++){
    if(pairDiscriJetTagRef[i].first == theLabel){
       theJetTagRef= pairDiscriJetTagRef[i].second;
    }
  } 
  
  return  theJetTagRef;
  
}

//----------------------------------------------------------------------------
//get all the labels present
void            TopJet::dumpBTagLabels() const{

 if(pairDiscriVector.size() == 0){
   std::cout << "no Label found" << std::endl;
 }
 for(unsigned int i=0; i!=pairDiscriVector.size(); i++){
   std::cout << "Label : " << pairDiscriVector[i].first << std::endl;
 }
 
 
 

}
