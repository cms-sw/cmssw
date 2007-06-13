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
// $Id: TopJet.cc,v 1.2 2007/05/23 09:00:15 heyninck Exp $
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

void            TopJet::AddBdiscriminantPair( pair<string,double>  thepair ){ PairDiscriVector.push_back(thepair);}
void            TopJet::AddBJetTagRefPair( pair<string,JetTagRef>  thepair2 ){ PairDiscriJetTagRef.push_back(thepair2);}
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
double          TopJet::getBdiscriminantFromPair(string theLabel) const{
   
  double discriminator = -10000000;
  
  vector<pair<string,double> > pairdiscri = PairDiscriVector;
  if(pairdiscri.size() == 0){
    cout << "no discriminators found" << endl;
    return discriminator;
  }
  for(unsigned int i=0; i!=pairdiscri.size(); i++){
    if(pairdiscri[i].first == theLabel){
      discriminator = pairdiscri[i].second;
    }
  }
  
  
  return discriminator;

}






//----------------------------------------------------------------------------
//get JetTagRef from labael name
JetTagRef          TopJet::getBJetTagRefFromPair(string theLabel) const{
  
  JetTagRef theJetTagRef ;
  
  if(PairDiscriJetTagRef.size() == 0){
    cout << "no JetTagRef found" << endl;
  }
  vector<pair<string,JetTagRef> > pairJetTagRef = PairDiscriJetTagRef;
  for(unsigned int i=0; i!=pairJetTagRef.size(); i++){
    if(pairJetTagRef[i].first == theLabel){
       theJetTagRef= pairJetTagRef[i].second;
    }
  } 
  
  return  theJetTagRef;
  
}

//----------------------------------------------------------------------------
//get all the labels present
void            TopJet::DumpLabel() const{

vector<pair<string,double> > pairdiscri = PairDiscriVector;
 if(pairdiscri.size() == 0){
   cout << "no Label found" << endl;
 }
 for(unsigned int i=0; i!=pairdiscri.size(); i++){
   cout << "Label : " << pairdiscri[i].first << endl;
 }
 
 
 

}
