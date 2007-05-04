// File: ExtKtJetAlgorithmWrapper.cc
// Author:  Andreas Oehler, University Karlsruhe (TH)
// Creation Date:  Feb. 1 2007 Initial version.
//--------------------------------------------


#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"

#include "RecoJets/JetAlgorithms/interface/ExtKtJetAlgorithmWrapper.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//KtJet.org
#include "KtJet/KtEvent.h"
#include "KtJet/KtLorentzVector.h"
#include <string.h>
#include <map>

//  Wrapper around ktjet-package (http://projects.hepforge.org/ktjet)
//  See Reference: Comp. Phys. Comm. vol 153/1 85-96 (2003)
//  Also:  http://www.arxiv.org/abs/hep-ph/0210022
//  this package is included in the external CMSSW-dependencies
//  License of package: GPL

using namespace edm;
using namespace std;

ExtKtJetAlgorithmWrapper::ExtKtJetAlgorithmWrapper(){
  
}

ExtKtJetAlgorithmWrapper::ExtKtJetAlgorithmWrapper(const edm::ParameterSet& ps)
{
  
  //Getting information from the ParameterSet (reading config files):
  theRparam=ps.getParameter<double>("ExtKtRParam");
  thePtMin=ps.getParameter<double>("PtMin");
  theDcut=ps.getParameter<double>("dcut");
  theNjets=ps.getParameter<int>("njets");
  theAngle=ps.getParameter<int>("KtAngle");
  theRecom=ps.getParameter<int>("KtRecom");
  
  
  theColType=4; //set to pp-collision

  //configuring algorithm 
  theMode=-1;
  
  if ((theDcut==-1)&&(theNjets==-1)) theMode=0;
  else if ((theDcut!=-1)&&(theNjets==-1)&&(thePtMin==-1)) theMode=1;
  else if ((theDcut==-1)&&(theNjets!=-1)&&(thePtMin==-1)) theMode=2;
  else LogWarning("ExtKtJetDefinition")<<"[ExtKtJetWrapper] Wrong Configuration! - will not produce KtJets!";
}

void ExtKtJetAlgorithmWrapper::run(const std::vector <FJCand>& fInput, 
				  std::vector<ProtoJet>* fOutput){
  if (theMode==-1) return;
   int index_=0;
   vector<KtJet::KtLorentzVector> theInput;
   //map<KtJet::KtLorentzVector,const std::vector<FJCand>> theMap;
   map<const unsigned int,FJCand> theIDMap;
   for (std::vector<FJCand>::const_iterator inputCand=fInput.begin();
	inputCand!=fInput.end();inputCand++){
      double px=(*inputCand)->px();
      double py=(*inputCand)->py();
      double pz=(*inputCand)->pz();
      double E=(*inputCand)->energy();
      KtJet::KtLorentzVector p(px,py,pz,E);
      //p.set_user_index(index_);
      const unsigned int lvID=p.getID();
      theIDMap[lvID]=(*inputCand);
      theInput.push_back(p);
      index_++;
   }
   
   //Construct KtEvent
   KtJet::KtEvent *ktev;
   if (theMode==0) ktev=new KtJet::KtEvent(theInput,theColType,theAngle,theRecom,theRparam);
   else ktev=new KtJet::KtEvent(theInput,theColType,theAngle,theRecom);

   if (theMode==1) ktev->findJetsD(theDcut);
   else if (theMode==2) ktev->findJetsN(theNjets);
   
   //getJets
   vector<KtJet::KtLorentzVector>* theJets=new vector<KtJet::KtLorentzVector>(ktev->getJetsPt());
   


   // run the jet clustering with the above jet definition
   
  //make CMSSW-Objects:
   std::vector<FJCand> jetConst;
   for (std::vector<KtJet::KtLorentzVector>::const_iterator itJet=theJets->begin();
	itJet!=theJets->end();itJet++){
     const std::vector<const KtJet::KtLorentzVector*>  jet_constituents = (*itJet).getConstituents();
     for (std::vector<const KtJet::KtLorentzVector*>::const_iterator itConst=jet_constituents.begin();
	  itConst!=jet_constituents.end();++itConst){
       const unsigned int tID=(*itConst)->getID();
       jetConst.push_back(theIDMap[tID]);
     }
     double px=(*itJet).px();
     double py=(*itJet).py();
     double pz=(*itJet).pz();
     double E=(*itJet).e();
     math::XYZTLorentzVector p4(px,py,pz,E);
     if (p4.Pt()>=thePtMin) fOutput->push_back(ProtoJet(p4,jetConst));
     jetConst.clear();
   }
   delete theJets;
   delete ktev;
   
}
