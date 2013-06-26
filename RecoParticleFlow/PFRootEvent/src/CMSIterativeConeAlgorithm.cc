// Original author: A. Ulyanov
// $Id: CMSIterativeConeAlgorithm.cc,v 1.1 2009/08/24 14:35:59 srappocc Exp $

#include "RecoParticleFlow/PFRootEvent/interface/CMSIterativeConeAlgorithm.h"

#include <list>

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoParticleFlow/PFRootEvent/interface/ProtoJet.h"
#include "RecoParticleFlow/PFRootEvent/interface/JetAlgoHelper.h"
using namespace std;
using namespace reco;
using namespace JetReco;



//  Run the algorithm
//  ------------------
void CMSIterativeConeAlgorithm::run(const InputCollection& fInput, OutputCollection* fOutput) const {
  if (!fOutput) return;
  //make a list of input objects ordered by ET
  list<InputItem> input;
  for (InputCollection::const_iterator towerIter = fInput.begin();
       towerIter != fInput.end(); ++towerIter) {
    input.push_back(*towerIter);
  }   
  GreaterByEtRef <InputItem> compCandidate;
  input.sort(compCandidate);

  //find jets
  while( !input.empty() && input.front()->et() > theSeedThreshold ) {
    //cone centre 
    double eta0=input.front()->eta();
    double phi0=input.front()->phi();
    //protojet properties
    double eta=0;
    double phi=0;
    double et=0;
    //list of towers in cone
    list< list<InputItem>::iterator> cone;
    for(int iteration=0;iteration<100;iteration++){
      cone.clear();
      eta=0;
      phi=0;
      et=0;
      for(list<InputItem>::iterator inp=input.begin();
	  inp!=input.end();inp++){
	InputItem tower = *inp;	
	if( deltaR2(eta0,phi0,tower->eta(),tower->phi()) < 
	    theConeRadius*theConeRadius) {
          cone.push_back(inp);
          eta+= tower->et()*tower->eta();
          double dphi=tower->phi()-phi0;
          if(dphi>M_PI) dphi-=2*M_PI;
          else if(dphi<=-M_PI) dphi+=2*M_PI;
          phi+=tower->et()*dphi;
          et +=tower->et();
        }
      }
      eta=eta/et;
      phi=phi0+phi/et;
      if(phi>M_PI)phi-=2*M_PI;
      else if(phi<=-M_PI)phi+=2*M_PI;
      
      if(fabs(eta-eta0)<.001 && fabs(phi-phi0)<.001) break;//stable cone found
      eta0=eta;
      phi0=phi;
    }

    //make a final jet and remove the jet constituents from the input list 
    InputCollection jetConstituents;     
    list< list<InputItem>::iterator>::const_iterator inp;
    for(inp=cone.begin();inp!=cone.end();inp++)  {
      jetConstituents.push_back(**inp);
      input.erase(*inp);
    }
    fOutput->push_back (ProtoJet (jetConstituents));

  } //loop over seeds ended
  GreaterByPt<ProtoJet> compJets;
  sort (fOutput->begin (), fOutput->end (), compJets);
}
   
