#include "RecoJets/JetAlgorithms/interface/CMSKtJetAlgorithm.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"
#include "RecoJets/JetAlgorithms/interface/MakeCaloJet.h"

#include "RecoJets/JetAlgorithms/interface/KtEvent.h"
#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "RecoJets/JetAlgorithms/interface/JetableObjectHelper.h"

#include <stdio.h>
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

/** Implemented by Fernando Varela Rodriguez

    Based on the ORCA implmentation of the Kt Algorithm by Arno Heister
 
    19-Oct-2005: R. Harris Wasn't able to port it to the new CaloTower implementation, 
                 so it just returns an empty jet collection, and couts an error message.
                 See comments below in the code.
    12-Jan-2006  R. Harris.  Now works with the new CaloTower implementation. Discovered
                 algorithm only successfully clusters 2 or more towers: fails to cluster 
		 TB events with single pions putting energy in a single tower. Caveat Emptor!
*/


CMSKtJetAlgorithm::CMSKtJetAlgorithm()
{
  theKtJetType       = 4;    // for pp collisons
  theKtJetAngle      = 2;    // angular
  theKtJetRecom      = 1;    // E
  theKtJetRParameter = 1.0;  // value corresponding to the Snowmass convention
  theKtJetECut = 0.;
}

CMSKtJetAlgorithm::CMSKtJetAlgorithm(int aKtJetAngle,int aKtJetRecom, float aKtJetECut)
{
  theKtJetType  = 4;            // for pp collisons
  theKtJetAngle = aKtJetAngle;
  theKtJetRecom = aKtJetRecom;
  theKtJetRParameter = 1.0;     // value corresponding to the Snowmass convention
  theKtJetECut = aKtJetECut;
}

CMSKtJetAlgorithm::CMSKtJetAlgorithm(int aKtJetAngle,int aKtJetRecom, float aKtJetECut, float aKtJetRParameter)
{
  theKtJetType       = 4; // for pp collisons
  theKtJetAngle      = aKtJetAngle;
  theKtJetRecom      = aKtJetRecom;
  theKtJetRParameter = aKtJetRParameter;
  theKtJetECut       = aKtJetECut;
}

void CMSKtJetAlgorithm::setKtJetAngle(int aKtJetAngle)
{
  theKtJetAngle = aKtJetAngle;
}

void CMSKtJetAlgorithm::setKtJetRecom(int aKtJetRecom)
{
  theKtJetRecom = aKtJetRecom;
}

void CMSKtJetAlgorithm::setKtJetRParameter(float aKtJetRParameter)
{
  theKtJetRParameter = aKtJetRParameter; 
}
void CMSKtJetAlgorithm::setKtJetECut(float aKtJetECut)
{
  theKtJetECut = aKtJetECut; 
}

CaloJetCollection* CMSKtJetAlgorithm::findJets(const CaloTowerCollection & aTowerCollection)
{
  CaloJetCollection* result = new CaloJetCollection;
  
  // fill the KtLorentzVector
  std::vector<KtJet::KtLorentzVector> avec;
  for (CaloTowerCollection::const_iterator i = aTowerCollection.begin(); 
                                              i != aTowerCollection.end(); 
					      i++){
    if((i->e()) >= theKtJetECut){
      float e = i->e();
      float px = e * (1.0/cosh(i->eta)) * cos(i->phi);
      float py = e * (1.0/cosh(i->eta)) * sin(i->phi);
      float pz = e * tanh(i->eta); 
    
      HepLorentzVector theLorentzVector;
    
      theLorentzVector.setPx(px);
      theLorentzVector.setPy(py);
      theLorentzVector.setPz(pz);
      theLorentzVector.setE(e);
      
      avec.push_back(theLorentzVector);
    }
  }
  
  // R. Harris, 1/12/06, If there are no inputs, simply return empty CaloJetCollection.
  // Prevents call to KtEvent which crashes if called with empty vector of inputs.
  if( avec.size() == 0 ) return result;
  
  // construct the KtEvent object
  KtJet::KtEvent ev(avec,theKtJetType,theKtJetAngle,theKtJetRecom,theKtJetRParameter);
  
  // retrieve the final state jets as an array of KtLorentzVectors from KtEvent sorted by Et
  std::vector<KtJet::KtLorentzVector> jets = ev.getJetsEt();
  
  // fill jets into the result JetCollection
  //Create ProtoJets from the KtLorentz Vectors 
  std::vector<ProtoJet> ProtoJetCollection;
  
  //For each jet, get the list the list of input constituents the jet consists of:
  for(std::vector<KtJet::KtLorentzVector>::const_iterator itr = jets.begin(); 
                                                          itr != jets.end(); 
							  ++itr){	
    //For each of the jets get its final constituents:
    std::vector<KtJet::KtLorentzVector> constituents = itr->copyConstituents();

    //Loop over all input constituents and try to find them as jet final constituents
    int nInputConstituent = 0; //# of input constituent in the STL vector (i.e. in the CaloTowerCollection)
    int nConstituent = 0;      //# of final constituent of the Jet
    std::vector<unsigned int> listAssignedInputs; // Array containing the list of indices of the array of inputs
                                                  // (i.e. CaloTowerCollection) found in the final list of Jet
						  // constituents
   
    for(unsigned int j = 0; j < avec.size(); j++){
      if(itr->contains(avec[j])){
	nConstituent++;

        //Make sure that an input constituent is assigned to a single jet
	std::vector<unsigned int>::iterator found_iter = find(listAssignedInputs.begin(), listAssignedInputs.end(), j); // Search the list.
        if(found_iter != listAssignedInputs.end())
          printf("[KtJetAlgorithm]->ERROR: input constituent %d assigned to multiple Jets. Continuing...", j);
        else 
	  listAssignedInputs.push_back(j);	
      }    
      nInputConstituent++;
    }

    if(listAssignedInputs.size() != constituents.size())      
      printf("[KtJetAltgorithm]->ERROR:Could not find all jet constituents in the input list\n");
    else{
      //Everything went ok -> pick the subset of CaloTowers forming the Jet:
      //Instancite the Sub Collection of CaloTowers
      std::vector<const CaloTower*> jetSubCollection;

      //Populate the subcollection with objects from the input collection
      for(unsigned int k = 0; k < listAssignedInputs.size(); k++)
      jetSubCollection.push_back(&aTowerCollection[listAssignedInputs[k]]);

      ProtoJet pj;
      
      pj.putTowers(jetSubCollection);

      ProtoJetCollection.push_back(pj);    
      
      listAssignedInputs.clear();  //Not needed anymore
      
    }
  }
 
  //Now we have a collection of ProtoJets-> instanciate the CaloJetCollection:
  MakeCaloJet(aTowerCollection, ProtoJetCollection, *result);
  
  return result;
}
