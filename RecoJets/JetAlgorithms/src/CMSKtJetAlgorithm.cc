#include "DataFormats/Candidate/interface/Candidate.h"
#include "RecoJets/JetAlgorithms/interface/ProtoJet.h"

#include "RecoJets/JetAlgorithms/interface/KtEvent.h"
#include "RecoJets/JetAlgorithms/interface/KtLorentzVector.h"

#include "RecoJets/JetAlgorithms/interface/CMSKtJetAlgorithm.h"

using namespace std;
using namespace reco;

/** Implemented by Fernando Varela Rodriguez

    Based on the ORCA implmentation of the Kt Algorithm by Arno Heister
 
    19-Oct-2005: R. Harris Wasn't able to port it to the new CaloTower implementation, 
                 so it just returns an empty jet collection, and couts an error message.
                 See comments below in the code.
    12-Jan-2006  R. Harris.  Now works with the new CaloTower implementation. Discovered
                 algorithm only successfully clusters 2 or more towers: fails to cluster 
		 TB events with single pions putting energy in a single tower. Caveat Emptor!
*/

namespace {
  int DEBUG = 0;
}

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

void CMSKtJetAlgorithm::run (const InputCollection& fInput, OutputCollection* fOutput)
{
  if (!fOutput) return;
  fOutput->clear ();
  // fill the KtLorentzVector
  if (DEBUG >= 1) {
    std::cout << "CMSKtJetAlgorithm::run-> " << fInput.size () << "input towers" << std::endl;
  }
  std::vector<int> indexMap;
  std::vector<KtJet::KtLorentzVector> ktInput;
  int index = 0;
  for (; index < (int)fInput.size (); index++) {
    const Candidate* constituent = fInput [index];
    if (constituent->energy() >= theKtJetECut) {
      ktInput.push_back (KtJet::KtLorentzVector (constituent->px (), constituent->py (), 
						 constituent->pz (), constituent->energy ()));
      indexMap.push_back (index);
      if (DEBUG >= 2) {
	printf ("tower: index:%4d eta:%5.2f phi:%5.2f e:%7.2f px:%7.2f py:%7.2f pz:%7.2f\n",
		index, constituent->eta(), constituent->phi(), constituent->energy (),
		constituent->px (), constituent->py (), constituent->pz ());
      }
    }
  }
  if (DEBUG >= 1) {
    std::cout << "CMSKtJetAlgorithm::run-> " << ktInput.size () << " towers above " << theKtJetECut << std::endl;
  }
  
  // R. Harris, 1/12/06, If there are no inputs, simply return empty CaloJetCollection.
  // Prevents call to KtEvent which crashes if called with empty vector of inputs.
  if( ktInput.size() == 0 ) return;
  
  // construct the KtEvent object
  KtJet::KtEvent ev(ktInput,theKtJetType,theKtJetAngle,theKtJetRecom,theKtJetRParameter);
  
  // retrieve the final state jets as an array of KtLorentzVectors from KtEvent sorted by Et
  std::vector<KtJet::KtLorentzVector> jets = ev.getJetsEt();
  
  // fill jets into the result JetCollection
  //For each jet, get the list the list of input constituents the jet consists of:
  for(std::vector<KtJet::KtLorentzVector>::const_iterator itr = jets.begin(); 
                                                          itr != jets.end(); 
							  ++itr){	
    //For each of the jets get its final constituents:
    std::vector<const Candidate*> protoJetConstituents;
    const std::vector<const KtJet::KtLorentzVector*>* constituents = &(itr->getConstituents());
    //Loop over all input constituents and try to find them as jet final constituents
    int constit = constituents->size ();
    while (--constit > 0) {
      int ktInputIndex = ktInput.size ();
      while (--ktInputIndex >= 0) {
	// std::cout << "Comparing objects: " << (*constituents)[constit] << " and " << &(ktInput[ktInputIndex]) << std::endl;
	if (*(*constituents)[constit] == ktInput[ktInputIndex]) { // found
	  protoJetConstituents.push_back (fInput [indexMap [ktInputIndex]]);
	  break;
	}
      }
      if (ktInputIndex < 0) {
	std::cerr << "[KtJetAltgorithm]->ERROR:Could not find jet constituent #" 
		  << constit << " in the input list" << std::endl;
      }
    }

    fOutput->push_back (ProtoJet (protoJetConstituents));
    if (DEBUG >= 2) {
      ProtoJet pjet (protoJetConstituents);
      printf ("jet: eta:%5.2f phi:%5.2f e:%7.2f px:%7.2f py:%7.2f pz:%7.2f\n",
	      pjet.eta(), pjet.phi(), pjet.e (),
	      pjet.px (), pjet.py (), pjet.pz ());
    }
  }
  if (DEBUG >= 1) {
    std::cout << "CMSKtJetAlgorithm::run-> " << fOutput->size () << " jets found" << std::endl;
  }
}
