#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"

#include <sstream>

using std::cout;
using std::string;
using std::stringstream;
using std::endl;
using std::vector;

//DEFINE STATICS
const int L1GlobalCaloTrigger::N_SOURCE_CARDS = 54;
const int L1GlobalCaloTrigger::N_JET_LEAF_CARDS = 6;
const int L1GlobalCaloTrigger::N_EM_LEAF_CARDS = 2;
const int L1GlobalCaloTrigger::N_WHEEL_JET_FPGAS = 2;
const int L1GlobalCaloTrigger::N_WHEEL_ENERGY_FPGAS = 2;

// constructor
L1GlobalCaloTrigger::L1GlobalCaloTrigger() :
  theSourceCards(N_SOURCE_CARDS),
  theJetLeafCards(N_JET_LEAF_CARDS),
  theEmLeafCards(N_EM_LEAF_CARDS),
  theWheelJetFpgas(N_WHEEL_JET_FPGAS),
  theWheelEnergyFpgas(N_WHEEL_ENERGY_FPGAS)
{
  
  build();
  
}

L1GlobalCaloTrigger::~L1GlobalCaloTrigger()
{
  theSourceCards.clear();
}

void L1GlobalCaloTrigger::openSourceCardFiles(string fileBase){
  //Loop running over the 18 RCT-crate files, allocating 3 sourcecards per file
  for(int i = 0;i < 18; i++){
    string fileNo;
    stringstream ss;
    ss << i;
    ss >> fileNo;
    string fileName = fileBase+fileNo;
    theSourceCards[3*i]->openInputFile(fileName);
    theSourceCards[3*i+1]->openInputFile(fileName);
    theSourceCards[3*i+2]->openInputFile(fileName);
  }
}

void L1GlobalCaloTrigger::reset() {

  // Source cards
  for (int i=0; i<54; i++) {
    theSourceCards[i]->reset();
  }

  // EM Leaf Card
  for (int i=0; i<4; i++) {
    theEmLeafCards[i]->reset();
  }

  // Jet Leaf cards
  for (int i=0; i<6; i++) {
    theJetLeafCards[i]->reset();
  }

  // Wheel Cards
  for (int i=0; i<2; i++) {
    theWheelJetFpgas[i]->reset();
  }

  for (int i=0; i<2; i++) {
    theWheelEnergyFpgas[i]->reset();
  }

  // Electron Final Stage
  theIsoEmFinalStage->reset();
  theNonIsoEmFinalStage->reset();

  // Jet Final Stage
  theJetFinalStage->reset();

  // Energy Final Stage
  theEnergyFinalStage->reset();

}

void L1GlobalCaloTrigger::process() {
		
  // Source cards
  for (int i=0; i<54; i++) {
    theSourceCards[i]->fetchInput();
  }

  // EM Leaf Card
  for (int i=0; i<4; i++) {
    theEmLeafCards[i]->fetchInput();
    theEmLeafCards[i]->process();
  }

  // Jet Leaf cards
  for (int i=0; i<6; i++) {
    theJetLeafCards[i]->fetchInput();
    theJetLeafCards[i]->process();
  }

  // Wheel Cards
  for (int i=0; i<2; i++) {
    theWheelJetFpgas[i]->fetchInput();
    theWheelJetFpgas[i]->process();
  }

  for (int i=0; i<2; i++) {
    theWheelEnergyFpgas[i]->fetchInput();
    theWheelEnergyFpgas[i]->process();
  }

  // Electron Final Stage
  theIsoEmFinalStage->fetchInput();
  theIsoEmFinalStage->process();

  theNonIsoEmFinalStage->fetchInput();
  theNonIsoEmFinalStage->process();


  // Jet Final Stage
  theJetFinalStage->fetchInput();
  theJetFinalStage->process();

  // Energy Final Stage
  theEnergyFinalStage->fetchInput();
  theEnergyFinalStage->process();
	
}

void L1GlobalCaloTrigger::print() {

  cout << "=== Global Calo Trigger ===" << endl;
  cout << "=== START DEBUG OUTPUT  ===" << endl;

  cout << endl;
  cout << "N Source Cards " << theSourceCards.size() << endl;
  cout << "N Jet Leaf Cards " << theJetLeafCards.size() << endl;
  cout << "N Wheel Jet Fpgas " << theWheelJetFpgas.size() << endl;
  cout << "N Wheel Energy Fpgas " << theWheelEnergyFpgas.size() << endl;
  cout << "N Em Leaf Cards" << theEmLeafCards.size() << endl;
  cout << endl;

  for (unsigned i=0; i<theSourceCards.size(); i++) {
    cout << "Source Card : " << i << " at " << theSourceCards[i] << endl;
    //cout << (*theSourceCards[i]); 
  }
  cout << endl;

  for (unsigned i=0; i<theJetLeafCards.size(); i++) {
    cout << "Jet Leaf : " << i << endl;
    cout << (*theJetLeafCards[i]);
  }
  cout << endl;

  for (unsigned i=0; i<theWheelJetFpgas.size(); i++) {
    cout << "Wheel Jet FPGA : " << i << endl; 
    cout << (*theWheelJetFpgas[i]);
  }
  cout << endl;

  for (unsigned i=0; i<theWheelEnergyFpgas.size(); i++) {
    cout << "Wheel Energy FPGA : " << i << endl; 
    cout << (*theWheelEnergyFpgas[i]);
  }
  cout << endl;

  cout << (*theJetFinalStage);
  cout << endl;

  cout << (*theEnergyFinalStage);
  cout << endl;

  for (unsigned i=0; i<theEmLeafCards.size(); i++) {
    cout << "EM Leaf : " << i << endl;
    cout << (*theEmLeafCards[i]);
  }
  cout << endl;

//   cout << (*theIsoEmFinalStage);
//   cout << endl;

//   cout << (*theNonIsoEmFinalStage);

  cout << "=== Global Calo Trigger ===" << endl;
  cout << "===  END DEBUG OUTPUT   ===" << endl;
 
}

// isolated EM outputs
vector<L1GctEmCand> L1GlobalCaloTrigger::getIsoElectrons() { 
  return theIsoEmFinalStage->OutputCands();
}	

// non isolated EM outputs
vector<L1GctEmCand> L1GlobalCaloTrigger::getNonIsoElectrons() {
  return theNonIsoEmFinalStage->OutputCands(); 
}

// central jet outputs to GT
vector<L1GctJetCand> L1GlobalCaloTrigger::getCentralJets() {
  return theJetFinalStage->getCentralJets();
}

// forward jet outputs to GT
vector<L1GctJetCand> L1GlobalCaloTrigger::getForwardJets() { 
  return theJetFinalStage->getForwardJets(); 
}

// tau jet outputs to GT
vector<L1GctJetCand> L1GlobalCaloTrigger::getTauJets() { 
  return theJetFinalStage->getTauJets(); 
}

// total Et output
unsigned L1GlobalCaloTrigger::getEtSum() {
  return theEnergyFinalStage->getEtSum().value();
}

unsigned L1GlobalCaloTrigger::getEtHad() {
  return theEnergyFinalStage->getEtHad().value();
}

unsigned L1GlobalCaloTrigger::getEtMiss() {
  return theEnergyFinalStage->getEtMiss().value();
}

unsigned L1GlobalCaloTrigger::getEtMissPhi() {
  return theEnergyFinalStage->getEtMissPhi().value();
}





/* PRIVATE METHODS */

// instantiate hardware/algorithms
void L1GlobalCaloTrigger::build() {

  // Jet Et LUT
  m_jetEtCalLut = new L1GctJetEtCalibrationLut();

  // Source cards
  for (int i=0; i<18; i++) {
    theSourceCards[3*i]   = new L1GctSourceCard(3*i,   L1GctSourceCard::cardType1);
    theSourceCards[3*i+1] = new L1GctSourceCard(3*i+1, L1GctSourceCard::cardType2);
    theSourceCards[3*i+2] = new L1GctSourceCard(3*i+2, L1GctSourceCard::cardType3);
  }

  // Now we have the source cards prepare vectors of the relevent cards for the connections

  // Jet leaf cards
  vector<L1GctSourceCard*> jetSourceCards(15);

   // Jet Leaf cards
  for (int i=0; i<6; i++) {
    for (int j=0; j<6; j++) {
      int k = 3*(j/2) + (j%2) + 1;
      jetSourceCards[j]=theSourceCards[(i*9+k)];
      // Neighbour connections
      int iup = (i*3+3) % 9;
      int idn = (i*3+8) % 9;
      int ii, i0, i1, i2, i3, i4, i5;
      if (i<3) {
	ii = iup;
	// Remaining connections for the TDR jetfinder only
	i0 = idn;
	i1 = idn+9;
	i2 = i*3+9;
	i3 = i*3+10;
	i4 = i*3+11;
	i5 = iup+9;
      } else {
	ii = iup+9;
	// Remaining connections for the TDR jetfinder only
	i0 = idn+9;
	i1 = idn;
	i2 = i*3-9;
	i3 = i*3-8;
	i4 = i*3-7;
	i5 = iup;
      }
      jetSourceCards[3] = theSourceCards[ii*3+2];
      jetSourceCards[4] = theSourceCards[ii*3+1];
      jetSourceCards[5] = theSourceCards[ii*3+2];
      jetSourceCards[6] = theSourceCards[ii*3+1];
      jetSourceCards[7] = theSourceCards[ii*3+2];
      // Remaining connections for the TDR jetfinder only
      jetSourceCards[8] = theSourceCards[i0*3+1];
      jetSourceCards[9] = theSourceCards[i0*3+2];
      jetSourceCards[10]= theSourceCards[i1*3+1];
      jetSourceCards[11]= theSourceCards[i2*3+1];
      jetSourceCards[12]= theSourceCards[i3*3+1];
      jetSourceCards[13]= theSourceCards[i4*3+1];
      jetSourceCards[14]= theSourceCards[i5*3+1];
      //
      
    }
    theJetLeafCards[i] = new L1GctJetLeafCard(i,i % 3,jetSourceCards, m_jetEtCalLut);
  }

  // EM leaf cards  
  vector<L1GctSourceCard*> emSourceCards(9);

  for (int i=0; i<2; i++) {
    for (int j=0; j<9; j++) {
      emSourceCards[j]=theSourceCards[(i*9+j)*3];
    }
    theEmLeafCards[i] = new L1GctEmLeafCard(i,emSourceCards);
  }

   // Wheel Fpgas
   vector<L1GctJetLeafCard*> wheelJetLeafCards(3);
   vector<L1GctJetLeafCard*> wheelEnergyLeafCards(3);

   for (int i=0; i<2; i++) {
     for (int j=0; j<3; j++) {
       wheelJetLeafCards[j]=theJetLeafCards[i*2+j];
       wheelEnergyLeafCards[j]=theJetLeafCards[i*2+j];
     }
     theWheelJetFpgas[i] = new L1GctWheelJetFpga(i,wheelJetLeafCards);
     theWheelEnergyFpgas[i] = new L1GctWheelEnergyFpga(i,wheelEnergyLeafCards);
   }
  
   // Jet Final Stage  
   theJetFinalStage = new L1GctJetFinalStage(theWheelJetFpgas);

  // Electron Final Sort
   theIsoEmFinalStage = new L1GctElectronFinalSort(true,theEmLeafCards[0], theEmLeafCards[1]);
   theNonIsoEmFinalStage = new L1GctElectronFinalSort(false,theEmLeafCards[0], theEmLeafCards[1]);  

  // Global Energy Algos
  theEnergyFinalStage = new L1GctGlobalEnergyAlgos(theWheelEnergyFpgas, theWheelJetFpgas);

}
