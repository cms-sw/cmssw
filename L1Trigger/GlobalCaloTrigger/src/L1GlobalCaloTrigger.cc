#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctSourceCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounterLut.h"

#include "L1Trigger/L1Scales/interface/L1CaloEtScale.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
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
const int L1GlobalCaloTrigger::N_WHEEL_CARDS = 2;

const unsigned int L1GlobalCaloTrigger::N_JET_COUNTERS_PER_WHEEL = L1GctWheelJetFpga::N_JET_COUNTERS;


// constructor
L1GlobalCaloTrigger::L1GlobalCaloTrigger(bool useFile, L1GctJetLeafCard::jetFinderType jfType, string jetEtLutFile) :
  readFromFile(useFile),
  theSourceCards(N_SOURCE_CARDS),
  theJetLeafCards(N_JET_LEAF_CARDS),
  theEmLeafCards(N_EM_LEAF_CARDS),
  theWheelJetFpgas(N_WHEEL_CARDS),
  theWheelEnergyFpgas(N_WHEEL_CARDS)// ,
{

  // set default et scale
  m_defaultJetEtScale = new L1CaloEtScale();

  // Jet Et LUT
  m_jetEtCalLut = new L1GctJetEtCalibrationLut(jetEtLutFile);
  //  m_jetEtCalLut->setOutputEtScale(m_defaultJetEtScale);

  // construct hardware
  build(jfType);
  
  // jet counter LUT
  setupJetCounterLuts();

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
    theSourceCards.at(3*i)->openInputFile(fileName);
    theSourceCards.at(3*i+1)->openInputFile(fileName);
    theSourceCards.at(3*i+2)->openInputFile(fileName);
  }
}

void L1GlobalCaloTrigger::reset() {

  // Source cards
  for (int i=0; i<N_SOURCE_CARDS; i++) {
    theSourceCards.at(i)->reset();
  }

  // EM Leaf Card
  for (int i=0; i<N_EM_LEAF_CARDS; i++) {
    theEmLeafCards.at(i)->reset();
  }

  // Jet Leaf cards
  for (int i=0; i<N_JET_LEAF_CARDS; i++) {
    theJetLeafCards.at(i)->reset();
  }

  // Wheel Cards
  for (int i=0; i<N_WHEEL_CARDS; i++) {
    theWheelJetFpgas.at(i)->reset();
  }

  for (int i=0; i<N_WHEEL_CARDS; i++) {
    theWheelEnergyFpgas.at(i)->reset();
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
  for (int i=0; i<N_SOURCE_CARDS; i++) {
    if (readFromFile) {
      theSourceCards.at(i)->readBX();
    }
  }

  // EM Leaf Card
  for (int i=0; i<N_EM_LEAF_CARDS; i++) {
    theEmLeafCards.at(i)->fetchInput();
    theEmLeafCards.at(i)->process();
  }

  // Jet Leaf cards - first stage processing
  for (int i=0; i<N_JET_LEAF_CARDS; i++) {
    theJetLeafCards.at(i)->fetchInput();
  }

  // Jet Leaf cards - second stage processing
  for (int i=0; i<N_JET_LEAF_CARDS; i++) {
    theJetLeafCards.at(i)->process();
  }

  // Wheel Cards
  for (int i=0; i<N_WHEEL_CARDS; i++) {
    theWheelJetFpgas.at(i)->fetchInput();
    theWheelJetFpgas.at(i)->process();
  }

  for (int i=0; i<N_WHEEL_CARDS; i++) {
    theWheelEnergyFpgas.at(i)->fetchInput();
    theWheelEnergyFpgas.at(i)->process();
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

void L1GlobalCaloTrigger::setRegion(L1CaloRegion region) {

  if (readFromFile) {
    throw cms::Exception("L1GctInputError")
      << " L1 Global Calo Trigger is set to read input data from file, "
      << " setRegion method should not be used\n"; 
  }
  unsigned scnum = region.gctCard();
  unsigned input = region.gctRegionIndex();
  L1GctSourceCard* sc = theSourceCards.at(scnum);
  std::vector<L1CaloRegion> tempRegions = sc->getRegions();
  tempRegions.at(input) = region;
  sc->setRegions(tempRegions);
}

void L1GlobalCaloTrigger::setRegion(unsigned et, unsigned ieta, unsigned iphi, bool overFlow, bool fineGrain)
{
  L1CaloRegion temp(et, overFlow, fineGrain, false, false, ieta, iphi);
  setRegion(temp);
}

void L1GlobalCaloTrigger::setIsoEm(L1CaloEmCand em) {

  if (readFromFile) {
    throw cms::Exception("L1GctInputError")
      << " L1 Global Calo Trigger is set to read input data from file, "
      << " setIsoEm method should not be used\n"; 
  }
  unsigned scnum = em.rctCrate()*3;
  L1GctSourceCard* sc = theSourceCards.at(scnum);
  std::vector<L1CaloEmCand> tempEmCands = sc->getIsoElectrons();
  for (uint input=0; input<tempEmCands.size();input++){
    if (tempEmCands.at(input).rank()==0) {
      tempEmCands.at(input) = em;
      break;
    }
  }
  sc->setIsoEm(tempEmCands);
}

void L1GlobalCaloTrigger::setNonIsoEm(L1CaloEmCand em) {

  if (readFromFile) {
    throw cms::Exception("L1GctInputError")
      << " L1 Global Calo Trigger is set to read input data from file, "
      << " setIsoEm method should not be used\n"; 
  }
  unsigned scnum = em.rctCrate()*3;
  L1GctSourceCard* sc = theSourceCards.at(scnum);
  std::vector<L1CaloEmCand> tempEmCands = sc->getNonIsoElectrons();
  for (uint input=0; input<tempEmCands.size();input++){
    if (tempEmCands.at(input).rank()==0) {
      tempEmCands.at(input) = em;
      break;
    }
  }
  sc->setNonIsoEm(tempEmCands);
}

void L1GlobalCaloTrigger::fillRegions(vector<L1CaloRegion> rgn)
{
  for (uint i=0; i<rgn.size(); i++){
    setRegion(rgn.at(i));
  }
}

void L1GlobalCaloTrigger::fillEmCands(vector<L1CaloEmCand> em)
{
  for (uint i=0; i<em.size(); i++){
    if (em.at(i).isolated()){
      setIsoEm(em.at(i));
    } else {
      setNonIsoEm(em.at(i));
    }
  }
}

void L1GlobalCaloTrigger::print() {

  cout << "=== Global Calo Trigger ===" << endl;
  cout << "=== START DEBUG OUTPUT  ===" << endl;

  cout << endl;
  cout << "N Source Cards " << theSourceCards.size() << endl;
  cout << "N Jet Leaf Cards " << theJetLeafCards.size() << endl;
  cout << "N Wheel Jet Fpgas " << theWheelJetFpgas.size() << endl;
  cout << "N Wheel Energy Fpgas " << theWheelEnergyFpgas.size() << endl;
  cout << "N Em Leaf Cards " << theEmLeafCards.size() << endl;
  cout << endl;

  for (unsigned i=0; i<theSourceCards.size(); i++) {
    cout << "Source Card " << i << " : " << theSourceCards.at(i) << endl;
    //cout << (*theSourceCards.at(i)); 
  }
  cout << endl;

  for (unsigned i=0; i<theJetLeafCards.size(); i++) {
    cout << "Jet Leaf Card " << i << " : " << theJetLeafCards.at(i) << endl;
    cout << (*theJetLeafCards.at(i));
  }
  cout << endl;

  for (unsigned i=0; i<theWheelJetFpgas.size(); i++) {
    cout << "Wheel Jet FPGA " << i << " : " << theWheelJetFpgas.at(i) << endl; 
    cout << (*theWheelJetFpgas.at(i));
  }
  cout << endl;

  for (unsigned i=0; i<theWheelEnergyFpgas.size(); i++) {
    cout << "Wheel Energy FPGA " << i <<" : " << theWheelEnergyFpgas.at(i) << endl; 
    cout << (*theWheelEnergyFpgas.at(i));
  }
  cout << endl;

  cout << (*theJetFinalStage);
  cout << endl;

  cout << (*theEnergyFinalStage);
  cout << endl;

  for (unsigned i=0; i<theEmLeafCards.size(); i++) {
    cout << "EM Leaf Card " << i << " : " << theEmLeafCards.at(i) << endl;
    cout << (*theEmLeafCards.at(i));
  }
  cout << endl;

  cout << (*theIsoEmFinalStage);
  cout << endl;

  cout << (*theNonIsoEmFinalStage);

  cout << "=== Global Calo Trigger ===" << endl;
  cout << "===  END DEBUG OUTPUT   ===" << endl;
 
}

// isolated EM outputs
vector<L1GctEmCand> L1GlobalCaloTrigger::getIsoElectrons() const { 
  return theIsoEmFinalStage->getOutputCands();
}	

// non isolated EM outputs
vector<L1GctEmCand> L1GlobalCaloTrigger::getNonIsoElectrons() const {
  return theNonIsoEmFinalStage->getOutputCands(); 
}

// central jet outputs to GT
vector<L1GctJet> L1GlobalCaloTrigger::getCentralJets() const {
  return theJetFinalStage->getCentralJets();
}

// forward jet outputs to GT
vector<L1GctJet> L1GlobalCaloTrigger::getForwardJets() const { 
  return theJetFinalStage->getForwardJets(); 
}

// tau jet outputs to GT
vector<L1GctJet> L1GlobalCaloTrigger::getTauJets() const { 
  return theJetFinalStage->getTauJets(); 
}

// total Et output
L1GctScalarEtVal L1GlobalCaloTrigger::getEtSum() const {
  return theEnergyFinalStage->getEtSum();
}

L1GctScalarEtVal L1GlobalCaloTrigger::getEtHad() const {
  return theEnergyFinalStage->getEtHad();
}

L1GctScalarEtVal L1GlobalCaloTrigger::getEtMiss() const {
  return theEnergyFinalStage->getEtMiss();
}

L1GctEtAngleBin L1GlobalCaloTrigger::getEtMissPhi() const {
  return theEnergyFinalStage->getEtMissPhi();
}

L1GctJcFinalType L1GlobalCaloTrigger::getJetCount(unsigned jcnum) const {
  return theEnergyFinalStage->getJetCount(jcnum);
}



/* PRIVATE METHODS */

// instantiate hardware/algorithms
void L1GlobalCaloTrigger::build(L1GctJetLeafCard::jetFinderType jfType) {

  // Source cards
  for (int i=0; i<(N_SOURCE_CARDS/3); i++) {
    theSourceCards.at(3*i)   = new L1GctSourceCard(3*i,   L1GctSourceCard::cardType1);
    theSourceCards.at(3*i+1) = new L1GctSourceCard(3*i+1, L1GctSourceCard::cardType2);
    theSourceCards.at(3*i+2) = new L1GctSourceCard(3*i+2, L1GctSourceCard::cardType3);
  }

  // Now we have the source cards prepare vectors of the relevent cards for the connections

  // Jet leaf cards
  vector<L1GctSourceCard*> jetSourceCards(15);

  // Jet Leaf cards
  for (int jlc=0; jlc<N_JET_LEAF_CARDS; jlc++) {
    // Define local constant for ease of typing
    static const int NL = N_JET_LEAF_CARDS/2;
    // neighbour leaf cards
    int nlc = (jlc+NL) % N_JET_LEAF_CARDS;         // The neighbour across the eta=0 boundary
    int jup = (NL*(jlc/NL)) + ((jlc+1)%NL);        // The adjacent leaf in increasing phi direction
    int jdn = (NL*(jlc/NL)) + ((jlc+(NL-1))%NL);   // The adjacent leaf in decreasing phi direction
    int nup = (jup+NL) % N_JET_LEAF_CARDS;         // The leaf across eta=0 and increasing phi
    int ndn = (jdn+NL) % N_JET_LEAF_CARDS;         // The leaf across eta=0 and decreasing phi
    // Initialise index for counting SourceCards for this JetLeafCard
    int sc = 0;
    // Each Leaf card contains three jetFinders
    for (int jf=0; jf<3; jf++) {
      // Source card numbering:
      // 3*i+1 cover the endcap and HF regions
      // 3*i+2 cover the barrel regions
      //
      // Three source cards for each jetFinder:
      // First is endcap source card
      jetSourceCards.at(sc++) = theSourceCards.at((jlc*9) + (3*jf) + 1);
      // Second is barrel source card
      jetSourceCards.at(sc++) = theSourceCards.at((jlc*9) + (3*jf) + 2);
      // Third is the barrel source card that supplies
      // data from across the eta=0 boundary
      jetSourceCards.at(sc++) = theSourceCards.at((nlc*9) + (3*jf) + 2);
    }
    // Neighbour connections
    jetSourceCards.at(sc++)=theSourceCards.at((jup*9+1));
    jetSourceCards.at(sc++)=theSourceCards.at((jup*9+2));
    jetSourceCards.at(sc++)=theSourceCards.at((nup*9+2));
    jetSourceCards.at(sc++)=theSourceCards.at((jdn*9+7));
    jetSourceCards.at(sc++)=theSourceCards.at((jdn*9+8));
    jetSourceCards.at(sc++)=theSourceCards.at((ndn*9+8));

    theJetLeafCards.at(jlc) = new L1GctJetLeafCard(jlc,jlc % 3,jetSourceCards, m_jetEtCalLut, jfType);
  }

  //Link jet leaf cards together
  vector<L1GctJetLeafCard*> neighbours(2);
  for (int jlc=0 ; jlc<N_JET_LEAF_CARDS/2; jlc++) {
    // Define local constant for ease of typing
    static const int NL = N_JET_LEAF_CARDS/2;
    int nlc = (jlc+1)%NL;
    int mlc = (jlc+(NL-1))%NL;
    neighbours.at(0) = theJetLeafCards.at(mlc);
    neighbours.at(1) = theJetLeafCards.at(nlc);
    theJetLeafCards.at(jlc)->setNeighbourLeafCards(neighbours);
    neighbours.at(0) = theJetLeafCards.at(NL+mlc);
    neighbours.at(1) = theJetLeafCards.at(NL+nlc);
    theJetLeafCards.at(NL+jlc)->setNeighbourLeafCards(neighbours);
  }

  // EM leaf cards  
  vector<L1GctSourceCard*> emSourceCards(9);

  for (int i=0; i<N_EM_LEAF_CARDS; i++) {
    for (int j=0; j<9; j++) {
      emSourceCards.at(j)=theSourceCards.at((i*9+j)*3);
    }
    theEmLeafCards.at(i) = new L1GctEmLeafCard(i,emSourceCards);
  }

   // Wheel Fpgas
   vector<L1GctJetLeafCard*> wheelJetLeafCards(3);
   vector<L1GctJetLeafCard*> wheelEnergyLeafCards(3);

   for (int i=0; i<N_WHEEL_CARDS; i++) {
     for (int j=0; j<3; j++) {
       wheelJetLeafCards.at(j)=theJetLeafCards.at(i*3+j);
       wheelEnergyLeafCards.at(j)=theJetLeafCards.at(i*3+j);
     }
     theWheelJetFpgas.at(i)    = new L1GctWheelJetFpga   (i,wheelJetLeafCards);
     theWheelEnergyFpgas.at(i) = new L1GctWheelEnergyFpga(i,wheelEnergyLeafCards);
   }
  
   // Jet Final Stage  
   theJetFinalStage = new L1GctJetFinalStage(theWheelJetFpgas);

  // Electron Final Sort
   theIsoEmFinalStage = new L1GctElectronFinalSort(true,theEmLeafCards.at(0), theEmLeafCards.at(1));
   theNonIsoEmFinalStage = new L1GctElectronFinalSort(false,theEmLeafCards.at(0), theEmLeafCards.at(1));  

  // Global Energy Algos
  theEnergyFinalStage = new L1GctGlobalEnergyAlgos(theWheelEnergyFpgas, theWheelJetFpgas);

}

void L1GlobalCaloTrigger::setupJetCounterLuts() {

  // Initialise look-up tables for Minus and Plus wheels
  for (unsigned wheel=0; wheel<2; ++wheel) {
    unsigned j=0;
    // Setup the first counters in the list for some arbitrary conditions
    // Energy cut
    theWheelJetFpgas.at(wheel)->getJetCounter(j)->setLut(L1GctJetCounterLut::minRank, 5);
    j++;

    // Eta cuts
    theWheelJetFpgas.at(wheel)->getJetCounter(j)->setLut(L1GctJetCounterLut::centralEta, 5);
    j++;

    // Some one-sided eta cuts
    if (wheel==0) {
      theWheelJetFpgas.at(wheel)->getJetCounter(j)->setLut(L1GctJetCounterLut::forwardEta, 6);
    }
    j++;

    if (wheel==1) {
      theWheelJetFpgas.at(wheel)->getJetCounter(j)->setLut(L1GctJetCounterLut::forwardEta, 6);
    }
    j++;

  }
}
