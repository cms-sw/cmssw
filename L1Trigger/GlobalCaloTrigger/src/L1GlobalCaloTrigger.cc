#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounterLut.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>

using std::cout;
using std::endl;
using std::vector;

//DEFINE STATICS
const int L1GlobalCaloTrigger::N_JET_LEAF_CARDS = 6;
const int L1GlobalCaloTrigger::N_EM_LEAF_CARDS = 2;
const int L1GlobalCaloTrigger::N_WHEEL_CARDS = 2;

const unsigned int L1GlobalCaloTrigger::N_JET_COUNTERS_PER_WHEEL = L1GctWheelJetFpga::N_JET_COUNTERS;


// constructor
L1GlobalCaloTrigger::L1GlobalCaloTrigger(L1GctJetLeafCard::jetFinderType jfType) :
  theJetLeafCards(N_JET_LEAF_CARDS),
  theJetFinders(N_JET_LEAF_CARDS*3),
  theEmLeafCards(N_EM_LEAF_CARDS),
  theIsoElectronSorters(N_EM_LEAF_CARDS*2),
  theNonIsoElectronSorters(N_EM_LEAF_CARDS*2),
  theWheelJetFpgas(N_WHEEL_CARDS),
  theWheelEnergyFpgas(N_WHEEL_CARDS),
  m_jetEtCalLut(0)
{

  // construct hardware
  build(jfType);
  
  // jet counter LUT
  setupJetCounterLuts();

}

/// GCT Destructor
L1GlobalCaloTrigger::~L1GlobalCaloTrigger()
{
  // Delete the components of the GCT that we made in build()
  // (But not the LUTs, since these don't belong to us)

  if (theNonIsoEmFinalStage != 0) delete theNonIsoEmFinalStage;
  
  if (theIsoEmFinalStage != 0) delete theIsoEmFinalStage;
  
  if (theEnergyFinalStage != 0) delete theEnergyFinalStage;	
  
  if (theJetFinalStage != 0) delete theJetFinalStage;			
  
  for (unsigned i=0; i<theWheelEnergyFpgas.size(); ++i) { 
    if (theWheelEnergyFpgas.at(i) != 0) delete theWheelEnergyFpgas.at(i); }
  theWheelEnergyFpgas.clear();
  
  for (unsigned i=0; i<theWheelJetFpgas.size(); ++i) { 
    if (theWheelJetFpgas.at(i) != 0) delete theWheelJetFpgas.at(i); }
  theWheelJetFpgas.clear();		

  for (unsigned i=0; i<theEmLeafCards.size(); ++i) { 
    if (theEmLeafCards.at(i) != 0) delete theEmLeafCards.at(i); }
  theEmLeafCards.clear();

  for (unsigned i=0; i<theJetLeafCards.size(); ++i) { 
    if (theJetLeafCards.at(i) != 0) delete theJetLeafCards.at(i); }
  theJetLeafCards.clear();
  
}

void L1GlobalCaloTrigger::reset() {

  // EM Leaf Card
  for (int i=0; i<N_EM_LEAF_CARDS; i++) {
    theEmLeafCards.at(i)->reset();
  }

  // Jet Leaf cards
  for (int i=0; i<N_JET_LEAF_CARDS; i++) {
    theJetLeafCards.at(i)->reset();
  }

  // Jet Finders
  for (int i=0; i<N_JET_LEAF_CARDS*3; i++) {
    theJetFinders.at(i)->reset();
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

  // Shouldn't get here unless the setup has been completed
  assert (setupOk());

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

/// setup the Jet Finder parameters
void L1GlobalCaloTrigger::setJetFinderParams(const L1GctJetFinderParams* jfpars) {

  // Some parameters not (yet?) implemented
  assert (jfpars->CENTRAL_FORWARD_ETA_BOUNDARY==7);
  assert (jfpars->CENTRAL_JET_SEED==jfpars->TAU_JET_SEED);

  m_jetFinderParams = jfpars;
  // Need to propagate the new parameters to all the JetFinders
  for (int i=0; i<N_JET_LEAF_CARDS; i++) {
    theJetLeafCards.at(i)->getJetFinderA()->setJetFinderParams(jfpars);
    theJetLeafCards.at(i)->getJetFinderB()->setJetFinderParams(jfpars);
    theJetLeafCards.at(i)->getJetFinderC()->setJetFinderParams(jfpars);
  }
}

/// setup the Jet Calibration Lut
void L1GlobalCaloTrigger::setJetEtCalibrationLut(const L1GctJetEtCalibrationLut* lut) {

  std::cout << "gct::setJetEtCalibrationLut called, lut is " << lut << std::endl;
  m_jetEtCalLut = lut;
  // Need to propagate the new lut to all the JetFinders
  for (int i=0; i<N_JET_LEAF_CARDS; i++) {
    theJetLeafCards.at(i)->getJetFinderA()->setJetEtCalibrationLut(lut);
    theJetLeafCards.at(i)->getJetFinderB()->setJetEtCalibrationLut(lut);
    theJetLeafCards.at(i)->getJetFinderC()->setJetEtCalibrationLut(lut);
  }
}

void L1GlobalCaloTrigger::setRegion(L1CaloRegion region) 
{
  if (region.bx()==0) {
    unsigned crate = region.rctCrate();
    // Find the relevant jetFinders
    static const unsigned NPHI = L1CaloRegionDetId::N_PHI/2;
    unsigned thisphi = crate % NPHI;
    unsigned nextphi = (crate+1) % NPHI;
    unsigned prevphi = (crate+NPHI-1) % NPHI;

    // Send the region to six jetFinders.
    theJetFinders.at(thisphi)->setInputRegion(region);
    theJetFinders.at(nextphi)->setInputRegion(region);
    theJetFinders.at(prevphi)->setInputRegion(region);
    theJetFinders.at(thisphi+NPHI)->setInputRegion(region);
    theJetFinders.at(nextphi+NPHI)->setInputRegion(region);
    theJetFinders.at(prevphi+NPHI)->setInputRegion(region);
  }
}

void L1GlobalCaloTrigger::setRegion(unsigned et, unsigned ieta, unsigned iphi, bool overFlow, bool fineGrain)
{
  L1CaloRegion temp(et, overFlow, fineGrain, false, false, ieta, iphi);
  setRegion(temp);
}

void L1GlobalCaloTrigger::setIsoEm(L1CaloEmCand em) 
{
  if (em.bx()==0) {
    unsigned crate = em.rctCrate();
    unsigned sorterNo;
    if ((crate%9) < 4) {
      sorterNo = (crate/9)*2;
    } else {
      sorterNo = (crate/9)*2 + 1;
    }
    theIsoElectronSorters.at(sorterNo)->setInputEmCand(em); 
  }
}

void L1GlobalCaloTrigger::setNonIsoEm(L1CaloEmCand em) 
{
  if (em.bx()==0) {
    unsigned crate = em.rctCrate();
    unsigned sorterNo;
    if ((crate%9) < 4) {
      sorterNo = (crate/9)*2;
    } else {
      sorterNo = (crate/9)*2 + 1;
    }
    theNonIsoElectronSorters.at(sorterNo)->setInputEmCand(em); 
  }
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
  cout << "N Jet Leaf Cards " << theJetLeafCards.size() << endl;
  cout << "N Wheel Jet Fpgas " << theWheelJetFpgas.size() << endl;
  cout << "N Wheel Energy Fpgas " << theWheelEnergyFpgas.size() << endl;
  cout << "N Em Leaf Cards " << theEmLeafCards.size() << endl;
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
vector<L1GctJetCand> L1GlobalCaloTrigger::getCentralJets() const {
  return theJetFinalStage->getCentralJets();
}

// forward jet outputs to GT
vector<L1GctJetCand> L1GlobalCaloTrigger::getForwardJets() const { 
  return theJetFinalStage->getForwardJets(); 
}

// tau jet outputs to GT
vector<L1GctJetCand> L1GlobalCaloTrigger::getTauJets() const { 
  return theJetFinalStage->getTauJets(); 
}

// total Et output
L1GctUnsignedInt<12> L1GlobalCaloTrigger::getEtSum() const {
  return theEnergyFinalStage->getEtSum();
}

L1GctUnsignedInt<12> L1GlobalCaloTrigger::getEtHad() const {
  return theEnergyFinalStage->getEtHad();
}

L1GctUnsignedInt<12> L1GlobalCaloTrigger::getEtMiss() const {
  return theEnergyFinalStage->getEtMiss();
}

L1GctUnsignedInt<7> L1GlobalCaloTrigger::getEtMissPhi() const {
  return theEnergyFinalStage->getEtMissPhi();
}

L1GctJetCount<5> L1GlobalCaloTrigger::getJetCount(unsigned jcnum) const {
  return theEnergyFinalStage->getJetCount(jcnum);
}



/* PRIVATE METHODS */

// instantiate hardware/algorithms
void L1GlobalCaloTrigger::build(L1GctJetLeafCard::jetFinderType jfType) {

  // Jet Leaf cards
  for (int jlc=0; jlc<N_JET_LEAF_CARDS; jlc++) {
    theJetLeafCards.at(jlc) = new L1GctJetLeafCard(jlc,jlc % 3, jfType);
    theJetFinders.at( 3*jlc ) = theJetLeafCards.at(jlc)->getJetFinderA();
    theJetFinders.at(3*jlc+1) = theJetLeafCards.at(jlc)->getJetFinderB();
    theJetFinders.at(3*jlc+2) = theJetLeafCards.at(jlc)->getJetFinderC();
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
  for (int i=0; i<N_EM_LEAF_CARDS; i++) {
    theEmLeafCards.at(i) = new L1GctEmLeafCard(i);
    theIsoElectronSorters.at( 2*i ) = theEmLeafCards.at(i)->getIsoElectronSorter0();
    theIsoElectronSorters.at(2*i+1) = theEmLeafCards.at(i)->getIsoElectronSorter1();
    theNonIsoElectronSorters.at( 2*i ) = theEmLeafCards.at(i)->getNonIsoElectronSorter0();
    theNonIsoElectronSorters.at(2*i+1) = theEmLeafCards.at(i)->getNonIsoElectronSorter1();
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
