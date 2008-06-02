#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include "CondFormats/L1TObjects/interface/L1GctJetFinderParams.h"
#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetEtCalibrationLut.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmLeafCard.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinderBase.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelJetFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctWheelEnergyFpga.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetFinalStage.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctGlobalEnergyAlgos.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctElectronFinalSort.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetCounter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <cassert>

using std::vector;

//DEFINE STATICS
const int L1GlobalCaloTrigger::N_JET_LEAF_CARDS = 6;
const int L1GlobalCaloTrigger::N_EM_LEAF_CARDS = 2;
const int L1GlobalCaloTrigger::N_WHEEL_CARDS = 2;

// constructor
L1GlobalCaloTrigger::L1GlobalCaloTrigger(const L1GctJetLeafCard::jetFinderType jfType) :
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

  m_jetEtCalLut = lut;
  // Need to propagate the new lut to all the JetFinders
  for (int i=0; i<N_JET_LEAF_CARDS; i++) {
    theJetLeafCards.at(i)->getJetFinderA()->setJetEtCalibrationLut(lut);
    theJetLeafCards.at(i)->getJetFinderB()->setJetEtCalibrationLut(lut);
    theJetLeafCards.at(i)->getJetFinderC()->setJetEtCalibrationLut(lut);
  }
}

void L1GlobalCaloTrigger::setupJetCounterLuts(const L1GctJetCounterSetup* jcPosPars,
                                              const L1GctJetCounterSetup* jcNegPars) {

  // Initialise look-up tables for Plus and Minus wheels
  for (unsigned j=0; j<jcPosPars->numberOfJetCounters(); ++j) {
    theWheelJetFpgas.at(0)->getJetCounter(j)->setLut(
                 jcPosPars->getCutsForJetCounter(j) );
  }
  for (unsigned j=0; j<jcNegPars->numberOfJetCounters(); ++j) {
    theWheelJetFpgas.at(1)->getJetCounter(j)->setLut(
                 jcNegPars->getCutsForJetCounter(j) );
  }
}

void L1GlobalCaloTrigger::setRegion(const L1CaloRegion& region) 
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

void L1GlobalCaloTrigger::setRegion(const unsigned et, const unsigned ieta, const unsigned iphi,
                                    const bool overFlow, const bool fineGrain)
{
  L1CaloRegion temp(et, overFlow, fineGrain, false, false, ieta, iphi);
  setRegion(temp);
}

void L1GlobalCaloTrigger::setIsoEm(const L1CaloEmCand& em) 
{
  if (em.bx()==0) {
    theIsoElectronSorters.at(sorterNo(em))->setInputEmCand(em); 
  }
}

void L1GlobalCaloTrigger::setNonIsoEm(const L1CaloEmCand& em) 
{
  if (em.bx()==0) {
    theNonIsoElectronSorters.at(sorterNo(em))->setInputEmCand(em); 
  }
}

void L1GlobalCaloTrigger::fillRegions(const vector<L1CaloRegion>& rgn)
{
  for (uint i=0; i<rgn.size(); i++){
    setRegion(rgn.at(i));
  }
}

void L1GlobalCaloTrigger::fillEmCands(const vector<L1CaloEmCand>& em)
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

  using edm::LogInfo;
  using std::endl;

  LogInfo("L1GlobalCaloTrigger") << "=== Global Calo Trigger ===" << endl;
  LogInfo("L1GlobalCaloTrigger") << "=== START DEBUG OUTPUT  ===" << endl;

  LogInfo("L1GlobalCaloTrigger") << endl;
  LogInfo("L1GlobalCaloTrigger") << "N Jet Leaf Cards " << theJetLeafCards.size() << endl;
  LogInfo("L1GlobalCaloTrigger") << "N Wheel Jet Fpgas " << theWheelJetFpgas.size() << endl;
  LogInfo("L1GlobalCaloTrigger") << "N Wheel Energy Fpgas " << theWheelEnergyFpgas.size() << endl;
  LogInfo("L1GlobalCaloTrigger") << "N Em Leaf Cards " << theEmLeafCards.size() << endl;
  LogInfo("L1GlobalCaloTrigger") << endl;

  for (unsigned i=0; i<theJetLeafCards.size(); i++) {
    LogInfo("L1GlobalCaloTrigger") << "Jet Leaf Card " << i << " : " << theJetLeafCards.at(i) << endl;
    LogInfo("L1GlobalCaloTrigger") << (*theJetLeafCards.at(i));
  }
  LogInfo("L1GlobalCaloTrigger") << endl;

  for (unsigned i=0; i<theWheelJetFpgas.size(); i++) {
    LogInfo("L1GlobalCaloTrigger") << "Wheel Jet FPGA " << i << " : " << theWheelJetFpgas.at(i) << endl; 
    LogInfo("L1GlobalCaloTrigger") << (*theWheelJetFpgas.at(i));
  }
  LogInfo("L1GlobalCaloTrigger") << endl;

  for (unsigned i=0; i<theWheelEnergyFpgas.size(); i++) {
    LogInfo("L1GlobalCaloTrigger") << "Wheel Energy FPGA " << i <<" : " << theWheelEnergyFpgas.at(i) << endl; 
    LogInfo("L1GlobalCaloTrigger") << (*theWheelEnergyFpgas.at(i));
  }
  LogInfo("L1GlobalCaloTrigger") << endl;

  LogInfo("L1GlobalCaloTrigger") << (*theJetFinalStage);
  LogInfo("L1GlobalCaloTrigger") << endl;

  LogInfo("L1GlobalCaloTrigger") << (*theEnergyFinalStage);
  LogInfo("L1GlobalCaloTrigger") << endl;

  for (unsigned i=0; i<theEmLeafCards.size(); i++) {
    LogInfo("L1GlobalCaloTrigger") << ( (i==0) ? "Positive eta " : "Negative eta " ); 
    LogInfo("L1GlobalCaloTrigger") << "EM Leaf Card " << i << " : " << theEmLeafCards.at(i) << endl;
    LogInfo("L1GlobalCaloTrigger") << (*theEmLeafCards.at(i));
  }
  LogInfo("L1GlobalCaloTrigger") << endl;

  LogInfo("L1GlobalCaloTrigger") << (*theIsoEmFinalStage);
  LogInfo("L1GlobalCaloTrigger") << endl;

  LogInfo("L1GlobalCaloTrigger") << (*theNonIsoEmFinalStage);

  LogInfo("L1GlobalCaloTrigger") << "=== Global Calo Trigger ===" << endl;
  LogInfo("L1GlobalCaloTrigger") << "===  END DEBUG OUTPUT   ===" << endl;
 
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
L1GlobalCaloTrigger::etTotalType   L1GlobalCaloTrigger::getEtSum() const {
  return theEnergyFinalStage->getEtSum();
}

L1GlobalCaloTrigger::etHadType     L1GlobalCaloTrigger::getEtHad() const {
  return theEnergyFinalStage->getEtHad();
}

L1GlobalCaloTrigger::etMissType    L1GlobalCaloTrigger::getEtMiss() const {
  return theEnergyFinalStage->getEtMiss();
}

L1GlobalCaloTrigger::etMissPhiType L1GlobalCaloTrigger::getEtMissPhi() const {
  return theEnergyFinalStage->getEtMissPhi();
}

L1GctJetCount<5> L1GlobalCaloTrigger::getJetCount(unsigned jcnum) const {
  return theEnergyFinalStage->getJetCount(jcnum);
}

std::vector<unsigned> L1GlobalCaloTrigger::getJetCountValues() const {
  return theEnergyFinalStage->getJetCountValues();
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
  // Card 0 is positive eta, card 1 is negative eta
  for (int i=0; i<N_EM_LEAF_CARDS; i++) {
    theEmLeafCards.at(i) = new L1GctEmLeafCard(i);
    theIsoElectronSorters.at( 2*i ) = theEmLeafCards.at(i)->getIsoElectronSorterU1();
    theIsoElectronSorters.at(2*i+1) = theEmLeafCards.at(i)->getIsoElectronSorterU2();
    theNonIsoElectronSorters.at( 2*i ) = theEmLeafCards.at(i)->getNonIsoElectronSorterU1();
    theNonIsoElectronSorters.at(2*i+1) = theEmLeafCards.at(i)->getNonIsoElectronSorterU2();
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

/// ordering of the electron sorters to give the correct
/// priority to the candidates in the final sort 
/// The priority ordering is:
///    crates  4 - 8 : priority 0 (highest)
///    crates  0 - 3 : priority 1
///    crates 13 -17 : priority 2
///    crates  9 -12 : priority 3 (lowest)
unsigned L1GlobalCaloTrigger::sorterNo(const L1CaloEmCand& em) const {
  unsigned crate = em.rctCrate();
  assert (crate<18);
  unsigned result = ( ((crate%9) < 4) ? 1 : 0 );
  if (crate>=9) result += 2;
  return result;
}

