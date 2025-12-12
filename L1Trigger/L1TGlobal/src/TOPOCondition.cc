/**
 * \class TOPOCondition
 *
 *
 * Description: evaluation of a condition for TOPO anomaly detection algorithm
 *
 * Author: Melissa Quinnan, Lukas Ebeling, Artur Lobanov
 *
 **/

// this class header
#include "L1Trigger/L1TGlobal/interface/CorrCondition.h"

// system include files
#include <iostream>
#include <iomanip>
#include <fstream>

#include <string>
#include <vector>
#include <algorithm>
#include "ap_fixed.h"

// user include files
//   base classes
#include "L1Trigger/L1TGlobal/interface/TOPOTemplate.h"
#include "L1Trigger/L1TGlobal/interface/ConditionEvaluation.h"

#include "L1Trigger/L1TGlobal/interface/MuCondition.h"
#include "L1Trigger/L1TGlobal/interface/TOPOCondition.h"
#include "L1Trigger/L1TGlobal/interface/CaloCondition.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumCondition.h"
#include "L1Trigger/L1TGlobal/interface/MuonTemplate.h"
#include "L1Trigger/L1TGlobal/interface/CaloTemplate.h"
#include "L1Trigger/L1TGlobal/interface/EnergySumTemplate.h"
#include "L1Trigger/L1TGlobal/interface/GlobalScales.h"

#include "DataFormats/L1Trigger/interface/L1Candidate.h"

#include "L1Trigger/L1TGlobal/interface/GlobalBoard.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

l1t::TOPOCondition::TOPOCondition()
    : ConditionEvaluation(), m_gtTOPOTemplate{nullptr}, m_gtGTB{nullptr}, m_model{nullptr} {}

l1t::TOPOCondition::TOPOCondition(const GlobalCondition* topoTemplate, const GlobalBoard* ptrGTB)
    : ConditionEvaluation(),
      m_gtTOPOTemplate(static_cast<const TOPOTemplate*>(topoTemplate)),
      m_gtGTB(ptrGTB),
      m_model_loader{kModelNamePrefix + m_gtTOPOTemplate->modelVersion()} {
  loadModel();
}

// copy constructor
void l1t::TOPOCondition::copy(const l1t::TOPOCondition& cp) {
  m_gtTOPOTemplate = cp.gtTOPOTemplate();
  m_gtGTB = cp.gtGTB();

  m_condMaxNumberObjects = cp.condMaxNumberObjects();
  m_condLastResult = cp.condLastResult();
  m_combinationsInCond = cp.getCombinationsInCond();

  m_verbosity = cp.m_verbosity;

  m_model_loader.reset(cp.model_loader().model_name());
  loadModel();
}

l1t::TOPOCondition::TOPOCondition(const l1t::TOPOCondition& cp) : ConditionEvaluation() { copy(cp); }

// destructor
l1t::TOPOCondition::~TOPOCondition() {
  // empty
}

// equal operator
l1t::TOPOCondition& l1t::TOPOCondition::operator=(const l1t::TOPOCondition& cp) {
  copy(cp);
  return *this;
}

// methods
void l1t::TOPOCondition::setGtTOPOTemplate(const TOPOTemplate* caloTempl) { m_gtTOPOTemplate = caloTempl; }

///   set the pointer to uGT GlobalBoard
void l1t::TOPOCondition::setuGtB(const GlobalBoard* ptrGTB) { m_gtGTB = ptrGTB; }

/// set score for score saving
void l1t::TOPOCondition::setScore(const float scoreval) const { m_savedscore = scoreval; }

void l1t::TOPOCondition::loadModel() {
  try {
    m_model = m_model_loader.load_model();
  } catch (std::runtime_error& e) {
    throw cms::Exception("ModelError") << " ERROR: failed to load TOPO model version \"" << m_model_loader.model_name()
                                       << "\". Model version not found in cms-hls4ml externals.";
  }
}

const bool l1t::TOPOCondition::evaluateCondition(const int bxEval) const {
  if (m_model == nullptr) {
    throw cms::Exception("ModelError") << " ERROR: no model was loaded for TOPO model version \""
                                       << m_model_loader.model_name() << "\".";
  }

  bool condResult = false;
  int useBx = bxEval + m_gtTOPOTemplate->condRelativeBx();

  // //pointers to objects
  const BXVector<const l1t::Muon*>* candMuVec = m_gtGTB->getCandL1Mu();
  const BXVector<const l1t::L1Candidate*>* candJetVec = m_gtGTB->getCandL1Jet();
  const BXVector<const l1t::L1Candidate*>* candEGVec = m_gtGTB->getCandL1EG();
  const BXVector<const l1t::EtSum*>* candEtSumVec = m_gtGTB->getCandL1EtSum();

  const int NMuons = 2;
  const int NJets = 4;
  const int NEgammas = 0;
  const int NEtSums = 1;

  //number of indices in vector is #objects * 3 for et, eta, phi
  const int MuVecSize = NMuons * 4;      //with hwquality
  const int JVecSize = NJets * 3;        //so 12
  const int EGVecSize = NEgammas * 3;    //so 0
  const int EtSumVecSize = NEtSums * 1;  //no eta/phi

  //total # inputs in vector
  const int NInputs = MuVecSize + JVecSize + EGVecSize + EtSumVecSize;  //so 21

  //types of inputs and outputs modified for topo
  typedef ap_fixed<23, 23> inputtype;
  typedef ap_fixed<16, 6> losstype;

  //input vector declaration, will fill later
  inputtype ModelInput[NInputs];
  inputtype fillzero = 0.0;

  //initializing vector by type for my sanity
  double MuInput[MuVecSize];
  double JetInput[JVecSize];
  double EgammaInput[EGVecSize];
  double EtSumInput[EtSumVecSize];

  //declare result vectors +score
  losstype loss;
  float score = -1.0;

  //check number of input objects we actually have (muons, jets etc)
  int NCandMu = candMuVec->size(useBx);
  int NCandJet = candJetVec->size(useBx);
  int NCandEG = candEGVec->size(useBx);
  int NCandEtSum = candEtSumVec->size(useBx);

  //initialize arrays to zero (std::fill(first, last, value);)
  std::fill(EtSumInput, EtSumInput + EtSumVecSize, 0.0);
  std::fill(MuInput, MuInput + MuVecSize, 0.0);
  std::fill(JetInput, JetInput + JVecSize, 0.0);
  std::fill(EgammaInput, EgammaInput + EGVecSize, 0.0);
  std::fill(ModelInput, ModelInput + NInputs, fillzero);

  //then fill the object vectors
  if (NCandEtSum > 0) {  //check if not empty
    for (int iEtSum = 0; iEtSum < NCandEtSum; iEtSum++) {
      if ((candEtSumVec->at(useBx, iEtSum))->getType() == 1) {
        EtSumInput[0] = (candEtSumVec->at(useBx, iEtSum))->hwPt();
      }
    }
  }

  //next egammas
  if (NCandEG > 0) {  //check if not empty
    for (int iEG = 0; iEG < NCandEG; iEG++) {
      if (iEG < NEgammas) {                                                 //stop if fill the Nobjects we need
        EgammaInput[0 + (3 * iEG)] = (candEGVec->at(useBx, iEG))->hwPt();   //index 0,3,6,9
        EgammaInput[1 + (3 * iEG)] = (candEGVec->at(useBx, iEG))->hwEta();  //index 1,4,7,10
        EgammaInput[2 + (3 * iEG)] = (candEGVec->at(useBx, iEG))->hwPhi();  //index 2,5,8,11
      }
    }
  }

  //next muons
  if (NCandMu > 0) {  //check if not empty
    for (int iMu = 0; iMu < NCandMu; iMu++) {
      if (iMu < NMuons) {                                               //stop if fill the Nobjects we need
        MuInput[0 + (4 * iMu)] = (candMuVec->at(useBx, iMu))->hwPt();   //index 0,4,8,12
        MuInput[1 + (4 * iMu)] = (candMuVec->at(useBx, iMu))->hwEtaAtVtx();  //index 1,5,9,13
        MuInput[2 + (4 * iMu)] = (candMuVec->at(useBx, iMu))->hwPhiAtVtx();  //index 2,6,10,14
        MuInput[3 + (4 * iMu)] = (candMuVec->at(useBx, iMu))->hwQual(); //index 3,7,11,15
      }
    }
  }

  //next jets
  if (NCandJet > 0) {  //check if not empty
    for (int iJet = 0; iJet < NCandJet; iJet++) {
      if (iJet < NJets) {                                                   //stop if fill the Nobjects we need
        JetInput[0 + (3 * iJet)] = (candJetVec->at(useBx, iJet))->hwPt();   //index 0,3,6,9
        JetInput[1 + (3 * iJet)] = (candJetVec->at(useBx, iJet))->hwEta();  //index 1,4,7,10
        JetInput[2 + (3 * iJet)] = (candJetVec->at(useBx, iJet))->hwPhi();  //index 2,5,8,11
      }
    }
  }

  //now put it all together-> EtSum+EGamma+Muon+Jet into ModelInput
  int index = 0;
  for (int idET = 0; idET < EtSumVecSize; idET++) {
    ModelInput[index++] = EtSumInput[idET];
  }
  for (int idEG = 0; idEG < EGVecSize; idEG++) {
    ModelInput[index++] = EgammaInput[idEG];
  }
  for (int idMu = 0; idMu < MuVecSize; idMu++) {
    ModelInput[index++] = MuInput[idMu];
  }
  for (int idJ = 0; idJ < JVecSize; idJ++) {
    ModelInput[index++] = JetInput[idJ];
  }

  //now run the inference
  m_model->prepare_input(ModelInput);  //scaling internal here
  m_model->predict();
  m_model->read_result(&loss);  //store result as loss variable
  score = ((loss).to_float() * 1023);
  setScore(score);

  //number of objects/thrsholds to check
  int iCondition = 0;  // number of conditions: there is only one
  int nObjInCond = m_gtTOPOTemplate->nrObjects();

  if (iCondition >= nObjInCond || iCondition < 0) {
    return false;
  }

  const TOPOTemplate::ObjectParameter objPar = (*(m_gtTOPOTemplate->objectParameter()))[iCondition];

  // condGEqVal indicates the operator used for the condition (>=, =): true for >=
  bool condGEqVal = m_gtTOPOTemplate->condGEq();
  bool passCondition = false;

  passCondition = checkCut(objPar.minTOPOThreshold, score, condGEqVal);

  condResult |= passCondition;  //condresult true if passCondition true else it is false

  //return result
  return condResult;
}

void l1t::TOPOCondition::print(std::ostream& myCout) const {
  myCout << "Dummy Print for TOPOCondition" << std::endl;
  m_gtTOPOTemplate->print(myCout);

  ConditionEvaluation::print(myCout);
}
