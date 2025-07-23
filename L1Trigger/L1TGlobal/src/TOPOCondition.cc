/**
 * \class TOPOCondition
 *
 *
 * Description: evaluation of a condition for TOPO anomaly detection algorithm
 *
 * Author: Melissa Quinnan
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

namespace {
  //template function for reading results
  template <typename ResultType, typename LossType>
  LossType readResult(hls4mlEmulator::Model& model) {
    std::pair<ResultType, LossType> ADModelResult;  //model outputs a pair of the (result vector, loss)
    model.read_result(&ADModelResult);
    return ADModelResult.second;
  }
}  // namespace

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
    throw cms::Exception("ModelError") << " ERROR: failed to load TOPO model version \""
                                       << m_model_loader.model_name()
                                       << "\". Model version not found in cms-hls4ml externals.";
  }
}

const bool l1t::TOPOCondition::evaluateCondition(const int bxEval) const {
  if (m_model == nullptr) {
    throw cms::Exception("ModelError") << " ERROR: no model was loaded for TOPO model version \""
                                       << m_model_loader.model_name() << "\".";
  }

  // overwrite model for now using local path
  std::cout << "Overwriting topo model loading..." << m_model_loader.model_name() <<std::endl;
  std::string TOPOmodelversion = "/eos/user/l/lebeling/AXO_v5/test_TOPO/topo_v1";
  hls4mlEmulator::ModelLoader loader(TOPOmodelversion);
  std::shared_ptr<hls4mlEmulator::Model> m_model;
  m_model = loader.load_model();
  std::cout << "loading model... " << TOPOmodelversion << std::endl;
  
  std::cout << "#### evaluate topo condition ####" << std::endl;

  bool condResult = false;
  int useBx = bxEval + m_gtTOPOTemplate->condRelativeBx();

  // //pointers to objects
  const BXVector<const l1t::Muon*>* candMuVec = m_gtGTB->getCandL1Mu();
  const BXVector<const l1t::L1Candidate*>* candJetVec = m_gtGTB->getCandL1Jet();
  const BXVector<const l1t::L1Candidate*>* candEGVec = m_gtGTB->getCandL1EG();
  const BXVector<const l1t::EtSum*>* candEtSumVec = m_gtGTB->getCandL1EtSum();

  // const int NMuons = 4;
  // const int NJets = 10;
  // const int NEgammas = 4;
  // const int NEtSums = 1;  
  
  const int NMuons = 2;
  const int NJets = 4;
  const int NEgammas = 0;
  const int NEtSums = 1;

  //number of indices in vector is #objects * 3 for et, eta, phi
  const int MuVecSize = NMuons * 3;      //so 6
  const int JVecSize = NJets * 3;        //so 12
  const int EGVecSize = NEgammas * 3;    //so 0
  const int EtSumVecSize = NEtSums * 2;    //no eta

  //total # inputs in vector
  const int NInputs = MuVecSize + JVecSize + EGVecSize + EtSumVecSize;  //so 20

  //types of inputs and outputs modified for topo
  typedef ap_fixed<16, 6> inputtype;
  typedef ap_fixed<16, 6> losstype;

  //define zero
  inputtype fillzero = 0.0;

  //AD vector declaration, will fill later
  inputtype ADModelInput[NInputs] = {};

  //initializing vector by type for my sanity
  inputtype MuInput[MuVecSize];
  inputtype JetInput[JVecSize];
  inputtype EgammaInput[EGVecSize];
  inputtype EtSumInput[EtSumVecSize];

  //declare result vectors +score
  // resulttype result;
  losstype loss;
  // pairtype ADModelResult;  //model outputs a pair of the (result vector, loss)
  float score = -1.0;  //not sure what the best default is hm??

  //check number of input objects we actually have (muons, jets etc)
  int NCandMu = candMuVec->size(useBx);
  int NCandJet = candJetVec->size(useBx);
  int NCandEG = candEGVec->size(useBx);
  int NCandEtSum = candEtSumVec->size(useBx);

  //initialize arrays to zero (std::fill(first, last, value);)
  std::fill(EtSumInput, EtSumInput + EtSumVecSize, fillzero);
  std::fill(MuInput, MuInput + MuVecSize, fillzero);
  std::fill(JetInput, JetInput + JVecSize, fillzero);
  std::fill(EgammaInput, EgammaInput + EGVecSize, fillzero);
  std::fill(ADModelInput, ADModelInput + NInputs, fillzero);

  //then fill the object vectors
  //NOTE assume candidates are already sorted by pt
  //loop over EtSums first, easy because there is max 1 of them
  if (NCandEtSum > 0) {  //check if not empty
    for (int iEtSum = 0; iEtSum < NCandEtSum; iEtSum++) {
      if ((candEtSumVec->at(useBx, iEtSum))->getType() == l1t::EtSum::EtSumType::kMissingEt) {
        EtSumInput[0] = (candEtSumVec->at(useBx, iEtSum))->hwPt();
        EtSumInput[1] = (candEtSumVec->at(useBx, iEtSum))->hwPhi();
      }
    }
  }

  //next egammas
  if (NCandEG > 0) {  //check if not empty
    for (int iEG = 0; iEG < NCandEG; iEG++) {
      if (iEG < NEgammas) {  //stop if fill the Nobjects we need
        EgammaInput[0 + (3 * iEG)] = (candEGVec->at(useBx, iEG))->hwPt();   //index 0,3,6,9
        EgammaInput[1 + (3 * iEG)] = (candEGVec->at(useBx, iEG))->hwEta();  //index 1,4,7,10
        EgammaInput[2 + (3 * iEG)] = (candEGVec->at(useBx, iEG))->hwPhi();  //index 2,5,8,11
      }
    }
  }

  //next muons
  if (NCandMu > 0) {  //check if not empty
    for (int iMu = 0; iMu < NCandMu; iMu++) {
      if (iMu < NMuons) {  //stop if fill the Nobjects we need
        MuInput[0 + (3 * iMu)] = (candMuVec->at(useBx, iMu))->hwPt();        //index 0,3,6,9
        MuInput[1 + (3 * iMu)] = (candMuVec->at(useBx, iMu))->hwEta();  //index 1,4,7,10
        MuInput[2 + (3 * iMu)] = (candMuVec->at(useBx, iMu))->hwPhi();  //index 2,5,8,11
      }
    }
  }

  //next jets
  if (NCandJet > 0) {  //check if not empty
    for (int iJet = 0; iJet < NCandJet; iJet++) {
      if (iJet < NJets) {  //stop if fill the Nobjects we need
        JetInput[0 + (3 * iJet)] = (candJetVec->at(useBx, iJet))->hwPt(); //index 0,3,6,9
        JetInput[1 + (3 * iJet)] = (candJetVec->at(useBx, iJet))->hwEta();  //index 1,4,7,10
        JetInput[2 + (3 * iJet)] = (candJetVec->at(useBx, iJet))->hwPhi();  //index 2,5,8,11
      }
    }
  }

  //now put it all together-> EtSum+EGamma+Muon+Jet into ADModelInput
  int index = 0;
  for (int idET = 0; idET < EtSumVecSize; idET++) {
    ADModelInput[index++] = EtSumInput[idET];
  }
  for (int idEG = 0; idEG < EGVecSize; idEG++) {
    ADModelInput[index++] = EgammaInput[idEG];
  }
  for (int idMu = 0; idMu < MuVecSize; idMu++) {
    ADModelInput[index++] = MuInput[idMu];
  }
  for (int idJ = 0; idJ < JVecSize; idJ++) {
    ADModelInput[index++] = JetInput[idJ];
  }

  //now run the inference
  m_model->prepare_input(ADModelInput);  //scaling internal here
  m_model->predict();
  m_model->read_result(&loss); //store result as loss variable
    
  //CHECK: I'm not sure if topo needs this or not
  score = ((loss).to_float()) * 16.0;  //scaling to match threshold
  //save score to class variable in case score saving needed
  setScore(score);

  // Write ADModelInput to text file (append mode)
  std::ofstream inputFile("features.txt", std::ios::app);
  if (inputFile.is_open()) {
    for (int i = 0; i < NInputs; i++) {
      inputFile << ADModelInput[i].to_float();
      if (i < NInputs - 1) {
        inputFile << ", ";
      }
    }
    inputFile << std::endl;
    inputFile.close();
  } else {
    std::cout << "Error: Could not open ADModelInput.txt for writing" << std::endl;
  }
  
  // Write score to text file (append mode)
  std::ofstream scoreFile("scores.txt", std::ios::app);
  if (scoreFile.is_open()) {
    scoreFile << score << std::endl;
    scoreFile.close();
  } else {
    std::cout << "Error: Could not open scores.txt for writing" << std::endl;
  }

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
  
  std::cout << "score (loss*16) :" << score << std::endl;
  
  //return result
  return condResult;
}

void l1t::TOPOCondition::print(std::ostream& myCout) const {
  myCout << "Dummy Print for TOPOCondition" << std::endl;
  m_gtTOPOTemplate->print(myCout);

  ConditionEvaluation::print(myCout);
}
