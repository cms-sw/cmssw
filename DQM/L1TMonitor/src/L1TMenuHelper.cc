
/*
 * \file L1TMenuHelper.cc
 *
 * $Date: 2011/04/06 16:49:34 $
 * $Revision: 1.1 $
 * \author J. Pela, P. Musella
 *
*/

#include "TString.h"

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

//#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"
#include "HLTrigger/HLTanalyzers/test/RateEff/L1GtLogicParser.h"

// L1Gt - Trigger Menu
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// L1Gt - Prescales
#include "CondFormats/L1TObjects/interface/L1GtPrescaleFactors.h"
#include "CondFormats/DataRecord/interface/L1GtPrescaleFactorsAlgoTrigRcd.h"

// L1Gt - Masks
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"



#include "DQM/L1TMonitor/interface/L1TMenuHelper.h"

using namespace edm;
using namespace std;

//-------------------------------------------------------------------------------------
//
//-------------------------------------------------------------------------------------
L1TMenuHelper::L1TMenuHelper(const edm::EventSetup& iSetup){

  ESHandle<L1GtTriggerMenu>     menuRcd;
  ESHandle<L1GtPrescaleFactors> l1GtPfAlgo;

  iSetup.get<L1GtTriggerMenuRcd>()            .get(menuRcd);
  iSetup.get<L1GtPrescaleFactorsAlgoTrigRcd>().get(l1GtPfAlgo);

  const L1GtPrescaleFactors* m_l1GtPfAlgo = l1GtPfAlgo.product();

  m_l1GtMenu                = menuRcd   .product();                 // Getting the menu
  m_prescaleFactorsAlgoTrig = &(m_l1GtPfAlgo->gtPrescaleFactors()); // Retriving the list of prescale sets

  myUtils.retrieveL1EventSetup(iSetup);

}


//-------------------------------------------------------------------------------------
//
//-------------------------------------------------------------------------------------
L1TMenuHelper::~L1TMenuHelper(){}                      

//-------------------------------------------------------------------------------------
// Method: fetLUSOTrigger
//   * Get Lowest Unprescaled Single Object Triggers and Energy Sums
//-------------------------------------------------------------------------------------
map<string,string> L1TMenuHelper::getLUSOTrigger(map<string,bool> iCategories, int IndexRefPrescaleFactors){

  map<string,string> out;

  // Getting information from the menu
  const AlgorithmMap                            *theAlgoMap           = &m_l1GtMenu->gtAlgorithmAliasMap();
  const vector< vector<L1GtMuonTemplate> >      *vMuonConditions      = &m_l1GtMenu->vecMuonTemplate();
  const vector< vector<L1GtCaloTemplate> >      *vCaloConditions      = &m_l1GtMenu->vecCaloTemplate();
  const vector< vector<L1GtEnergySumTemplate> > *vEnergySumConditions = &m_l1GtMenu->vecEnergySumTemplate();

  // Getting reference prescales
  const vector<int>& refPrescaleFactors = (*m_prescaleFactorsAlgoTrig).at(IndexRefPrescaleFactors); 

  AlgorithmMap MyAlgos;

  map<string,SingleObjectCondition> myConditions;
  vector<SingleObjectTrigger> vTrigMu;
  vector<SingleObjectTrigger> vTrigEG ;  
  vector<SingleObjectTrigger> vTrigIsoEG;
  vector<SingleObjectTrigger> vTrigJet  ;
  vector<SingleObjectTrigger> vTrigCenJet;
  vector<SingleObjectTrigger> vTrigForJet;
  vector<SingleObjectTrigger> vTrigTauJet;
  vector<SingleObjectTrigger> vTrigETM;   
  vector<SingleObjectTrigger> vTrigETT;   
  vector<SingleObjectTrigger> vTrigHTT;   
  vector<SingleObjectTrigger> vTrigHTM; 

  for(unsigned int a=0 ; a<vMuonConditions->size() ; a++){
    for(unsigned int b=0 ; b<(*vMuonConditions)[a].size() ; b++){

      const L1GtMuonTemplate* MuonCondition = &(*vMuonConditions)[a][b];

      // Selecting conditions that require single objects
      if(MuonCondition->condType() == Type1s && MuonCondition->nrObjects() == 1){

        SingleObjectCondition tCondition;
 
        tCondition.name              = MuonCondition->condName();
        tCondition.conditionCategory = MuonCondition->condCategory();
        tCondition.conditionType     = MuonCondition->condType();
        tCondition.object            = MuonCondition->objectType()[0];
        tCondition.threshold         = (*MuonCondition->objectParameter())[0].ptHighThreshold;
        tCondition.quality           = (*MuonCondition->objectParameter())[0].qualityRange;
        tCondition.etaRange          = (*MuonCondition->objectParameter())[0].etaRange;

        myConditions[MuonCondition->condName()] = tCondition;

      }
    }
  }

  for(unsigned int a=0 ; a<vCaloConditions->size() ; a++){
    for(unsigned int b=0 ; b<(*vCaloConditions)[a].size() ; b++){

      const L1GtCaloTemplate* CaloCondition = &(*vCaloConditions)[a][b];

      if(CaloCondition->condType() == Type1s && CaloCondition->nrObjects() == 1){

        SingleObjectCondition tCondition;
 
        tCondition.name              = CaloCondition->condName();
        tCondition.conditionCategory = CaloCondition->condCategory();
        tCondition.conditionType     = CaloCondition->condType();
        tCondition.object            = CaloCondition->objectType()[0];
        tCondition.threshold         = (*CaloCondition->objectParameter())[0].etThreshold;
        tCondition.quality           = 0;
        tCondition.etaRange          = 0;

        myConditions[CaloCondition->condName()] = tCondition;

      }
    }
  }

  for(unsigned int a=0 ; a<vEnergySumConditions->size() ; a++){
    for(unsigned int b=0 ; b<(*vEnergySumConditions)[a].size() ; b++){

      const L1GtEnergySumTemplate* EnergySumCondition = &(*vEnergySumConditions)[a][b];

      if((EnergySumCondition->condType() == TypeETT || 
          EnergySumCondition->condType() == TypeETM ||
          EnergySumCondition->condType() == TypeHTT ||
          EnergySumCondition->condType() == TypeHTM
        ) && EnergySumCondition->nrObjects() == 1){

        SingleObjectCondition tCondition;
 
        tCondition.name              = EnergySumCondition->condName();
        tCondition.conditionCategory = EnergySumCondition->condCategory();
        tCondition.conditionType     = EnergySumCondition->condType();
        tCondition.object            = EnergySumCondition->objectType()[0];
        tCondition.threshold         = (*EnergySumCondition->objectParameter())[0].etThreshold;
        tCondition.quality           = 0;
        tCondition.etaRange          = 0;

        myConditions[EnergySumCondition->condName()] = tCondition;

      }
    }
  }

  for (CItAlgo iAlgo = theAlgoMap->begin(); iAlgo!=theAlgoMap->end(); ++iAlgo){

    int error;

    bool         algoIsValid  = true;
    unsigned int maxThreshold = 0;
    int          tAlgoMask    = myUtils.triggerMask(iAlgo->first,error);
    L1GtObject   tObject      = Mu;  // FIXME: Initial dummy value
    unsigned int tQuality     = 0;   // Only aplicable to Muons
    unsigned int tEtaRange    = 0;

    // Objects associated
    bool isMu      = false;
    bool isNoIsoEG = false;
    bool isIsoEG   = false;
    bool isCenJet  = false;
    bool isForJet  = false;
    bool isTauJet  = false;
    bool isETM     = false;
    bool isETT     = false;
    bool isHTT     = false;
    bool isHTM     = false;

    // Check if the trigger is masked
    if(tAlgoMask != 0){
      algoIsValid = false;
    }
    else{

      const L1GtAlgorithm *pAlgo = &(iAlgo->second);

      for(unsigned int i=0; i<pAlgo->algoRpnVector().size() ; i++){

        // Algorithm cannot be single algo if it requires 2 simultaneous conditions
        // FIXME: Should be improved to be sure one of the conditions is not technical (ex: BPTX)
        if(pAlgo->algoRpnVector()[i].operation == 4){
          algoIsValid = false;
          break;
        }
        else if(pAlgo->algoRpnVector()[i].operation == 32){

          string AlgoCondition                                     = pAlgo->algoRpnVector()[i].operand;
          map<string,SingleObjectCondition>::const_iterator ciCond = myConditions.find(AlgoCondition);

          // If there is no matching condition (i.e. its not a single object or energy sum condition)
          // ignore this this L1 algo
          if(ciCond == myConditions.end()){
            algoIsValid = false;
            break;
          }
          // If trigger was not invalidated by this condition we register its objects and threshold
          else{

            // Updating value for the object with the maximum threshold for this triger          
            if(maxThreshold < (*ciCond).second.threshold){
               maxThreshold = (*ciCond).second.threshold;
               tObject      = (*ciCond).second.object;
               tQuality     = (*ciCond).second.quality;
               tEtaRange    = (*ciCond).second.etaRange;
            }

            if     ((*ciCond).second.object == Mu)     {isMu      = true;}
            else if((*ciCond).second.object == NoIsoEG){isNoIsoEG = true;}
            else if((*ciCond).second.object == IsoEG)  {isIsoEG   = true;}        
            else if((*ciCond).second.object == CenJet) {isCenJet  = true;}
            else if((*ciCond).second.object == ForJet) {isForJet  = true;}
            else if((*ciCond).second.object == TauJet) {isTauJet  = true;}
            else if((*ciCond).second.object == ETM)    {isETM     = true;}
            else if((*ciCond).second.object == ETT)    {isETT     = true;}
            else if((*ciCond).second.object == HTT)    {isHTT     = true;}
            else if((*ciCond).second.object == HTM)    {isHTM     = true;}
  
          }
        }
      }
    }    

    if(algoIsValid){
  
      SingleObjectTrigger tTrigger;
      tTrigger.alias     = iAlgo->first;
      tTrigger.bit       = (iAlgo->second).algoBitNumber();
      tTrigger.prescale  = refPrescaleFactors[tTrigger.bit];
      tTrigger.threshold = maxThreshold;
      tTrigger.object    = tObject;
      tTrigger.quality   = tQuality;   // Only aplicable to Muons
      tTrigger.etaRange  = tEtaRange;  // Only aplicable to Muons


      // Counting the number of different trigger conditions
      int nCond = 0;
      if(isMu)     {nCond++;}
      if(isNoIsoEG){nCond++;}
      if(isIsoEG)  {nCond++;}
      if(isCenJet) {nCond++;}
      if(isForJet) {nCond++;}
      if(isTauJet) {nCond++;}
      if(isETM)    {nCond++;}
      if(isETT)    {nCond++;}
      if(isHTT)    {nCond++;}
      if(isHTM)    {nCond++;}

      // If the trigger matched one of the pre-defined categories it is added to
      // the corresponding trigger vector
      if     (nCond==1 && isMu     ==true)                                    {vTrigMu    .push_back(tTrigger);}
      else if(nCond==2 && isNoIsoEG==true && isIsoEG ==true)                  {vTrigEG    .push_back(tTrigger);}
      else if(nCond==1 && isIsoEG  ==true)                                    {vTrigIsoEG .push_back(tTrigger);}
      else if(nCond==3 && isCenJet ==true && isForJet==true && isTauJet==true){vTrigJet   .push_back(tTrigger);}
      else if(nCond==1 && isCenJet ==true)                                    {vTrigCenJet.push_back(tTrigger);}
      else if(nCond==1 && isForJet ==true)                                    {vTrigForJet.push_back(tTrigger);}
      else if(nCond==1 && isTauJet ==true)                                    {vTrigTauJet.push_back(tTrigger);}
      else if(nCond==1 && isETT    ==true)                                    {vTrigETT   .push_back(tTrigger);}
      else if(nCond==1 && isETM    ==true)                                    {vTrigETM   .push_back(tTrigger);}
      else if(nCond==1 && isHTT    ==true)                                    {vTrigHTT   .push_back(tTrigger);}
      else if(nCond==1 && isHTM    ==true)                                    {vTrigHTM   .push_back(tTrigger);}
    }

  }

  //--------------------------------------------------------------------------------------
  // Now that we have built vectors of SingleObjectTrigger by category we can select for
  // each category the lowest unprescaled single object trigger.
  // NOTE: Since it is not guaranteed that all categories will have at least one unprescaled
  //       trigger this method will return in that case the lowest prescale trigger available
  //--------------------------------------------------------------------------------------

  string selTrigMu     = "Undefined";
  string selTrigEG     = "Undefined";
  string selTrigIsoEG  = "Undefined";
  string selTrigJet    = "Undefined";
  string selTrigCenJet = "Undefined";
  string selTrigForJet = "Undefined";
  string selTrigTauJet = "Undefined";
  string selTrigETT    = "Undefined";
  string selTrigETM    = "Undefined";
  string selTrigHTT    = "Undefined";
  string selTrigHTM    = "Undefined";
 
  if(vTrigMu    .size() > 0){sort(vTrigMu    .begin(),vTrigMu    .end()); selTrigMu     = vTrigMu    [0].alias;}
  if(vTrigEG    .size() > 0){sort(vTrigEG    .begin(),vTrigEG    .end()); selTrigEG     = vTrigEG    [0].alias;}
  if(vTrigIsoEG .size() > 0){sort(vTrigIsoEG .begin(),vTrigIsoEG .end()); selTrigIsoEG  = vTrigIsoEG [0].alias;}
  if(vTrigJet   .size() > 0){sort(vTrigJet   .begin(),vTrigJet   .end()); selTrigJet    = vTrigJet   [0].alias;}
  if(vTrigCenJet.size() > 0){sort(vTrigCenJet.begin(),vTrigCenJet.end()); selTrigCenJet = vTrigCenJet[0].alias;}
  if(vTrigForJet.size() > 0){sort(vTrigForJet.begin(),vTrigForJet.end()); selTrigForJet = vTrigForJet[0].alias;}
  if(vTrigTauJet.size() > 0){sort(vTrigTauJet.begin(),vTrigTauJet.end()); selTrigTauJet = vTrigTauJet[0].alias;}
  if(vTrigETT   .size() > 0){sort(vTrigETT   .begin(),vTrigETT   .end()); selTrigETT    = vTrigETT   [0].alias;}
  if(vTrigETM   .size() > 0){sort(vTrigETM   .begin(),vTrigETM   .end()); selTrigETM    = vTrigETM   [0].alias;}
  if(vTrigHTT   .size() > 0){sort(vTrigHTT   .begin(),vTrigHTT   .end()); selTrigHTT    = vTrigHTT   [0].alias;}
  if(vTrigHTM   .size() > 0){sort(vTrigHTM   .begin(),vTrigHTM   .end()); selTrigHTM    = vTrigHTM   [0].alias;}

  cout << "START:   Final Selection ------------------------" << endl;
  if(iCategories["Mu"])    {cout << "Mu:"     << selTrigMu << endl;}
  if(iCategories["EG"])    {cout << "EG:"     << selTrigEG << endl;}
  if(iCategories["IsoEG"]) {cout << "IsoEG:"  << selTrigIsoEG << endl;}
  if(iCategories["Jet"])   {cout << "Jet:"    << selTrigJet << endl;}
  if(iCategories["CenJet"]){cout << "CenJet:" << selTrigCenJet << endl;}
  if(iCategories["ForJet"]){cout << "ForJet:" << selTrigForJet << endl;}
  if(iCategories["TauJet"]){cout << "TauJet:" << selTrigTauJet << endl;}
  if(iCategories["ETT"])   {cout << "ETT:"    << selTrigETT << endl;}
  if(iCategories["ETM"])   {cout << "ETM:"    << selTrigETM << endl;}
  if(iCategories["HTT"])   {cout << "HTT:"    << selTrigHTT << endl;}
  if(iCategories["HTM"])   {cout << "HTM:"    << selTrigHTM << endl;}
  cout << "END:   Final Selection ------------------------" << endl;

  if(iCategories["Mu"])    {out["Mu"]     = selTrigMu;}
  if(iCategories["EG"])    {out["EG"]     = selTrigEG;}
  if(iCategories["IsoEG"]) {out["IsoEG"]  = selTrigIsoEG;}
  if(iCategories["Jet"])   {out["Jet"]    = selTrigJet;}
  if(iCategories["CenJet"]){out["CenJet"] = selTrigCenJet;}
  if(iCategories["ForJet"]){out["ForJet"] = selTrigForJet;}
  if(iCategories["TauJet"]){out["TauJet"] = selTrigTauJet;}
  if(iCategories["ETT"])   {out["ETT"]    = selTrigETT;}
  if(iCategories["ETM"])   {out["ETM"]    = selTrigETM;}
  if(iCategories["HTT"])   {out["HTT"]    = selTrigHTT;}
  if(iCategories["HTM"])   {out["HTM"]    = selTrigHTM;}

  return out;

}

map<string,string> L1TMenuHelper::testAlgos(map<string,string> iAlgos){

  // Getting information from the menu
  const AlgorithmMap *theAlgoMap = &m_l1GtMenu->gtAlgorithmAliasMap();

  for(map<string,string>::const_iterator i = iAlgos.begin() ; iAlgos.end() != i ; i++){

    string tCategory = (*i).first;
    string tTrigger  = (*i).second;

    if(tTrigger == "" ){iAlgos[tCategory] = "Undefined";}
    else{
      if(theAlgoMap->find(tTrigger) == theAlgoMap->end()){iAlgos[tCategory] = "Undefined (Wrong Name)";}
    }

  }

  return iAlgos;

}

//-------------------------------------------------------------------------------------
// Method: enumToStringL1GtObject
//   * Converts L1GtObject (enum) to string
//-------------------------------------------------------------------------------------
string L1TMenuHelper::enumToStringL1GtObject(L1GtObject iObject){

  string out;

  switch(iObject){
    case Mu:           out = "Mu";           break;
    case NoIsoEG:      out = "NoIsoEG";      break;
    case IsoEG:        out = "IsoEG";        break;
    case CenJet:       out = "CenJet";       break;
    case ForJet:       out = "ForJet";       break;
    case TauJet:       out = "TauJet";       break;
    case ETM:          out = "ETM";          break;
    case ETT:          out = "ETT";          break;
    case HTT:          out = "HTT";          break;
    case HTM:          out = "HTM";          break;
    case JetCounts:    out = "JetCounts";    break;
    case HfBitCounts:  out = "HfBitCounts";  break;
    case HfRingEtSums: out = "HfRingEtSums"; break;
    case TechTrig:     out = "TechTrig";     break;
    case Castor:       out = "Castor";       break;
    case BPTX:         out = "BPTX";         break;
    case GtExternal:   out = "GtExternal";   break;
    default:           out = "Unknown";      break;
  };

  return out;

}

//-------------------------------------------------------------------------------------
// Method: enumToStringL1GtConditionType
//   * Converts L1GtConditionType (enum) to string
//-------------------------------------------------------------------------------------
string L1TMenuHelper::enumToStringL1GtConditionType(L1GtConditionType iConditionType){

  string out;

  switch(iConditionType){
    case TypeNull:         out = "TypeNull";         break;
    case Type1s:           out = "Type1s";           break;
    case Type2s:           out = "Type2s";           break;
    case Type2wsc:         out = "Type2wsc";         break;
    case Type2cor:         out = "Type2cor";         break;
    case Type3s:           out = "Type3s";           break;
    case Type4s:           out = "Type4s";           break;
    case TypeETM:          out = "TypeETM";          break;
    case TypeETT:          out = "TypeETT";          break;
    case TypeHTT:          out = "TypeHTT";          break;
    case TypeHTM:          out = "TypeHTM";          break;
    case TypeJetCounts:    out = "TypeJetCounts";    break;
    case TypeCastor:       out = "TypeCastor";       break;
    case TypeHfBitCounts:  out = "TypeHfBitCounts";  break;
    case TypeHfRingEtSums: out = "TypeHfRingEtSums"; break;
    case TypeBptx:         out = "TypeBptx";         break;
    case TypeExternal:     out = "TypeExternal";     break;
    default:               out = "Unknown";          break;
  };

  return out;

} 

//-------------------------------------------------------------------------------------
// Method: enumToStringL1GtConditionCategory
//   * Converts L1GtConditionCategory (enum) to string
//-------------------------------------------------------------------------------------
string L1TMenuHelper::enumToStringL1GtConditionCategory(L1GtConditionCategory iConditionCategory){

  string out;

  switch(iConditionCategory){
    case CondNull:         out = "CondNull";         break;
    case CondMuon:         out = "CondMuon";         break;
    case CondCalo:         out = "CondCalo";         break;
    case CondEnergySum:    out = "CondEnergySum";    break;
    case CondJetCounts:    out = "CondJetCounts";    break;
    case CondCorrelation:  out = "CondCorrelation";  break;
    case CondCastor:       out = "CondCastor";       break;
    case CondHfBitCounts:  out = "CondHfBitCounts";  break;
    case CondHfRingEtSums: out = "CondHfRingEtSums"; break;
    case CondBptx:         out = "CondBptx";         break;
    case CondExternal:     out = "CondExternal";     break;
    default:               out = "Unknown";          break;
  };

  return out;

}

