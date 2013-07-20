
/*
 * \file L1TMenuHelper.cc
 *
 * $Date: 2012/11/20 14:57:08 $
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



#include "DQMOffline/L1Trigger/interface/L1TMenuHelper.h"

using namespace edm;
using namespace std;

//-------------------------------------------------------------------------------------
//
//-------------------------------------------------------------------------------------
L1TMenuHelper::L1TMenuHelper(const edm::EventSetup& iSetup){

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
        tCondition.etaRange          = (*CaloCondition->objectParameter())[0].etaRange;

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
    L1GtObject   tObject      = Mu;  // Initial dummy value
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
        if(pAlgo->algoRpnVector()[i].operation == L1GtLogicParser::OP_AND){
          algoIsValid = false;
          break;
        }
        else if(pAlgo->algoRpnVector()[i].operation == L1GtLogicParser::OP_OPERAND){

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
      tTrigger.etaRange  = tEtaRange;  // Only aplicable to EG and Muons 

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
      if     (nCond==1 && isMu     ==true){
	//TODO: tTrigger.etaRange
	m_vTrigMu    .push_back(tTrigger);
      }
      else if(nCond==2 && isNoIsoEG==true && isIsoEG ==true)                  {m_vTrigEG    .push_back(tTrigger);}
      else if(nCond==1 && isIsoEG  ==true)                                    {m_vTrigIsoEG .push_back(tTrigger);}
      else if(nCond==3 && isCenJet ==true && isForJet==true && isTauJet==true){m_vTrigJet   .push_back(tTrigger);}
      else if(nCond==1 && isCenJet ==true)                                    {m_vTrigCenJet.push_back(tTrigger);}
      else if(nCond==1 && isForJet ==true)                                    {m_vTrigForJet.push_back(tTrigger);}
      else if(nCond==1 && isTauJet ==true)                                    {m_vTrigTauJet.push_back(tTrigger);}
      else if(nCond==1 && isETT    ==true)                                    {m_vTrigETT   .push_back(tTrigger);}
      else if(nCond==1 && isETM    ==true)                                    {m_vTrigETM   .push_back(tTrigger);}
      else if(nCond==1 && isHTT    ==true)                                    {m_vTrigHTT   .push_back(tTrigger);}
      else if(nCond==1 && isHTM    ==true)                                    {m_vTrigHTM   .push_back(tTrigger);}
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
 
  if(m_vTrigMu    .size() > 0){sort(m_vTrigMu    .begin(),m_vTrigMu    .end()); selTrigMu     = m_vTrigMu    [0].alias;}
  if(m_vTrigEG    .size() > 0){sort(m_vTrigEG    .begin(),m_vTrigEG    .end()); selTrigEG     = m_vTrigEG    [0].alias;}
  if(m_vTrigIsoEG .size() > 0){sort(m_vTrigIsoEG .begin(),m_vTrigIsoEG .end()); selTrigIsoEG  = m_vTrigIsoEG [0].alias;}
  if(m_vTrigJet   .size() > 0){sort(m_vTrigJet   .begin(),m_vTrigJet   .end()); selTrigJet    = m_vTrigJet   [0].alias;}
  if(m_vTrigCenJet.size() > 0){sort(m_vTrigCenJet.begin(),m_vTrigCenJet.end()); selTrigCenJet = m_vTrigCenJet[0].alias;}
  if(m_vTrigForJet.size() > 0){sort(m_vTrigForJet.begin(),m_vTrigForJet.end()); selTrigForJet = m_vTrigForJet[0].alias;}
  if(m_vTrigTauJet.size() > 0){sort(m_vTrigTauJet.begin(),m_vTrigTauJet.end()); selTrigTauJet = m_vTrigTauJet[0].alias;}
  if(m_vTrigETT   .size() > 0){sort(m_vTrigETT   .begin(),m_vTrigETT   .end()); selTrigETT    = m_vTrigETT   [0].alias;}
  if(m_vTrigETM   .size() > 0){sort(m_vTrigETM   .begin(),m_vTrigETM   .end()); selTrigETM    = m_vTrigETM   [0].alias;}
  if(m_vTrigHTT   .size() > 0){sort(m_vTrigHTT   .begin(),m_vTrigHTT   .end()); selTrigHTT    = m_vTrigHTT   [0].alias;}
  if(m_vTrigHTM   .size() > 0){sort(m_vTrigHTM   .begin(),m_vTrigHTM   .end()); selTrigHTM    = m_vTrigHTM   [0].alias;}

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

//__________________________________________________________________
// Method: enumToStringL1GtConditionCategory
//   * Converts L1GtConditionCategory (enum) to string
//__________________________________________________________________
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

//__________________________________________________________________
int L1TMenuHelper::getPrescaleByAlias(TString iCategory,TString iAlias){

    int out = -1;

    if(iCategory == "Mu"){ 
      for(unsigned int i=0 ; i<m_vTrigMu.size() ; i++){if(m_vTrigMu[i].alias==iAlias)        {return m_vTrigMu[i].prescale;}} 
    }else if(iCategory == "EG"){ 
      for(unsigned int i=0 ; i<m_vTrigEG.size() ; i++){if(m_vTrigEG[i].alias==iAlias)        {return m_vTrigEG[i].prescale;}} 
    }else if(iCategory == "IsoEG"){ 
      for(unsigned int i=0 ; i<m_vTrigIsoEG.size()  ; i++){if(m_vTrigIsoEG[i].alias==iAlias) {return m_vTrigIsoEG[i].prescale;}} 
    }else if(iCategory == "Jet"){
      for(unsigned int i=0 ; i<m_vTrigJet.size()    ; i++){if(m_vTrigJet[i].alias==iAlias)   {return m_vTrigJet[i].prescale;}} 
    }else if(iCategory == "CenJet"){ 
      for(unsigned int i=0 ; i<m_vTrigCenJet.size() ; i++){if(m_vTrigCenJet[i].alias==iAlias){return m_vTrigCenJet[i].prescale;}}
    }else if(iCategory == "ForJet"){ 
      for(unsigned int i=0 ; i<m_vTrigForJet.size() ; i++){if(m_vTrigForJet[i].alias==iAlias){return m_vTrigForJet[i].prescale;}}
    }else if(iCategory == "TauJet"){ 
      for(unsigned int i=0 ; i<m_vTrigTauJet.size() ; i++){if(m_vTrigTauJet[i].alias==iAlias){return m_vTrigTauJet[i].prescale;}}
    }else if(iCategory == "ETT"){ 
      for(unsigned int i=0 ; i<m_vTrigETT.size()    ; i++){if(m_vTrigETT[i].alias==iAlias)   {return m_vTrigETT[i].prescale;}}
    }else if(iCategory == "ETM"){ 
      for(unsigned int i=0 ; i<m_vTrigETM.size()    ; i++){if(m_vTrigETM[i].alias==iAlias)   {return m_vTrigETM[i].prescale;}}
    }else if(iCategory == "HTT"){ 
      for(unsigned int i=0 ; i<m_vTrigHTT.size()    ; i++){if(m_vTrigHTT[i].alias==iAlias)   {return m_vTrigHTT[i].prescale;}}
    }else if(iCategory == "HTM"){ 
      for(unsigned int i=0 ; i<m_vTrigHTM.size()    ; i++){if(m_vTrigHTM[i].alias==iAlias)   {return m_vTrigHTM[i].prescale;}}
    }

  return out;

}

//__________________________________________________________________
unsigned int L1TMenuHelper::getEtaRangeByAlias(TString iCategory,TString iAlias){

    unsigned int out = -1;

    if(iCategory == "Mu"){ 
      for(unsigned int i=0 ; i<m_vTrigMu.size() ; i++){if(m_vTrigMu[i].alias==iAlias)        {return m_vTrigMu[i].etaRange;}} 
    }else if(iCategory == "EG"){ 
      for(unsigned int i=0 ; i<m_vTrigEG.size() ; i++){if(m_vTrigEG[i].alias==iAlias)        {return m_vTrigEG[i].etaRange;}} 
    }else if(iCategory == "IsoEG"){ 
      for(unsigned int i=0 ; i<m_vTrigIsoEG.size()  ; i++){if(m_vTrigIsoEG[i].alias==iAlias) {return m_vTrigIsoEG[i].etaRange;}} 
    }else if(iCategory == "Jet"){
      for(unsigned int i=0 ; i<m_vTrigJet.size()    ; i++){if(m_vTrigJet[i].alias==iAlias)   {return m_vTrigJet[i].etaRange;}} 
    }else if(iCategory == "CenJet"){ 
      for(unsigned int i=0 ; i<m_vTrigCenJet.size() ; i++){if(m_vTrigCenJet[i].alias==iAlias){return m_vTrigCenJet[i].etaRange;}} 
    }else if(iCategory == "ForJet"){ 
      for(unsigned int i=0 ; i<m_vTrigForJet.size() ; i++){if(m_vTrigForJet[i].alias==iAlias){return m_vTrigForJet[i].etaRange;}} 
    }else if(iCategory == "TauJet"){ 
      for(unsigned int i=0 ; i<m_vTrigTauJet.size() ; i++){if(m_vTrigTauJet[i].alias==iAlias){return m_vTrigTauJet[i].etaRange;}} 
    }else if(iCategory == "ETT"){ 
      for(unsigned int i=0 ; i<m_vTrigETT.size()    ; i++){if(m_vTrigETT[i].alias==iAlias)   {return m_vTrigETT[i].etaRange;}} 
    }else if(iCategory == "ETM"){ 
      for(unsigned int i=0 ; i<m_vTrigETM.size()    ; i++){if(m_vTrigETM[i].alias==iAlias)   {return m_vTrigETM[i].etaRange;}} 
    }else if(iCategory == "HTT"){ 
      for(unsigned int i=0 ; i<m_vTrigHTT.size()    ; i++){if(m_vTrigHTT[i].alias==iAlias)   {return m_vTrigHTT[i].etaRange;}}  
    }else if(iCategory == "HTM"){ 
      for(unsigned int i=0 ; i<m_vTrigHTM.size()    ; i++){if(m_vTrigHTM[i].alias==iAlias)   {return m_vTrigHTM[i].etaRange;}} 
    }

  return out;

}

//__________________________________________________________________
unsigned int L1TMenuHelper::getQualityAlias(TString iCategory,TString iAlias){

    unsigned int out = -1;

    if(iCategory == "Mu"){ 
      for(unsigned int i=0 ; i<m_vTrigMu.size() ; i++){if(m_vTrigMu[i].alias==iAlias)        {return m_vTrigMu[i].quality;}} 
    }else if(iCategory == "EG"){ 
      for(unsigned int i=0 ; i<m_vTrigEG.size() ; i++){if(m_vTrigEG[i].alias==iAlias)        {return m_vTrigEG[i].quality;}} 
    }else if(iCategory == "IsoEG"){ 
      for(unsigned int i=0 ; i<m_vTrigIsoEG.size()  ; i++){if(m_vTrigIsoEG[i].alias==iAlias) {return m_vTrigIsoEG[i].quality;}} 
    }else if(iCategory == "Jet"){
      for(unsigned int i=0 ; i<m_vTrigJet.size()    ; i++){if(m_vTrigJet[i].alias==iAlias)   {return m_vTrigJet[i].quality;}} 
    }else if(iCategory == "CenJet"){ 
      for(unsigned int i=0 ; i<m_vTrigCenJet.size() ; i++){if(m_vTrigCenJet[i].alias==iAlias){return m_vTrigCenJet[i].quality;}} 
    }else if(iCategory == "ForJet"){ 
      for(unsigned int i=0 ; i<m_vTrigForJet.size() ; i++){if(m_vTrigForJet[i].alias==iAlias){return m_vTrigForJet[i].quality;}} 
    }else if(iCategory == "TauJet"){ 
      for(unsigned int i=0 ; i<m_vTrigTauJet.size() ; i++){if(m_vTrigTauJet[i].alias==iAlias){return m_vTrigTauJet[i].quality;}} 
    }else if(iCategory == "ETT"){ 
      for(unsigned int i=0 ; i<m_vTrigETT.size()    ; i++){if(m_vTrigETT[i].alias==iAlias)   {return m_vTrigETT[i].quality;}} 
    }else if(iCategory == "ETM"){ 
      for(unsigned int i=0 ; i<m_vTrigETM.size()    ; i++){if(m_vTrigETM[i].alias==iAlias)   {return m_vTrigETM[i].quality;}} 
    }else if(iCategory == "HTT"){ 
      for(unsigned int i=0 ; i<m_vTrigHTT.size()    ; i++){if(m_vTrigHTT[i].alias==iAlias)   {return m_vTrigHTT[i].quality;}}  
    }else if(iCategory == "HTM"){ 
      for(unsigned int i=0 ; i<m_vTrigHTM.size()    ; i++){if(m_vTrigHTM[i].alias==iAlias)   {return m_vTrigHTM[i].quality;}} 
    }

  return out;

}
