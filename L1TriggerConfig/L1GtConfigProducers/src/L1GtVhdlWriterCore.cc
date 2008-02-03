/**
 * \class L1GtVhdlWriterCore
 *
 *
 * Description: a class to deal with VHDL template files
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Philipp Wagner
 *
 * $Date: 2008/02/02 22:13:25 $
 * $Revision: 1.3 $
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterCore.h"

// system include files
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

// CMSSW headers
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterBitManager.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"

// constructor(s)
L1GtVhdlWriterCore::L1GtVhdlWriterCore(const std::string &templatesDirectory, const std::string &outputDirectory, const bool &debug)
{

    // set templates directory
    vhdlDir_=templatesDirectory;

    // set output directory
    outputDir_=outputDirectory;

    objType2Str_[Mu]="muon";
    objType2Str_[NoIsoEG]="eg";
    objType2Str_[IsoEG]="ieg";
    objType2Str_[ForJet]="fwdjet";
    objType2Str_[TauJet]="tau";
    objType2Str_[CenJet]="jet";
    objType2Str_[JetCounts]="jet_cnts";
    objType2Str_[HTT]="htt";
    objType2Str_[ETT]="ett";
    objType2Str_[ETM]="etm";

    caloType2Int_[IsoEG]="0";
    caloType2Int_[NoIsoEG]="1";
    caloType2Int_[CenJet]="2";
    caloType2Int_[TauJet]="3";
    caloType2Int_[ForJet]="4";
    caloType2Int_[Mu]="5";
    caloType2Int_[ETM]="6";

    condType2Str_[Type1s]="1_s";
    condType2Str_[Type2s]="2_s";
    condType2Str_[Type2wsc]="2_wsc";
    condType2Str_[Type3s]="3";
    condType2Str_[Type4s]="4";
    condType2Str_[Type2cor]="Type2cor";
    condType2Str_[TypeETM]="cond";
    condType2Str_[TypeETT]="cond";
    condType2Str_[TypeHTT]="cond";
    condType2Str_[TypeJetCounts]="TypeJetCounts";

    caloObjects_.push_back(IsoEG);
    caloObjects_.push_back(NoIsoEG);
    caloObjects_.push_back(CenJet);
    caloObjects_.push_back(ForJet);
    caloObjects_.push_back(TauJet);

    esumObjects_.push_back(HTT);
    esumObjects_.push_back(ETT);
    esumObjects_.push_back(ETM);

    // prepare number of condition vector with two empty string vectors for the two condition chips
    std::vector<std::string> temp;
    numberOfConditions_.push_back(temp);
    numberOfConditions_.push_back(temp);
    
    // set debug mode
    debugMode_= debug;

}


// destructor
L1GtVhdlWriterCore::~L1GtVhdlWriterCore()
{
    // empty
}


bool L1GtVhdlWriterCore::returnConditionsOfOneClass(const L1GtConditionType &type,  const L1GtConditionCategory &category, const L1GtObject &object, const ConditionMap &map, ConditionMap &outputMap)
{
    bool status = false;

    ConditionMap::const_iterator condIter = map.begin();
    while (condIter!=map.end())
    {
        if (condIter->second->condCategory()==category && condIter->second->condType()==type  && (condIter->second->objectType())[0]==object)
        {
            outputMap[condIter->first]=condIter->second;
            status = true;
        }
        condIter++;
    }

    return status;
}


bool L1GtVhdlWriterCore::findObjectType(const L1GtObject &object,  ConditionMap &map)
{
    bool status = false;

    ConditionMap::iterator condIter = map.begin();
    while (condIter!=map.end())
    {
        for (unsigned int i=0; i<(condIter->second->objectType()).size(); i++)
        {
            if (condIter->second->objectType()[i]==object)
            {
                status = true;
                break;
            }
        }

        if (status) break;
        condIter++;
    }

    return status;
}


void L1GtVhdlWriterCore::buildMuonParameterMap(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
std::map<std::string,std::string> &muonParameters, const std::vector<ConditionMap> &conditionMap)
{
    // vector containing all relevant types for muon conditions
    std::vector<L1GtConditionType> muonConditionTypes;

    muonConditionTypes.push_back(Type1s);
    muonConditionTypes.push_back(Type2s);
    muonConditionTypes.push_back(Type2wsc);
    muonConditionTypes.push_back(Type3s);
    muonConditionTypes.push_back(Type4s);

    for (unsigned int i = 0; i<muonConditionTypes.size(); i++)
    {
        //std::cout<<i<<std::endl;

        ConditionMap MuonConditions1s;

        countCondsAndAdd2NumberVec(muonConditionTypes.at(i), CondMuon, Mu, conditionMap.at(condChip-1), MuonConditions1s, condChip);

        /*
        // Get the absoloute amount of conditions
        std::string initstr="CONSTANT nr_"+objType2Str_[Mu]+"_"+condType2Str_[muonConditionTypes.at(i)]+" : integer := ";

        if (returnConditionsOfOneClass(muonConditionTypes.at(i),CondMuon,Mu,conditionMap.at(condChip-1), MuonConditions1s))
            initstr+=(int2str(MuonConditions1s.size())+";");
        else
            initstr+=("0;");

        numberOfConditions_.at(condChip-1).push_back(initstr);

        */

        //std::cout<< std::hex << (*myObjectParameter).at(0).etThreshold << std::endl;
        unsigned int counter=0;

        for (ConditionMap::const_iterator iterCond = MuonConditions1s.begin(); iterCond != MuonConditions1s.end(); iterCond++)
        {

            // add this condition to name -> integer conversion map
            conditionToIntegerMap[iterCond->first]=counter;

            L1GtMuonTemplate* m_gtMuonTemplate = static_cast<L1GtMuonTemplate*>(iterCond->second);
            const std::vector<L1GtMuonTemplate::ObjectParameter>* op =  m_gtMuonTemplate->objectParameter();

            if (muonConditionTypes.at(i)==Type1s)
            {

                // build eta
                muonParameters["eta_1_s"] += (bm_.buildEtaMuon(op, 1,counter));

                // add the parameters to parameter map
                muonParameters["phi_h_1_s"]+= bm_.buildPhiMuon(op,1,counter,true);
                muonParameters["phi_l_1_s"]+=bm_.buildPhiMuon(op,1,counter,false);

            } else

            if (muonConditionTypes.at(i)==Type2s)
            {

                // build eta
                muonParameters["eta_2_s"] += (bm_.buildEtaMuon(op, 2,counter));

                // add the parameters to parameter map
                muonParameters["phi_h_2_s"]+= bm_.buildPhiMuon(op,2,counter,true);
                muonParameters["phi_l_2_s"]+=bm_.buildPhiMuon(op,2,counter,false);

            } else
            //m_gtMuonTemplate->print(std::cout);

            if (muonConditionTypes.at(i)==Type3s)
            {

                // build eta
                muonParameters["eta_3"] += (bm_.buildEtaMuon(op, 3,counter));

                // add the parameters to parameter map
                muonParameters["phi_h_3"]+= bm_.buildPhiMuon(op,3,counter,true);
                muonParameters["phi_l_3"]+=bm_.buildPhiMuon(op,3,counter,false);
            }

            if (muonConditionTypes.at(i)==Type4s)
            {

                // build eta
                muonParameters["eta_4"] += (bm_.buildEtaMuon(op, 4,counter));

                // add the parameters to parameter map
                muonParameters["phi_h_4"]+= bm_.buildPhiMuon(op,4,counter,true);
                muonParameters["phi_l_4"]+=bm_.buildPhiMuon(op,4,counter,false);
            }

            if (muonConditionTypes.at(i)==Type2wsc)
            {
                const L1GtMuonTemplate::CorrelationParameter* cp =  m_gtMuonTemplate->correlationParameter();

                // build eta
                muonParameters["eta_2_wsc"] += (bm_.buildEtaMuon(op, 2,counter));

                // build phi
                muonParameters["phi_h_2_wsc"]+= bm_.buildPhiMuon(op,2,counter,true);
                muonParameters["phi_l_2_wsc"]+=bm_.buildPhiMuon(op,2,counter,false);

                // build delta_eta
                std::ostringstream dEta;
                muonParameters["delta_eta"] += bm_.buildDeltaEtaMuon(cp,counter);

                // build delta_phi
                muonParameters["delta_phi"] += bm_.buildDeltaPhiMuon(cp,counter);

            }
            counter++;
        }
    }

}


bool L1GtVhdlWriterCore::buildCaloParameterMap(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
std::map<std::string,std::string> &caloParameters, const L1GtObject &caloObject,const std::vector<ConditionMap> &conditionMap)
{
    // vector containing all relevant types for calo conditions
    std::vector<L1GtConditionType> caloConditionTypes;

    caloConditionTypes.push_back(Type1s);
    caloConditionTypes.push_back(Type2s);
    caloConditionTypes.push_back(Type2wsc);
    caloConditionTypes.push_back(Type4s);

    for (unsigned int i = 0; i<caloConditionTypes.size(); i++)
    {
        unsigned int counter=0;

        ConditionMap caloConditions;

        // stores all conditions of type given in the first three parameters
        countCondsAndAdd2NumberVec(caloConditionTypes.at(i), CondCalo, caloObject, conditionMap.at(condChip-1), caloConditions, condChip);

        for (ConditionMap::const_iterator iterCond = caloConditions.begin(); iterCond != caloConditions.end(); iterCond++)
        {

            // add this condition to name -> integer conversion map
            conditionToIntegerMap[iterCond->first]=counter;

            L1GtCaloTemplate* m_gtCaloTemplate = static_cast<L1GtCaloTemplate*>(iterCond->second);
            const std::vector<L1GtCaloTemplate::ObjectParameter>* op =  m_gtCaloTemplate->objectParameter();

            if (caloConditionTypes.at(i)==Type1s)
            {

                //= static_cast<std::vector<L1GtCaloTemplate::ObjectPaenergySumParameter+=rameter*> >(op);

                // build eta
                caloParameters["eta_1_s"] += (bm_.buildEtaCalo(op, 1,counter));

                caloParameters["phi_1_s"]+= bm_.buildPhiCalo(op,1,counter);

            } else if (caloConditionTypes.at(i)==Type2s)
            {

                // build eta
                caloParameters["eta_2_s"] += (bm_.buildEtaCalo(op, 2,counter));

                // add the parameters to parameter map
                caloParameters["phi_2_s"]+= bm_.buildPhiCalo(op,2,counter);

            } else if (caloConditionTypes.at(i)==Type4s)
            {
                // build eta
                caloParameters["eta_4"] += (bm_.buildEtaCalo(op, 4,counter));

                // add the parameters to parameter map
                caloParameters["phi_4"]+= bm_.buildPhiCalo(op,4,counter);

            } else if (caloConditionTypes.at(i)==Type2wsc)
            {
                const L1GtCaloTemplate::CorrelationParameter* cp =  m_gtCaloTemplate->correlationParameter();

                // build eta
                caloParameters["eta_2_wsc"] += (bm_.buildEtaCalo(op, 2,counter));

                // build phi
                caloParameters["phi_2_wsc"]+= bm_.buildPhiCalo(op,2,counter);

                // build delta_eta
                caloParameters["delta_eta"] += bm_.buildDeltaEtaCalo(cp,counter);

                // build delta_phi
                caloParameters["delta_phi"] += bm_.buildDeltaPhiCalo(cp,counter);
            }

            counter++;

        }

    }

    return true;
}


bool L1GtVhdlWriterCore::buildEnergySumParameter(const unsigned short int &condChip, const L1GtObject &object, std::map<std::string,int> &conditionToIntegerMap,
std::string &energySumParameter, const std::vector<ConditionMap> &conditionMap)
{

    L1GtConditionType type;

    unsigned int counter=0;

    if (object==HTT) type= TypeHTT;
    else if (object==ETM) type= TypeETM;
    else if (object==ETT) type= TypeETT;

    ConditionMap esumsConditions;
    // stores all conditions of type given in the first three parameters
    countCondsAndAdd2NumberVec(type, CondEnergySum, object, conditionMap.at(condChip-1), esumsConditions, condChip);

    for (ConditionMap::const_iterator iterCond = esumsConditions.begin(); iterCond != esumsConditions.end(); iterCond++)
    {
        L1GtEnergySumTemplate* energySumTempl = static_cast<L1GtEnergySumTemplate*>(iterCond->second);
        const std::vector<L1GtEnergySumTemplate::ObjectParameter>* op =  energySumTempl->objectParameter();

        if (bm_.buildPhiEnergySum(op,1,counter)!="") energySumParameter+=bm_.buildPhiEnergySum(op,1,counter);

        conditionToIntegerMap[(iterCond->first)]=counter;

        counter++;
    }

    return true;

}


bool L1GtVhdlWriterCore::buildCommonParameter(L1GtVhdlTemplateFile &particle,const L1GtObject &object, const L1GtConditionCategory &category, std::string &parameterStr, const ConditionMap &conditionMap)
{

    std::map<L1GtConditionType,std::string> condType2Strcopy = condType2Str_;

    // make a copy since types will deleted later after they are processed
    std::map<L1GtConditionType,std::string>::iterator typeIterator = condType2Strcopy.begin();

    while (typeIterator != condType2Strcopy.end())
    {

        ConditionMap outputMap;

        if (returnConditionsOfOneClass( (*typeIterator).first, category, object, conditionMap,  outputMap))
        {

            // special treatment for jet counts in buildCodChipParameters
            if (object !=JetCounts)
            {
                std::string tempStr2 = particle.returnParameterMap()["COMMON"];
                particle.findAndReplaceString(tempStr2,"$(particle)",objType2Str_[object]);
                particle.findAndReplaceString(tempStr2,"$(type)",condType2Str_[(*typeIterator).first]);
                parameterStr+=(tempStr2+"<= '0';\n");

                // remove this type since it was already processed
                condType2Strcopy.erase(typeIterator);
            }

        }

        typeIterator++;

    }

    return true;
}


bool L1GtVhdlWriterCore::buildCondChipParameters(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
const std::vector<ConditionMap> &conditionMap, std::map<std::string, L1GtVhdlTemplateFile> &templates, std::map<std::string, std::string> &commonParams)
{

    // open the necessary internal templates
    L1GtVhdlTemplateFile muon,calo,esums,jet,charge1s, charge2s,charge2wsc,charge3s,charge4s;

    muon.open(vhdlDir_+"InternalTemplates/muon",true);
    muon.substitute("particle", "muon");

    calo.open(vhdlDir_+"InternalTemplates/calo",true);
    esums.open(vhdlDir_+"InternalTemplates/esums",true);
    jet.open(vhdlDir_+"InternalTemplates/jet_cnts",true);

    charge1s.open(vhdlDir_+"InternalTemplates/charge_1s",false);
    charge2s.open(vhdlDir_+"InternalTemplates/charge_2s",false);
    charge2wsc.open(vhdlDir_+"InternalTemplates/charge_2_wsc",false);
    charge3s.open(vhdlDir_+"InternalTemplates/charge_3",false);
    charge4s.open(vhdlDir_+"InternalTemplates/charge_4",false);

    // only for jet_cnts common parameter relevant since common parameter is build in this routine
    std::vector<unsigned int> processedTypes;

    //-------------------------------------build the common parameters-------------------------------------------

    // build $(muon_common)
    buildCommonParameter(muon,Mu, CondMuon, commonParams["muon_common"],conditionMap.at(condChip-1));

    // build $(calo_common) - loop over all calo objects
    std::vector<L1GtObject>::iterator caloObjIter = caloObjects_.begin();

    while (caloObjIter != caloObjects_.end())
    {

        buildCommonParameter(muon,(*caloObjIter), CondCalo, commonParams["calo_common"],conditionMap.at(condChip-1));
        caloObjIter++;

    }

    // build $(esums_common) - loop over all esums objects
    std::vector<L1GtObject>::iterator esumObjIter = esumObjects_.begin();

    while (esumObjIter != esumObjects_.end())
    {

        buildCommonParameter(esums,(*esumObjIter), CondEnergySum, commonParams["esums_common"],conditionMap.at(condChip-1));
        esumObjIter++;

    }

    //--------------------------------------build the parameter maps----------------------------------------------

    // loop over condition map
    for (ConditionMap::const_iterator iterCond = conditionMap.at(condChip-1).begin(); iterCond != conditionMap.at(condChip-1).end(); iterCond++)
    {

        switch ((iterCond->second)->condCategory())
        {

            case CondMuon:
            {

                int cond2int;
                if (!getIntVal(conditionToIntegerMap,(iterCond->first),cond2int))
                {
                    msg("Panik! Condition "+(iterCond->first)+" does not have a integer equivalent!");
                    break;
                }

                std::string intVal = index4CondChipVhd(cond2int);

                L1GtVhdlTemplateFile muoncopy = muon;

                muoncopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );

                muoncopy.substitute("ser_no",intVal);

                muoncopy.substitute("name",iterCond->first);

                if ((iterCond->second)->condType() == Type1s)
                {
                    muoncopy.insert("charge",charge1s);
                } else if ((iterCond->second)->condType() == Type2s)
                {
                    muoncopy.insert("charge",charge2s);
                } else if ((iterCond->second)->condType() == Type2wsc)
                {
                    muoncopy.insert("charge",charge2wsc);
                } else if ((iterCond->second)->condType() == Type3s)
                {
                    muoncopy.insert("charge",charge3s);
                } else if ((iterCond->second)->condType() == Type4s)
                {
                    muoncopy.insert("charge",charge4s);
                }

                std::string tempStr = muoncopy.returnParameterMap()["PREALGO"];

                muoncopy.findAndReplaceString(tempStr,"$(particle)","muon");
                muoncopy.findAndReplaceString(tempStr,"$(ser_no)",intVal);
                muoncopy.findAndReplaceString(tempStr,"$(type)",condType2Str_[(iterCond->second)->condType()]);

                muoncopy.append(tempStr);

                muoncopy.removeEmptyLines();

                // add the processed internal template to parameter map
                templates["muon"].append(muoncopy);

            }
            break;

            case CondCalo:
            {
                int cond2int;

                if (!getIntVal(conditionToIntegerMap,(iterCond->first),cond2int))
                {
                    msg("Panik! Condition "+(iterCond->first)+" does not have a integer equivalent!");
                    break;
                }

                std::string intVal = index4CondChipVhd(cond2int);

                L1GtVhdlTemplateFile calocopy = calo;

                calocopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );
                calocopy.substitute("particle",objType2Str_[((iterCond->second)->objectType()).at(0)] );
                calocopy.substitute("name",iterCond->first);
                calocopy.substitute("ser_no",intVal);
                calocopy.substitute("calo_nr",caloType2Int_[((iterCond->second)->objectType()).at(0)]);

                // builds something like tau_1_s(20));
                calocopy.append(objType2Str_[((iterCond->second)->objectType()).at(0)] +"_"+condType2Str_[(iterCond->second)->condType()]+"("+intVal+"));");

                templates["calo"].append(calocopy);

            }
            break;

            case CondEnergySum:
            {

                int cond2int;

                if (!getIntVal(conditionToIntegerMap,(iterCond->first),cond2int))
                {
                    msg("Panik! Condition "+(iterCond->first)+" does not have a integer equivalent!");
                    break;
                }

                std::string intVal = index4CondChipVhd(cond2int);

                L1GtVhdlTemplateFile esumscopy = esums;

                esumscopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );
                esumscopy.substitute("particle",objType2Str_[((iterCond->second)->objectType()).at(0)] );
                esumscopy.substitute("name",iterCond->first);
                esumscopy.substitute("ser_no",intVal);

                if (((iterCond->second)->objectType()).at(0)==ETM )
                    esumscopy.substitute("if_etm_then_1_else_0","1");
                else
                    esumscopy.substitute("if_etm_then_1_else_0","0");

                // builds something like htt_cond(4));
                esumscopy.append(objType2Str_[((iterCond->second)->objectType()).at(0)] +"_"+condType2Str_[(iterCond->second)->condType()]+"("+ intVal+"));");

                templates["esums"].append(esumscopy);
            }

            break;

            case CondJetCounts:
            {

                // build the common parameter for the jet counts

                L1GtJetCountsTemplate* jetsTemplate = static_cast<L1GtJetCountsTemplate*>(iterCond->second);

                int nObjects = iterCond->second->nrObjects();

                const std::vector<L1GtJetCountsTemplate::ObjectParameter>* op =  jetsTemplate->objectParameter();

                for (int i = 0; i < nObjects; i++)
                {

                    std::vector<unsigned int>::iterator p = find(processedTypes.begin(),
                        processedTypes.end(), (*op)[i].countIndex);

                    // check, weather this count index was already processed
                    // and process it if not
                    if (p==processedTypes.end())
                    {
                        std::ostringstream indStr;
                        indStr<<(*op)[i].countIndex;

                        std::string tempStr2 = jet.returnParameterMap()["COMMON"];
                        jet.findAndReplaceString(tempStr2,"$(particle)",objType2Str_[JetCounts]);
                        jet.findAndReplaceString(tempStr2,"$(type)",indStr.str());
                        commonParams["jet_cnts_common"]+=(tempStr2+"<= '0';\n");
                        processedTypes.push_back((*op)[i].countIndex);
                    }

                }

                int cond2int;

                if (!getIntVal(conditionToIntegerMap,(iterCond->first),cond2int))
                {
                    msg("Panik! Condition "+(iterCond->first)+" does not have a integer equivalent!");
                    break;
                }

                std::string intVal = index4CondChipVhd(cond2int);

                L1GtVhdlTemplateFile jetcopy = jet;

                jetcopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );
                jetcopy.substitute("name",iterCond->first);

                jetcopy.substitute("ser_no",intVal);

                // builds the final string
                std::string tempStr = jetcopy.returnParameterMap()["PREALGO"];

                jetcopy.findAndReplaceString(tempStr,"$(particle)","jet_cnt");
                jetcopy.findAndReplaceString(tempStr,"$(ser_no)",intVal);
                jetcopy.findAndReplaceString(tempStr,"$(type)",condType2Str_[(iterCond->second)->condType()]);

                jetcopy.append(tempStr);
                jetcopy.removeEmptyLines();

                templates["jet_cnts"].append(jetcopy);
            }

            break;

            case CondCorrelation:
            {
                //empty
            }
            break;

            case CondNull:
            {
                //empty
            }
            break;

            default:
            {
                //empty
            }
            break;
        }
    }

    // delete superfluous newlines at the end of common parameters
    std::map<std::string,std::string>::iterator pIt = commonParams.begin();

    while(pIt!=commonParams.end())
    {
        if ((pIt->second)[((pIt->second).length()-1)] == '\n' ) (pIt->second) = (pIt->second).substr(0,(pIt->second).length()-1);
        pIt++;
    }

    return true;

}


bool L1GtVhdlWriterCore::processAlgorithmMap(const AlgorithmMap &algorithmMap, std::vector<ConditionMap> &conditionMap,
std::map<std::string,int> &conditionToIntegerMap, std::map<int, std::string> &algorithmsChip1, std::map<int, std::string> &algorithmsChip2)
{
    AlgorithmMap::const_iterator algoIter=algorithmMap.begin();

    // loop over algorithm map
    while (algoIter!=algorithmMap.end())
    {
        // msg(algoIter->first);

        L1GtVhdlTemplateFile dummy;

        // get the logical expression
        std::string logicalExpr = algoIter->second->algoLogicalExpression();
        std::vector<std::string> conditions;

        dummy.getConditionsFromAlgo(logicalExpr,conditions);

        std::string logicalExprCopy= logicalExpr;

        // loop over all condition names appearing in algorithm and replace them by type and integer value
        for (unsigned int i=0; i<conditions.size(); i++)
        {
            std::ostringstream newExpr;
            // look for the condition on correct chip
            L1GtCondition* cond = conditionMap.at(algoIter->second->algoChipNumber())[conditions.at(i)];

            // check weather condition exists
            if (cond!=NULL)
            {

                newExpr<<objType2Str_[(cond->objectType()).at(0)] ;

                newExpr
                    <<"_"
                    << condType2Str_[cond->condType()]
                    << "(";

                newExpr
                    << conditionToIntegerMap[conditions.at(i)]
                    << ")";
                dummy.findAndReplaceString(logicalExpr,conditions.at(i), newExpr.str());
            }
            else msg("Panik! Didn't find Condition "+conditions.at(i));
        }

        // has to be checked!
        std::vector<int> orderConditionChip;
        orderConditionChip.push_back(2);
        orderConditionChip.push_back(1);

        int pin = algoIter->second->algoOutputPin(2,96,orderConditionChip);

        //msg(int2str(pin));

        if (pin<0) pin*=-1;

        if (algoIter->second->algoChipNumber()==0)
        {
            algorithmsChip1[pin]=logicalExpr;
            if (debugMode_) algorithmsChip1[pin]+=("-- "+logicalExprCopy);
        }
        else if (algoIter->second->algoChipNumber()==1)
        {
            algorithmsChip2[pin]=logicalExpr;
            if (debugMode_) algorithmsChip2[pin]+=("-- "+logicalExprCopy);
        }
        else ;

        algoIter++;
    }

    return true;

}


bool L1GtVhdlWriterCore::makeFirmware(std::vector<ConditionMap> &conditionMap,const AlgorithmMap &algorithmMap)
{
    std::map<std::string,int> conditionToIntegerMap;

    //--------------------------------------build setups-------------------------------------------------------

    // loop over the two condition chips
    for (unsigned short int i =1; i<=2; i++)
    {
        // ----------------------- muon setup -------------------------------------------------------

        std::map<std::string,std::string> muonParameters;
        buildMuonParameterMap(i, conditionToIntegerMap,muonParameters,conditionMap);
        writeMuonSetupVhdl(muonParameters,"muon",i);

        // ----------------------- calo setup -------------------------------------------------------

        // map containing all calo object types correlated with the strings
        // under which they appear in the firmware
        std::vector<L1GtObject>::iterator caloObjIter = caloObjects_.begin();

        // loop over all calo objectsbuildMuonSetupVhdl
        while (caloObjIter != caloObjects_.end())
        {
            std::map<std::string,std::string> caloParameters;
            buildCaloParameterMap(i, conditionToIntegerMap,caloParameters,(*caloObjIter),conditionMap);
            writeMuonSetupVhdl(caloParameters,objType2Str_[(*caloObjIter)],i);

            caloObjIter++;
        }

        // ----------------------- etm setup ---------------------------------------------------------

        // map containing all calo object types correlated with the strings
        // under which they appear in the firmware
        std::vector<L1GtObject>::iterator esumObjIter = esumObjects_.begin();

        while (esumObjIter != esumObjects_.end())
        {
            std::string etmParameter;

            buildEnergySumParameter(i, (*esumObjIter), conditionToIntegerMap,etmParameter, conditionMap);
            writeEtmSetup(etmParameter, i);

            esumObjIter++;

        }

        // add jet counts to condition 2 integer map
        addJetCountsToCond2IntMap(i, conditionMap, conditionToIntegerMap);

        // --------------------cond chip setup---------------------------------------------------------
        // Important: all other setups have to be build BEFORE this one because it needs a
        // complete condition2integer map !!!

        std::map<std::string, L1GtVhdlTemplateFile> templates;
        std::map<std::string, std::string> common;
        buildCondChipParameters(i, conditionToIntegerMap, conditionMap,templates,common);
        writeConditionChipSetup(templates,common,i);

        // despite yet not existing they have to appear in cond_chip_pkg vhds
        initializeDeltaConditions();

        // --------------------cond chip pkg------------------------------------------------------------
        writeCondChipPkg(i);

        // debug
        if (debugMode_)
        {
            //printConditionsOfCategory(CondEnergySum, conditionMap.at(i-1));
        }

    }

    if (debugMode_) writeCond2intMap2File(conditionToIntegerMap);

    //-------------------------------process algorithms----------------------------------------------------------

    std::map<int, std::string> algorithmsChip1;
    std::map<int, std::string> algorithmsChip2;

    processAlgorithmMap(algorithmMap,conditionMap,conditionToIntegerMap,algorithmsChip1,algorithmsChip2);
    writeAlgoSetup(algorithmsChip1, algorithmsChip2);

    return true;
}


std::string L1GtVhdlWriterCore::gtTemplatesPath()
{
    return vhdlDir_;
}


std::string L1GtVhdlWriterCore::int2str(const int &integerValue)
{
    std::ostringstream oss;
    oss<<integerValue;
    return oss.str();

}


void L1GtVhdlWriterCore::buildCommonHeader(std::map<std::string,std::string> &headerParameters, const std::vector<std::string> &connectedChannels)
{
    commonHeader_.open(vhdlDir_+"InternalTemplates/header");

    std::map<std::string,std::string>::iterator iter =  headerParameters.begin();

    // loop over the header parameter map and replace all subsitution
    // parameters with their values
    while( iter != headerParameters.end() )
    {
        commonHeader_.substitute((*iter).first,(*iter).second);
        iter++;
    }

    commonHeader_.insert("$(connected_channels_1)",connectedChannels);
    commonHeader_.insert("$(connected_channels_2)",connectedChannels);

    //---------------------build the Quartus configuration files----------------------------------------

    writeQsfSetupFiles(headerParameters["version"]);

    msg("Build Common header and Quartus setup files sucessuflly!");

}


L1GtVhdlTemplateFile L1GtVhdlWriterCore::openVhdlFileWithCommonHeader(const std::string &filename, const std::string &outputFilename)
{
    L1GtVhdlTemplateFile commonHeaderCp;
    commonHeaderCp = commonHeader_;

    commonHeaderCp.substitute("vhdl_file_name",outputFilename);

    L1GtVhdlTemplateFile myTemplate(vhdlDir_+"Templates/"+filename);
    myTemplate.insert("$(header)", commonHeaderCp);

    return myTemplate;
}


void L1GtVhdlWriterCore::writeMuonSetupVhdl(std::map<std::string,std::string> &muonParameters, const std::string &particle, unsigned short int &condChip)
{
    std::string filename;

    L1GtVhdlTemplateFile muonTemplate,internalTemplate,internalTemplateCopy;

    // choose the correct template file name
    if (particle=="muon")
    {
        filename = "muon_setup.vhd";
        internalTemplate.open(vhdlDir_+"InternalTemplates/muonbody",true);
    }
    else
    {
        filename = "calo_setup.vhd";
        internalTemplate.open(vhdlDir_+"InternalTemplates/calobody",true);
    }

    std::string outputFile;
    outputFile = filename;

    // modify filename
    if (particle!="muon")
        muonTemplate.findAndReplaceString(outputFile,"calo",particle);

    // add condition chip index to output filename
    muonTemplate.findAndReplaceString(outputFile,".",int2str(condChip)+".");

    // open the template file and insert header
    muonTemplate = openVhdlFileWithCommonHeader(filename, outputFile);

    std::map<std::string,std::string> parameterMap = internalTemplate.returnParameterMap();

    std::vector<std::string> substitutionParameters;

    // fills the map with all substitution parameters of the template file
    substitutionParameters=muonTemplate.getSubstitutionParametersFromTemplate();

    // insert a emty line at the end of each block
    internalTemplate.append("");

    std::map<std::string, std::string>::iterator iter;

    // loop over all substitution parameters that have been extracted from the template file
    for (unsigned int i=0; i<substitutionParameters.size(); i++)
    {
        // make a working copy of the template file
        internalTemplateCopy = internalTemplate;

        // choose which of the three constant template strings to take - only for calo!
        if (particle!="muon")
        {

            if(substitutionParameters.at(i).substr(0,3) == "eta")
                internalTemplateCopy.substitute("constant", parameterMap["eta"]);
            else if(substitutionParameters.at(i).substr(0,3) == "phi")
                internalTemplateCopy.substitute("constant", parameterMap["phi"]);
            else if(substitutionParameters.at(i).substr(0,3) == "del"/*ta*/)
            {
                internalTemplateCopy.substitute("constant", parameterMap["delta"]);
                while (internalTemplateCopy.substitute("delta", substitutionParameters[i])) internalTemplateCopy.substitute("delta", substitutionParameters[i]);

            }
        }

        // final preparation of the internal template before it is inserted in the actual template file
        internalTemplateCopy.substitute("type", substitutionParameters[i]);

        // subsitute the second occurance of type without "_l" and "_h"

        std::string paramCopy = substitutionParameters[i];

        internalTemplateCopy.findAndReplaceString(paramCopy,"_l","");
        internalTemplateCopy.findAndReplaceString(paramCopy,"_h","");

        internalTemplateCopy.substitute("type2", paramCopy);

        internalTemplateCopy.substitute("others", parameterMap[substitutionParameters[i]]);

        // replace all the occurances of "particle"
        while (muonTemplate.substitute("particle",particle)) muonTemplate.substitute("particle", particle);

        // remove the the parameter $(content) if its empty
        iter=muonParameters.find(substitutionParameters[i]);

        // check weather this parameter exists
        if (iter!=muonParameters.end())
        {
            if ((*iter).second[(*iter).second.length()-1]=='\n') (*iter).second[(*iter).second.length()-1]=' ';
            internalTemplateCopy.substitute("content", (*iter).second);
        }
        else
            internalTemplateCopy.removeLineWithContent("$(content)");

        // insert the processed internal template in the muon template file
        muonTemplate.insert("$("+substitutionParameters[i]+")",internalTemplateCopy);
    }

    // save the muon template file
    muonTemplate.save(outputDir_+outputFile);

}


void L1GtVhdlWriterCore::writeConditionChipSetup(std::map<std::string, L1GtVhdlTemplateFile> templates, const std::map<std::string, std::string> &common, const unsigned short int &chip)
{
    std::string filename;
    if (chip==1)
        filename="cond1_chip.vhd";
    else filename="cond2_chip.vhd";

    L1GtVhdlTemplateFile myTemplate = openVhdlFileWithCommonHeader(filename,filename);

    //open substitution parameters:
    //$(calo_common)
    //$(esums_common)
    //$(jet_cnts_common)
    //$(muon_common)
    //$(calo)
    //$(esums)
    //$(jet_cnts)
    //$(muon

    std::map<std::string, L1GtVhdlTemplateFile>::iterator iter= templates.begin();

    while (iter != templates.end())
    {

        myTemplate.insert("$("+(iter->first)+")",iter->second);

        iter++;
    }

    std::map<std::string, std::string>::const_iterator iter2= common.begin();

    while (iter2 != common.end())
    {

        myTemplate.substitute((iter2->first),iter2->second);

        iter2++;
    }

    myTemplate.save(outputDir_+filename);

}


void L1GtVhdlWriterCore::writeAlgoSetup(std::map<int, std::string> &algorithmsChip1, std::map<int, std::string> &algorithmsChip2)
{
    const std::string filename1="pre_algo_and_or_cond1.vhd";
    const std::string filename2="pre_algo_and_or_cond2.vhd";

    L1GtVhdlTemplateFile templ1,templ2;

    templ1 = openVhdlFileWithCommonHeader(filename1,filename1);
    templ2 = openVhdlFileWithCommonHeader(filename2,filename2);

    //open substitution parameters: $(algo-logic)

    std::ostringstream bufferChip1, bufferChip2;

    for (unsigned int i=1; i<96; i++)
    {
        // process chip 1
        bufferChip1<<"pre_algo_a("<<i<<")";
        if (algorithmsChip1[i]!="")
            bufferChip1<<" <= "<<algorithmsChip1[i]<<std::endl;
        else bufferChip1<<" <= '0';"<<std::endl;

        // process chip 2
        bufferChip2<<"pre_algo_a("<<i<<")";
        if (algorithmsChip2[i]!="")
            bufferChip2<<" <= "<<algorithmsChip2[i]<<std::endl;
        else bufferChip2<<" <= '0';"<<std::endl;

    }

    templ1.substitute("prealgos",bufferChip1.str());
    templ2.substitute("prealgos",bufferChip2.str());

    templ1.save(outputDir_+filename1);
    templ2.save(outputDir_+filename2);

}


void L1GtVhdlWriterCore::writeEtmSetup(std::string &etmString,  const int &condChip)
{

    std::string filename="etm_setup.vhd";
    std::string outputFile = filename;

    L1GtVhdlTemplateFile myTemplate;

    // modify output filename
    myTemplate.findAndReplaceString(outputFile,".",int2str(condChip)+".");

    myTemplate = openVhdlFileWithCommonHeader(filename,filename);

    // replace all occurances of $(particle)
    while (myTemplate.substitute("particle", "etm")) while (myTemplate.substitute("particle", "etm"));

    // delete last char if it is \n
    if (etmString[etmString.length()-1] == '\n' ) etmString = etmString.substr(0,etmString.length()-1);

    myTemplate.substitute("phi", etmString);

    myTemplate.save(outputDir_+outputFile);

}


void L1GtVhdlWriterCore::printCommonHeader()
{
    commonHeader_.print();
}


L1GtVhdlTemplateFile L1GtVhdlWriterCore::retrunCommonHeader()
{
    return commonHeader_;
}


void L1GtVhdlWriterCore::writeQsfSetupFiles(const std::string &version)
{

    std::vector<std::string> filenames;
    filenames.push_back("cond1_chip.qsf");
    filenames.push_back("cond2_chip.qsf");

    for (unsigned int i=0; i<filenames.size(); i++)
    {
        L1GtVhdlTemplateFile myTemplate;

        myTemplate.open(vhdlDir_+"Templates/"+filenames.at(i));
        myTemplate.substitute("version", version);

        myTemplate.save(outputDir_+filenames.at(i));
    }

}


void  L1GtVhdlWriterCore::msg(const std::string &message)
{
    internMessageBuf_.push_back(message);
}


std::vector<std::string> L1GtVhdlWriterCore::getMsgBuf()
{
    return internMessageBuf_;
}


bool L1GtVhdlWriterCore::getIntVal(const std::map<std::string,int> &map, const std::string &searchValue, int &intVal)
{
    std::map<std::string,int>::const_iterator iter = map.find(searchValue);
    if( iter == map.end() ) return false;
    intVal = (*iter).second;
    return true;
}


std::string L1GtVhdlWriterCore::index4CondChipVhd(int intval)
{
    intval++;
    return int2str(intval);
}


void L1GtVhdlWriterCore::addJetCountsToCond2IntMap(const int chip, const std::vector<ConditionMap> &conditionMap, std::map<std::string,int> &conditionToIntegerMap)
{
    ConditionMap jetConditions;

    returnConditionsOfOneClass(TypeJetCounts, CondJetCounts,JetCounts,conditionMap.at(chip-1), jetConditions);
    /*
    int counter = 0;

    for (ConditionMap::const_iterator iterCond =  jetConditions.begin(); iterCond !=  jetConditions.end(); iterCond++)
    {
        conditionToIntegerMap[iterCond->first]=counter;
        counter++;
        msg(iterCond->first);
    }

    */
    for(unsigned int i= 0; i<12; i++)
    {
        int counter = 0;

        for (ConditionMap::const_iterator iterCond =  jetConditions.begin(); iterCond !=  jetConditions.end(); iterCond++)
        {

            L1GtJetCountsTemplate* jetsTemplate = static_cast<L1GtJetCountsTemplate*>(iterCond->second);

            const std::vector<L1GtJetCountsTemplate::ObjectParameter>* op =  jetsTemplate->objectParameter();

            conditionToIntegerMap[iterCond->first]=counter;

            if ((*op)[0].countIndex==i) counter++;

            //msg(int2str((*op)[0].countIndex));

        }

        std::string initstr="CONSTANT nr_jet_cnts_"+int2str(i)+"_cond : integer := "+int2str(counter)+";";
        numberOfConditions_.at(chip-1).push_back(initstr);

    }

}


void L1GtVhdlWriterCore::writeCond2intMap2File(const std::map<std::string,int> &conditionToIntegerMap)
{
    std::string filename=outputDir_+"cond_names_integer.txt";

    std::ofstream outputFile(filename.c_str());

    for (std::map<std::string,int>::const_iterator iterCond =  conditionToIntegerMap.begin(); iterCond !=  conditionToIntegerMap.end(); iterCond++)
    {
        outputFile<<(iterCond->first)<<": "<<(iterCond->second)<<std::endl;
    }

}


void L1GtVhdlWriterCore::writeCondChipPkg(const int &chip)
{

    // build filename
    std::string filename="cond"+int2str(chip)+"_chip_pkg.vhd";
    L1GtVhdlTemplateFile myTemplate = openVhdlFileWithCommonHeader(filename,filename);
    myTemplate.insert("$(conditions_nr)",numberOfConditions_.at(chip-1));
    myTemplate.save(outputDir_+filename);
}


void L1GtVhdlWriterCore::countCondsAndAdd2NumberVec(const L1GtConditionType &type, const L1GtConditionCategory &category, const L1GtObject &object, const ConditionMap &map, ConditionMap &outputMap, const int &condChip)
{

    // Get the absoloute amount of conditions
    std::string initstr="CONSTANT nr_"+objType2Str_[object]+"_"+condType2Str_[type]+" : integer := ";

    if (returnConditionsOfOneClass(type,category,object,map, outputMap))
        initstr+=(int2str(outputMap.size())+";");
    else
        initstr+=("0;");

    numberOfConditions_.at(condChip-1).push_back(initstr);

}


void L1GtVhdlWriterCore::initializeDeltaConditions()
{

    for (unsigned int k=0; k<=1; k++)
    {

        // combine muon with calo particles
        for (unsigned int i=0; i<caloObjects_.size(); i++)
        {
            std::string initstr="CONSTANT nr_"+objType2Str_[Mu]+"_"+objType2Str_[caloObjects_.at(i)]+" : integer := 0;";
            numberOfConditions_.at(k).push_back(initstr);
        }

        // combine etm with muon
        numberOfConditions_.at(k).push_back("CONSTANT nr_etm_muon : integer := 0;");

        // combine etm with calo particles
        for (unsigned int i=0; i<caloObjects_.size(); i++)
        {
            std::string initstr="CONSTANT nr_etm_"+objType2Str_[caloObjects_.at(i)]+" : integer := 0;";
            numberOfConditions_.at(k).push_back(initstr);
        }

        std::vector<L1GtObject> caloObjectsCp = caloObjects_;

        while(caloObjectsCp.size()>0)
        {
            std::vector<L1GtObject>::iterator iter=caloObjectsCp.begin();
            L1GtObject firstPartner = (*iter);
            caloObjectsCp.erase(iter);

            iter=caloObjectsCp.begin();
            while (iter!=caloObjectsCp.end())
            {
                std::string initstr="CONSTANT nr_"+objType2Str_[firstPartner]+"_"+objType2Str_[(*iter)]+" : integer := 0;";
                numberOfConditions_.at(k).push_back(initstr);
                iter++;
            }

        }

    }
}


void L1GtVhdlWriterCore::printConditionsOfCategory(const L1GtConditionCategory &category, const ConditionMap &map)
{
    int counter =0;
    for (ConditionMap::const_iterator iterCond =  map.begin(); iterCond !=  map.end(); iterCond++)
    {
        msg(iterCond->first);
        counter++;
    }

    msg("Total Occurances: "+int2str(counter));

}
