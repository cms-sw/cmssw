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
#include <sys/stat.h>
#include <algorithm>

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
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlDefinitions.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"

// constructor(s)
L1GtVhdlWriterCore::L1GtVhdlWriterCore(const std::string &templatesDirectory,
        const std::string &outputDirectory, const bool &debug)
{

    // set templates directory
    vhdlDir_=templatesDirectory;

    // set output directory
    outputDir_=outputDirectory;

    // Get maps
    L1GtVhdlDefinitions maps;

    objType2Str_=maps.getObj2StrMap();
    caloType2Int_=maps.getCalo2IntMap();
    condType2Str_=maps.getCond2StrMap();

    // fill calo object vector
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

bool L1GtVhdlWriterCore::returnConditionsOfOneClass(
        const L1GtConditionType &type, const L1GtConditionCategory &category,
        const L1GtObject &object,
        const ConditionMap &map,
        ConditionMap &outputMap)
{
    bool status = false;

    ConditionMap::const_iterator condIter = map.begin();
    while (condIter!=map.end())
    {
        if (condIter->second->condCategory()==category
                && condIter->second->condType()==type && (condIter->second->objectType())[0]==object)
        {
            outputMap[condIter->first]=condIter->second;
            status = true;
        }
        condIter++;
    }

    return status;
}

bool L1GtVhdlWriterCore::findObjectType(const L1GtObject &object,
        ConditionMap &map)
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

        if (status)
            break;
        condIter++;
    }

    return status;
}

void L1GtVhdlWriterCore::getMuonSetupContentFromTriggerMenu(
        const unsigned short int &condChip,
        std::map<std::string,std::string> &muonParameters)
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

        countCondsAndAdd2NumberVec(muonConditionTypes.at(i), CondMuon, Mu,
                (*conditionMap_).at(condChip-1), MuonConditions1s, condChip);

        /*
         // Get the absoloute amount of conditions
         std::string initstr="CONSTANT nr_"+objType2Str_[Mu]+"_"+condType2Str_[muonConditionTypes.at(i)]+" : integer := ";

         if (returnConditionsOfOneClass(muonConditionTypes.at(i),CondMuon,Mu,conditionMap_.at(condChip-1), MuonConditions1s))
         initstr+=(int2str(MuonConditions1s.size())+";");
         else
         initstr+=("0;");

         numberOfConditions_.at(condChip-1).push_back(initstr);

         */

        //std::cout<< std::hex << (*myObjectParameter).at(0).etThreshold << std::endl;
        unsigned int counter=0;

        for (ConditionMap::const_iterator iterCond = MuonConditions1s.begin(); iterCond
                != MuonConditions1s.end(); iterCond++)
        {

            // add this condition to name -> integer conversion map
            conditionToIntegerMap_[iterCond->first]=counter;

            L1GtMuonTemplate* m_gtMuonTemplate =
                    static_cast<L1GtMuonTemplate*>(iterCond->second);
            const std::vector<L1GtMuonTemplate::ObjectParameter>* op =
                    m_gtMuonTemplate->objectParameter();

            if (muonConditionTypes.at(i)==Type1s)
            {

                // build eta
                muonParameters["eta_1_s"] += (bm_.buildEtaMuon(op, 1, counter));

                // add the parameters to parameter map
                muonParameters["phi_h_1_s"]+= bm_.buildPhiMuon(op, 1, counter,
                        true);
                muonParameters["phi_l_1_s"]+=bm_.buildPhiMuon(op, 1, counter,
                        false);

                if (debugMode_)
                {
                    muonParameters["phi_l_1_s"]+=("--"+iterCond->first+"\n");
                    muonParameters["phi_h_1_s"]+=("--"+iterCond->first+"\n");
                    muonParameters["eta_1_s"] +=("--"+iterCond->first+"\n");
                }

            } else

            if (muonConditionTypes.at(i)==Type2s)
            {

                // build eta
                muonParameters["eta_2_s"] += (bm_.buildEtaMuon(op, 2, counter));

                // add the parameters to parameter map
                muonParameters["phi_h_2_s"]+= bm_.buildPhiMuon(op, 2, counter,
                        true);
                muonParameters["phi_l_2_s"]+=bm_.buildPhiMuon(op, 2, counter,
                        false);

                if (debugMode_)
                {
                    muonParameters["phi_l_2_s"]+=("--"+iterCond->first+"\n");
                    muonParameters["phi_h_2_s"]+=("--"+iterCond->first+"\n");
                    muonParameters["eta_2_s"] +=("--"+iterCond->first+"\n");
                }

            } else
            //m_gtMuonTemplate->print(std::cout);

            if (muonConditionTypes.at(i)==Type3s)
            {

                // build eta
                muonParameters["eta_3"] += (bm_.buildEtaMuon(op, 3, counter));

                // add the parameters to parameter map
                muonParameters["phi_h_3"]+= bm_.buildPhiMuon(op, 3, counter,
                        true);
                muonParameters["phi_l_3"]+=bm_.buildPhiMuon(op, 3, counter,
                        false);

                if (debugMode_)
                {
                    muonParameters["phi_l_3"]+=("--"+iterCond->first+"\n");
                    muonParameters["phi_h_3"]+=("--"+iterCond->first+"\n");
                    muonParameters["eta_3"] +=("--"+iterCond->first+"\n");
                }

            }

            if (muonConditionTypes.at(i)==Type4s)
            {

                // build eta
                muonParameters["eta_4"] += (bm_.buildEtaMuon(op, 4, counter));

                // add the parameters to parameter map
                muonParameters["phi_h_4"]+= bm_.buildPhiMuon(op, 4, counter,
                        true);
                muonParameters["phi_l_4"]+=bm_.buildPhiMuon(op, 4, counter,
                        false);

                if (debugMode_)
                {
                    muonParameters["phi_l_4"]+=("--"+iterCond->first+"\n");
                    muonParameters["phi_h_4"]+=("--"+iterCond->first+"\n");
                    muonParameters["eta_4"] +=("--"+iterCond->first+"\n");
                }

            }

            if (muonConditionTypes.at(i)==Type2wsc)
            {
                const L1GtMuonTemplate::CorrelationParameter* cp =
                        m_gtMuonTemplate->correlationParameter();

                // build eta
                muonParameters["eta_2_wsc"]
                        += (bm_.buildEtaMuon(op, 2, counter));

                // build phi
                muonParameters["phi_h_2_wsc"]+= bm_.buildPhiMuon(op, 2,
                        counter, true);
                muonParameters["phi_l_2_wsc"]+=bm_.buildPhiMuon(op, 2, counter,
                        false);

                // build delta_eta
                std::ostringstream dEta;
                muonParameters["delta_eta"] += bm_.buildDeltaEtaMuon(cp,
                        counter);

                // build delta_phi
                muonParameters["delta_phi"] += bm_.buildDeltaPhiMuon(cp,
                        counter);

                if (debugMode_)
                {
                    muonParameters["eta_2_wsc"]+=("--"+iterCond->first+"\n");
                    muonParameters["phi_h_2_wsc"]+=("--"+iterCond->first+"\n");
                    muonParameters["phi_l_2_wsc"] +=("--"+iterCond->first+"\n");
                    muonParameters["delta_eta"]+=("--"+iterCond->first+"\n");
                    muonParameters["delta_phi"] +=("--"+iterCond->first+"\n");
                }

            }
            counter++;
        }
    }

}

bool L1GtVhdlWriterCore::getCaloSetupContentFromTriggerMenu(
        const unsigned short int &condChip,
        std::map<std::string,std::string> &caloParameters,
        const L1GtObject &caloObject)
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
        countCondsAndAdd2NumberVec(caloConditionTypes.at(i), CondCalo,
                caloObject, (*conditionMap_).at(condChip-1), caloConditions,
                condChip);

        for (ConditionMap::const_iterator iterCond = caloConditions.begin(); iterCond
                != caloConditions.end(); iterCond++)
        {

            // add this condition to name -> integer conversion map
            conditionToIntegerMap_[iterCond->first]=counter;

            L1GtCaloTemplate* m_gtCaloTemplate =
                    static_cast<L1GtCaloTemplate*>(iterCond->second);
            const std::vector<L1GtCaloTemplate::ObjectParameter>* op =
                    m_gtCaloTemplate->objectParameter();

            if (caloConditionTypes.at(i)==Type1s)
            {

                //= static_cast<std::vector<L1GtCaloTemplate::ObjectPaenergySumParameter+=rameter*> >(op);

                // build eta
                caloParameters["eta_1_s"] += (bm_.buildEtaCalo(op, 1, counter));

                caloParameters["phi_1_s"]+= bm_.buildPhiCalo(op, 1, counter);

                if (debugMode_)
                {
                    caloParameters["eta_1_s"]+=("--"+iterCond->first+"\n");
                    caloParameters["phi_1_s"]+=("--"+iterCond->first+"\n");
                }

            } else if (caloConditionTypes.at(i)==Type2s)
            {

                // build eta
                caloParameters["eta_2_s"] += (bm_.buildEtaCalo(op, 2, counter));

                // add the parameters to parameter map
                caloParameters["phi_2_s"]+= bm_.buildPhiCalo(op, 2, counter);

                if (debugMode_)
                {
                    caloParameters["eta_2_s"]+=("--"+iterCond->first+"\n");
                    caloParameters["phi_2_s"]+=("--"+iterCond->first+"\n");
                }

            } else if (caloConditionTypes.at(i)==Type4s)
            {
                // build eta
                caloParameters["eta_4"] += (bm_.buildEtaCalo(op, 4, counter));

                // add the parameters to parameter map
                caloParameters["phi_4"]+= bm_.buildPhiCalo(op, 4, counter);

                if (debugMode_)
                {
                    caloParameters["eta_4"]+=("--"+iterCond->first+"\n");
                    caloParameters["phi_4"]+=("--"+iterCond->first+"\n");
                }

            } else if (caloConditionTypes.at(i)==Type2wsc)
            {
                const L1GtCaloTemplate::CorrelationParameter* cp =
                        m_gtCaloTemplate->correlationParameter();

                // build eta
                caloParameters["eta_2_wsc"]
                        += (bm_.buildEtaCalo(op, 2, counter));

                // build phi
                caloParameters["phi_2_wsc"]+= bm_.buildPhiCalo(op, 2, counter);

                // build delta_eta
                caloParameters["delta_eta"] += bm_.buildDeltaEtaCalo(cp,
                        counter);

                // build delta_phi
                caloParameters["delta_phi"] += bm_.buildDeltaPhiCalo(cp,
                        counter);

                if (debugMode_)
                {
                    caloParameters["eta_2_wsc"]+=("--"+iterCond->first+"\n");
                    caloParameters["phi_2_wsc"]+=("--"+iterCond->first+"\n");
                    caloParameters["delta_eta"]+=("--"+iterCond->first+"\n");
                    caloParameters["delta_phi"]+=("--"+iterCond->first+"\n");
                }

            }

            counter++;

        }

    }

    return true;
}

bool L1GtVhdlWriterCore::getEsumsSetupContentFromTriggerMenu(
        const unsigned short int &condChip, const L1GtObject &object,
        std::string &energySumParameter)
{

    L1GtConditionType type;

    unsigned int counter=0;

    if (object==HTT)
        type= TypeHTT;
    else if (object==ETM)
        type= TypeETM;
    else if (object==ETT)
        type= TypeETT;

    ConditionMap esumsConditions;
    // stores all conditions of type given in the first three parameters
    countCondsAndAdd2NumberVec(type, CondEnergySum, object,
            (*conditionMap_).at(condChip-1), esumsConditions, condChip);

    for (ConditionMap::const_iterator iterCond = esumsConditions.begin(); iterCond
            != esumsConditions.end(); iterCond++)
    {
        L1GtEnergySumTemplate* energySumTempl =
                static_cast<L1GtEnergySumTemplate*>(iterCond->second);
        const std::vector<L1GtEnergySumTemplate::ObjectParameter>* op =
                energySumTempl->objectParameter();

        if (bm_.buildPhiEnergySum(op, 1, counter)!="")
            energySumParameter+=bm_.buildPhiEnergySum(op, 1, counter);

        if (debugMode_)
        {
            energySumParameter+=("--"+iterCond->first+"\n");
        }

        conditionToIntegerMap_[(iterCond->first)]=counter;

        counter++;
    }

    return true;

}

bool L1GtVhdlWriterCore::getSubstParamCommonFromTriggerMenu(const unsigned short int &condChip, L1GtVhdlTemplateFile &particle,
        const L1GtObject &object, const L1GtConditionCategory &category,
        std::string &parameterStr)
{

    std::map<L1GtConditionType,std::string> condType2Strcopy = condType2Str_;

    // make a copy since types will deleted later after they are processed
    std::map<L1GtConditionType,std::string>::iterator typeIterator =
            condType2Strcopy.begin();

    while (typeIterator != condType2Strcopy.end())
    {

        ConditionMap outputMap;
        
        if (returnConditionsOfOneClass((*typeIterator).first, category, object, (*conditionMap_).at(condChip-1), outputMap))
        {

            // special treatment for jet counts in buildCodChipParameters
            if (object !=JetCounts)
            {
                std::string tempStr2 = particle.returnParameterMap()[stringConstantCommon_];
                while (particle.findAndReplaceString(tempStr2,
                        sp(substParamParticle_), objType2Str_[object]))
                    ;
                while (particle.findAndReplaceString(tempStr2,
                        sp(substParamType_), condType2Str_[(*typeIterator).first]))
                    ;
                parameterStr+=(tempStr2+"\n\n");

                //parameterStr+=(tempStr2+"<= '0';\n");

                // remove this type since it was already processed
                condType2Strcopy.erase(typeIterator);
            }

        }

        typeIterator++;

    }

    return true;
}

bool L1GtVhdlWriterCore::getCondChipVhdContentFromTriggerMenu(
        const unsigned short int &condChip,
        std::map<std::string, L1GtVhdlTemplateFile> &templates,
        std::map<std::string, std::string> &commonParams)
{

    // open the necessary internal templates
    L1GtVhdlTemplateFile muon, calo, esums, jet;

    muon.open(vhdlDir_+"InternalTemplates/muon", true);
    muon.substitute("particle", "muon");

    calo.open(vhdlDir_+"InternalTemplates/calo", true);
    esums.open(vhdlDir_+"InternalTemplates/esums", true);
    jet.open(vhdlDir_+"InternalTemplates/jet_cnts", true);

    /*
     charge1s.open(vhdlDir_+"InternalTemplates/charge_1s",false);
     charge2s.open(vhdlDir_+"InternalTemplates/charge_2s",false);
     charge2wsc.open(vhdlDir_+"InternalTemplates/charge_2_wsc",false);
     charge3s.open(vhdlDir_+"InternalTemplates/charge_3",false);
     charge4s.open(vhdlDir_+"InternalTemplates/charge_4",false);
     */

    // only for jet_cnts common parameter relevant since common parameter is build in this routine
    std::vector<unsigned int> processedTypes;

    //-------------------------------------build the common parameters-------------------------------------------

    // build $(muon_common)
    getSubstParamCommonFromTriggerMenu(condChip,muon, Mu, CondMuon, commonParams["muon_common"]);

    // build $(calo_common) - loop over all calo objects
    std::vector<L1GtObject>::iterator caloObjIter = caloObjects_.begin();

    while (caloObjIter != caloObjects_.end())
    {

        getSubstParamCommonFromTriggerMenu(condChip, muon, (*caloObjIter), CondCalo,
                commonParams["calo_common"]);
        caloObjIter++;

    }

    // build $(esums_common) - loop over all esums objects
    std::vector<L1GtObject>::iterator esumObjIter = esumObjects_.begin();

    while (esumObjIter != esumObjects_.end())
    {

        getSubstParamCommonFromTriggerMenu(condChip, esums, (*esumObjIter), CondEnergySum,
                commonParams["esums_common"]);
        esumObjIter++;

    }

    //--------------------------------------build the parameter maps----------------------------------------------

    // loop over condition map
    for (ConditionMap::const_iterator iterCond = (*conditionMap_).at(condChip-1).begin(); iterCond != (*conditionMap_).at(condChip-1).end(); iterCond++)
    {

        switch ((iterCond->second)->condCategory())
        {

        case CondMuon:
        {

            int cond2int;
            if (!getIntVal(conditionToIntegerMap_, (iterCond->first), cond2int))
            {
                msg("Panik! Condition "+(iterCond->first)
                        +" does not have a integer equivalent!");
                break;
            }

            std::string intVal = index4CondChipVhd(cond2int);

            L1GtVhdlTemplateFile muoncopy = muon;

            muoncopy.substitute("type", condType2Str_[(iterCond->second)->condType()]);

            muoncopy.substitute("ser_no", intVal);

            muoncopy.substitute("name", iterCond->first);

            if ((iterCond->second)->condType() == Type1s)
            {
                muoncopy.substitute(substParamCharge_,
                        muon.getInternalParameter(stringConstantCharge1s_));
            } else if ((iterCond->second)->condType() == Type2s)
            {
                muoncopy.substitute(substParamCharge_,
                        muon.getInternalParameter(stringConstantCharge2s_));
            } else if ((iterCond->second)->condType() == Type2wsc)
            {
                muoncopy.substitute(substParamCharge_,
                        muon.getInternalParameter(stringConstantCharge2wsc_));
            } else if ((iterCond->second)->condType() == Type3s)
            {
                muoncopy.substitute(substParamCharge_,
                        muon.getInternalParameter(stringConstantCharge3s_));
            } else if ((iterCond->second)->condType() == Type4s)
            {
                muoncopy.substitute(substParamCharge_,
                        muon.getInternalParameter(stringConstantCharge4s_));
            }

            std::string tempStr = muoncopy.returnParameterMap()["PREALGO"];

            muoncopy.findAndReplaceString(tempStr, "$(particle)", "muon");
            muoncopy.findAndReplaceString(tempStr, "$(ser_no)", intVal);
            muoncopy.findAndReplaceString(tempStr, "$(type)", condType2Str_[(iterCond->second)->condType()]);

            muoncopy.append(tempStr);

            muoncopy.removeEmptyLines();

            // add the processed internal template to parameter map
            templates["muon"].append(muoncopy);

        }
            break;

        case CondCalo:
        {
            int cond2int;

            if (!getIntVal(conditionToIntegerMap_, (iterCond->first), cond2int))
            {
                msg("Panik! Condition "+(iterCond->first)
                        +" does not have a integer equivalent!");
                break;
            }

            std::string intVal = index4CondChipVhd(cond2int);

            L1GtVhdlTemplateFile calocopy = calo;

            calocopy.substitute("type", condType2Str_[(iterCond->second)->condType()]);
            calocopy.substitute("particle", objType2Str_[((iterCond->second)->objectType()).at(0)]);
            calocopy.substitute("name", iterCond->first);
            calocopy.substitute("ser_no", intVal);
            calocopy.substitute("calo_nr", caloType2Int_[((iterCond->second)->objectType()).at(0)]);

            // builds something like tau_1_s(20));
            calocopy.append(objType2Str_[((iterCond->second)->objectType()).at(0)] +"_"+condType2Str_[(iterCond->second)->condType()]+"("+intVal+"));");

            templates["calo"].append(calocopy);

        }
            break;

        case CondEnergySum:
        {

            int cond2int;

            if (!getIntVal(conditionToIntegerMap_, (iterCond->first), cond2int))
            {
                msg("Panik! Condition "+(iterCond->first)
                        +" does not have a integer equivalent!");
                break;
            }

            std::string intVal = index4CondChipVhd(cond2int);

            L1GtVhdlTemplateFile esumscopy = esums;

            esumscopy.substitute("type", condType2Str_[(iterCond->second)->condType()]);
            esumscopy.substitute("particle", objType2Str_[((iterCond->second)->objectType()).at(0)]);
            esumscopy.substitute("name", iterCond->first);
            esumscopy.substitute("ser_no", intVal);

            if (((iterCond->second)->objectType()).at(0)==ETM)
                esumscopy.substitute("if_etm_then_1_else_0", "1");
            else
                esumscopy.substitute("if_etm_then_1_else_0", "0");

            // builds something like htt_cond(4));
            esumscopy.append(objType2Str_[((iterCond->second)->objectType()).at(0)] +"_"+condType2Str_[(iterCond->second)->condType()]+"("+ intVal+"));");

            templates["esums"].append(esumscopy);
        }

            break;

        case CondJetCounts:
        {

            // build the common parameter for the jet counts

            L1GtJetCountsTemplate* jetsTemplate =
                    static_cast<L1GtJetCountsTemplate*>(iterCond->second);

            int nObjects = iterCond->second->nrObjects();

            const std::vector<L1GtJetCountsTemplate::ObjectParameter>* op =
                    jetsTemplate->objectParameter();

            for (int i = 0; i < nObjects; i++)
            {

                std::vector<unsigned int>::iterator p = find(
                        processedTypes.begin(), processedTypes.end(), (*op)[i].countIndex);

                // check, weather this count index was already processed
                // and process it if not
                if (p==processedTypes.end())
                {
                    std::ostringstream indStr;
                    indStr<<(*op)[i].countIndex;

                    std::string tempStr2 = jet.returnParameterMap()[stringConstantCommon_];
                    while (jet.findAndReplaceString(tempStr2,
                            sp(substParamParticle_), objType2Str_[JetCounts]))
                        ;
                    while (jet.findAndReplaceString(tempStr2,
                            sp(substParamType_), indStr.str()))
                        ;
                    commonParams[substParamJetCntsCommon_]+=tempStr2+"\n"; // +"<= '0';\n");
                    processedTypes.push_back((*op)[i].countIndex);
                }

            }

            int cond2int;

            if (!getIntVal(conditionToIntegerMap_, (iterCond->first), cond2int))
            {
                msg("Panik! Condition "+(iterCond->first)
                        +" does not have a integer equivalent!");
                break;
            }

            std::string intVal = index4CondChipVhd(cond2int);

            L1GtVhdlTemplateFile jetcopy = jet;

            jetcopy.substitute("particle", condType2Str_[(iterCond->second)->condType()]);
            jetcopy.substitute("type", int2str((*op)[0].countIndex));
            jetcopy.substitute("name", iterCond->first);

            jetcopy.substitute("ser_no", intVal);

            // builds the final string
            std::string tempStr = jetcopy.returnParameterMap()["PREALGO"];

            jetcopy.findAndReplaceString(tempStr, "$(particle)", "jet_cnt");
            jetcopy.findAndReplaceString(tempStr, "$(ser_no)", intVal);
            jetcopy.findAndReplaceString(tempStr, "$(type)", int2str((*op)[0].countIndex));

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

    while (pIt!=commonParams.end())
    {
        if ((pIt->second)[((pIt->second).length()-1)] == '\n')
            (pIt->second) = (pIt->second).substr(0, (pIt->second).length()-1);
        pIt++;
    }

    return true;

}

bool L1GtVhdlWriterCore::processAlgorithmMap(std::vector< std::map<int, std::string> > &algoStrings)
{
    std::map<int, std::string> algorithmsChip1;
    std::map<int, std::string> algorithmsChip2;
    
    AlgorithmMap::const_iterator algoIter=(*algorithmMap_).begin();

    // loop over algorithm map
    while (algoIter!=(*algorithmMap_).end())
    {
        // msg(algoIter->first);

        L1GtVhdlTemplateFile dummy;

        // get the logical expression
        std::string logicalExpr = (algoIter->second).algoLogicalExpression();
        std::vector<std::string> conditions;

        dummy.getConditionsFromAlgo(logicalExpr, conditions);

        std::string logicalExprCopy= logicalExpr;

        // loop over all condition names appearing in algorithm and replace them by type and integer value
        for (unsigned int i=0; i<conditions.size(); i++)
        {
            std::ostringstream newExpr;
            
            // look for the condition on correct chip
            ConditionMap  const& chip = conditionMap_->at(algoIter->second.algoChipNumber());
            L1GtCondition const* cond = (chip.find(conditions.at(i)) == chip.end()) ? nullptr : chip.at(conditions.at(i));

            // check weather condition exists
            if (cond!=NULL)
            {
                newExpr << objType2Str_[(cond->objectType()).at(0)];
                newExpr << "_" << condType2Str_[cond->condType()] << "(";
                newExpr << conditionToIntegerMap_[conditions.at(i)] << ")";
                dummy.findAndReplaceString(logicalExpr, conditions.at(i),
                        newExpr.str());
            } else
                msg("Panik! Didn't find Condition "+conditions.at(i));
        }

        // has to be checked!
        std::vector<int> orderConditionChip;
        orderConditionChip.push_back(2);
        orderConditionChip.push_back(1);

        int pin = (algoIter->second).algoOutputPin(2, 96, orderConditionChip);

        //msg(int2str(pin));

        if (pin<0)
            pin*=-1;

        if ((algoIter->second).algoChipNumber()==0)
        {
            algorithmsChip1[pin]=logicalExpr;
            if (debugMode_)
                algorithmsChip1[pin]+=("-- "+logicalExprCopy);
        } else if ((algoIter->second).algoChipNumber()==1)
        {
            algorithmsChip2[pin]=logicalExpr;
            if (debugMode_)
                algorithmsChip2[pin]+=("-- "+logicalExprCopy);
        } else
            ;

        algoIter++;
    }

    algoStrings.push_back(algorithmsChip1);
    algoStrings.push_back(algorithmsChip2);
    
    return true;

}

std::string L1GtVhdlWriterCore::chip2OutputSubDir(const int &chip)
{
    if (chip==1)
        return outputDir_+"/"+outputSubDir1_+"/";
    if (chip==2)
        return outputDir_+"/"+outputSubDir2_+"/";
    
    return "";
}


bool L1GtVhdlWriterCore::makeFirmware(const std::vector<ConditionMap> &conditionMap,
        const AlgorithmMap &algorithmMap)
{
    conditionMap_ = &conditionMap;

    algorithmMap_ = &algorithmMap;
    
    std::string subDir1 = outputDir_+outputSubDir1_;
    std::string subDir2  = outputDir_+outputSubDir2_;
    
    if (!mkdir(subDir1.c_str(), 666));
    if (!mkdir(subDir2.c_str(), 666));
    
    chmod(subDir1.c_str(), 0777);
    chmod(subDir2.c_str(), 0777);
    
    /*
    subDirs_.push_back(subDir1);
    subDirs_.push_back(subDir2);
    */
    
    writeQsfSetupFiles(version_);
    
    
    //--------------------------------------build setups-------------------------------------------------------

    // loop over the two condition chips
    for (unsigned short int i =1; i<=2; i++)
    {
        // ----------------------- muon setup -------------------------------------------------------

        std::map<std::string,std::string> muonParameters;
        getMuonSetupContentFromTriggerMenu(i, muonParameters);
        writeMuonSetupVhdl(muonParameters, "muon", i);

        // ----------------------- calo setup -------------------------------------------------------

        // map containing all calo object types correlated with the strings
        // under which they appear in the firmware
        std::vector<L1GtObject>::iterator caloObjIter = caloObjects_.begin();

        // loop over all calo objectsbuildMuonSetupVhdl
        while (caloObjIter != caloObjects_.end())
        {
            std::map<std::string,std::string> caloParameters;
            getCaloSetupContentFromTriggerMenu(i, caloParameters,
                    (*caloObjIter));
            writeMuonSetupVhdl(caloParameters, objType2Str_[(*caloObjIter)], i);

            caloObjIter++;
        }

        // ----------------------- etm setup ---------------------------------------------------------

        // map containing all calo object types correlated with the strings
        // under which they appear in the firmware
        std::vector<L1GtObject>::iterator esumObjIter = esumObjects_.begin();

        while (esumObjIter != esumObjects_.end())
        {
            std::string etmParameter;

            getEsumsSetupContentFromTriggerMenu(i, (*esumObjIter),
                    etmParameter);
            writeEtmSetup(etmParameter, i);

            esumObjIter++;

        }

        // add jet counts to condition 2 integer map
        addJetCountsToCond2IntMap(i, (*conditionMap_), conditionToIntegerMap_);

        // --------------------cond chip setup---------------------------------------------------------
        // Important: all other setups have to be build BEFORE this one because it needs a
        // complete condition2integer map !!!

        std::map<std::string, L1GtVhdlTemplateFile> templates;
        std::map<std::string, std::string> common;
        getCondChipVhdContentFromTriggerMenu(i, templates, common);
        writeConditionChipSetup(templates, common, i);

        // despite yet not existing they have to appear in cond_chip_pkg vhds
        initializeDeltaConditions();

        // --------------------cond chip pkg------------------------------------------------------------

        writeCondChipPkg(i);

        // -----------------------def val pkg -------------------------------------------------

        writeDefValPkg((*conditionMap_), i);

        // debug
        if (debugMode_)
        {
            //printConditionsOfCategory(CondEnergySum, (*conditionMap_).at(i-1));
        }

    }

    if (debugMode_)
        writeCond2intMap2File();

    //-------------------------------process algorithms----------------------------------------------------------

    std::vector< std::map<int, std::string> > algoStrings;

    processAlgorithmMap(algoStrings);
    
    writeAlgoSetup(algoStrings);

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

void L1GtVhdlWriterCore::buildCommonHeader(
        std::map<std::string,std::string> &headerParameters,
        const std::vector<std::string> &connectedChannels)
{
    commonHeader_.open(vhdlDir_+"InternalTemplates/header");

    std::map<std::string,std::string>::iterator iter = headerParameters.begin();

    // loop over the header parameter map and replace all subsitution
    // parameters with their values
    while (iter != headerParameters.end() )
    {
        commonHeader_.substitute((*iter).first, (*iter).second);
        iter++;
    }

    commonHeader_.insert("$(connected_channels_1)", connectedChannels);
    commonHeader_.insert("$(connected_channels_2)", connectedChannels);

    //---------------------build the Quartus configuration files----------------------------------------

    //writeQsfSetupFiles(headerParameters["version"]);
    
    version_=headerParameters["version"];

    msg("Build Common header and Quartus setup files sucessuflly!");

}

L1GtVhdlTemplateFile L1GtVhdlWriterCore::openVhdlFileWithCommonHeader(
        const std::string &filename, const std::string &outputFilename)
{
    L1GtVhdlTemplateFile commonHeaderCp;
    commonHeaderCp = commonHeader_;

    commonHeaderCp.substitute("vhdl_file_name", outputFilename);

    L1GtVhdlTemplateFile myTemplate(vhdlDir_+"Templates/"+filename);
    myTemplate.insert("$(header)", commonHeaderCp);

    return myTemplate;
}

void L1GtVhdlWriterCore::writeMuonSetupVhdl(
        std::map<std::string,std::string> &muonParameters,
        const std::string &particle, unsigned short int &condChip)
{
    std::string filename;

    L1GtVhdlTemplateFile muonTemplate, internalTemplate, internalTemplateCopy;

    // choose the correct template file name
    if (particle=="muon")
    {
        filename = vhdlTemplateMuonSetup_;
        internalTemplate.open(vhdlDir_+"InternalTemplates/muonsetup", true);
    } else
    {
        filename = vhdlTemplateCaloSetup_;
        internalTemplate.open(vhdlDir_+"InternalTemplates/calosetup", true);
    }

    std::string outputFile;
    outputFile = filename;

    // modify filename
    if (particle!="muon")
        muonTemplate.findAndReplaceString(outputFile, "calo", particle);

    // add condition chip index to output filename
    //muonTemplate.findAndReplaceString(outputFile, ".", int2str(condChip)+".");

    // open the template file and insert header
    muonTemplate = openVhdlFileWithCommonHeader(filename, outputFile);

    std::map<std::string,std::string> parameterMap =
            internalTemplate.returnParameterMap();

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

            if (substitutionParameters.at(i).substr(0, 3) == "eta")
                internalTemplateCopy.substitute("constant", parameterMap["eta"]);
            else if (substitutionParameters.at(i).substr(0, 3) == "phi")
                internalTemplateCopy.substitute("constant", parameterMap["phi"]);
            else if (substitutionParameters.at(i).substr(0, 3) == "del"/*ta*/)
            {
                internalTemplateCopy.substitute("constant",
                        parameterMap["delta"]);
                while (internalTemplateCopy.substitute("delta",
                        substitutionParameters[i]))
                    internalTemplateCopy.substitute("delta",
                            substitutionParameters[i]);

            }
        }

        if (particle=="muon")
        {
        // final preparation of the internal template before it is inserted in the actual template file
        internalTemplateCopy.substitute("type", substitutionParameters[i]);
        } else
            
        {
            internalTemplateCopy.substitute("type", substitutionParameters[i].substr(4)); 
        }

        // subsitute the second occurance of type without "_l" and "_h"

        std::string paramCopy = substitutionParameters[i];

        internalTemplateCopy.findAndReplaceString(paramCopy, "_l", "");
        internalTemplateCopy.findAndReplaceString(paramCopy, "_h", "");

        internalTemplateCopy.substitute("type2", paramCopy);

        internalTemplateCopy.substitute("others",
                parameterMap[substitutionParameters[i]]);

        // replace all the occurances of "particle"
        while (muonTemplate.substitute("particle", particle))
            muonTemplate.substitute("particle", particle);

        // remove the the parameter $(content) if its empty
        iter=muonParameters.find(substitutionParameters[i]);

        // check weather this parameter exists
        if (iter!=muonParameters.end())
        {
            if ((*iter).second[(*iter).second.length()-1]=='\n')
                (*iter).second[(*iter).second.length()-1]=' ';
            internalTemplateCopy.substitute("content", (*iter).second);
        } else
            internalTemplateCopy.removeLineWithContent("$(content)");

        // insert the processed internal template in the muon template file
        muonTemplate.insert("$("+substitutionParameters[i]+")",
                internalTemplateCopy);
    }

    // save the muon template file
    muonTemplate.save(chip2OutputSubDir(condChip)+outputFile);
    chmod((chip2OutputSubDir(condChip)+outputFile).c_str(), 0666);

}

void L1GtVhdlWriterCore::writeConditionChipSetup(
        const std::map<std::string, L1GtVhdlTemplateFile>& templates,
        const std::map<std::string, std::string> &common,
        const unsigned short int &chip)
{

    // get filename 
    std::string filename = vhdlTemplateCondChip_;

    // build output filename
    std::string outputFileName = filename;
    L1GtVhdlTemplateFile::findAndReplaceString(outputFileName, ".",
            int2str(chip)+".");

    L1GtVhdlTemplateFile myTemplate = openVhdlFileWithCommonHeader(filename,
            outputFileName);

    // map containing the subsitution parameters with their content (as L1GtVhdlTemplateFile file object)
    std::map<std::string, L1GtVhdlTemplateFile>::const_iterator iter=
            templates.begin();

    while (iter != templates.end())
    {

        myTemplate.insert("$("+(iter->first)+")", iter->second);

        iter++;
    }

    // subsitutes common parameters
    std::map<std::string, std::string>::const_iterator iter2= common.begin();

    while (iter2 != common.end())
    {

        myTemplate.substitute((iter2->first), iter2->second);

        iter2++;
    }

    myTemplate.save(chip2OutputSubDir(chip)+filename);
    chmod((chip2OutputSubDir(chip)+filename).c_str(), 0666);

}

void L1GtVhdlWriterCore::writeAlgoSetup(std::vector< std::map<int, std::string> > &algoStrings)
{
    // loop over the two condition chips
    for (unsigned int i=1; i<=2; i++)
    {
        std::string filename=vhdlTemplateAlgoAndOr_;

        // output file name
        std::string outputFileName = filename;
        L1GtVhdlTemplateFile::findAndReplaceString(outputFileName, ".",
                int2str(i)+".");

        L1GtVhdlTemplateFile templateFile = openVhdlFileWithCommonHeader(
                filename, outputFileName);

        std::ostringstream buffer;
        
        unsigned int algoPinsPerChip = 96;

        for (unsigned int k=1; k<=algoPinsPerChip; k++)
        {

            buffer<< stringConstantAlgo_<<"("<<k<<")";
            if (algoStrings.at(i-1)[k]!="")
                buffer<<" <= "<<algoStrings.at(i-1)[k]<<";"<<std::endl;
            else
                buffer<<" <= '0';"<<std::endl;
        }

        templateFile.substitute(substParamAlgos_, buffer.str());
        templateFile.save(chip2OutputSubDir(i)+filename);
        chmod((chip2OutputSubDir(i)+filename).c_str(), 0666);
    }

}

void L1GtVhdlWriterCore::writeEtmSetup(std::string &etmString,
        const int &condChip)
{

    // get filename
    std::string filename= vhdlTemplateEtmSetup_;
    std::string outputFile = filename;

    L1GtVhdlTemplateFile myTemplate;

    // modify output filename
    myTemplate.findAndReplaceString(outputFile, ".", int2str(condChip)+".");

    myTemplate = openVhdlFileWithCommonHeader(filename, filename);

    // replace all occurances of $(particle)
    while (myTemplate.substitute("particle", "etm"))
        while (myTemplate.substitute("particle", "etm"))
            ;

    // delete last char if it is \n
    if (etmString[etmString.length()-1] == '\n')
        etmString = etmString.substr(0, etmString.length()-1);

    myTemplate.substitute("phi", etmString);

    myTemplate.save(chip2OutputSubDir(condChip)+filename);
    chmod((chip2OutputSubDir(condChip)+filename).c_str(), 0666);

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
    filenames.push_back(quartusSetupFileChip1_);
    filenames.push_back(quartusSetupFileChip2_);

    for (unsigned int i=0; i<filenames.size(); i++)
    {
        L1GtVhdlTemplateFile myTemplate;

        myTemplate.open(vhdlDir_+"Templates/"+filenames.at(i));
        myTemplate.substitute("version", version);

        std::string tempStr = filenames.at(i);
        
        if (i==0)
            L1GtVhdlTemplateFile::findAndReplaceString(tempStr , "cond1","cond");
        if (i==1)
            L1GtVhdlTemplateFile::findAndReplaceString(tempStr , "cond2","cond");
        
        myTemplate.save(chip2OutputSubDir(i+1)+tempStr);
        chmod((chip2OutputSubDir(i+1)+tempStr).c_str(), 0666);
    }

}

void L1GtVhdlWriterCore::msg(const std::string &message)
{
    internMessageBuf_.push_back(message);
}

std::vector<std::string> L1GtVhdlWriterCore::getMsgBuf()
{
    return internMessageBuf_;
}

bool L1GtVhdlWriterCore::getIntVal(const std::map<std::string,int> &map,
        const std::string &searchValue, int &intVal)
{
    std::map<std::string,int>::const_iterator iter = map.find(searchValue);
    if (iter == map.end() )
        return false;
    intVal = (*iter).second;
    return true;
}

std::string L1GtVhdlWriterCore::index4CondChipVhd(int intval)
{
    intval++;
    return int2str(intval);
}

void L1GtVhdlWriterCore::addJetCountsToCond2IntMap(const int chip,
        const std::vector<ConditionMap> &conditionMap,
        std::map<std::string,int> &conditionToIntegerMap_)
{
    ConditionMap jetConditions;

    returnConditionsOfOneClass(TypeJetCounts, CondJetCounts, JetCounts,
            conditionMap.at(chip-1), jetConditions);
    /*
     int counter = 0;

     for (ConditionMap::const_iterator iterCond =  jetConditions.begin(); iterCond !=  jetConditions.end(); iterCond++)
     {
     conditionToIntegerMap_[iterCond->first]=counter;
     counter++;
     msg(iterCond->first);
     }

     */
    
    unsigned int maxJetsCountsIndex = 11;
    
    for (unsigned int i= 0; i<=maxJetsCountsIndex; i++)
    {
        int counter = 0;

        for (ConditionMap::const_iterator iterCond = jetConditions.begin(); iterCond
                != jetConditions.end(); iterCond++)
        {

            L1GtJetCountsTemplate* jetsTemplate =
                    static_cast<L1GtJetCountsTemplate*>(iterCond->second);
            const std::vector<L1GtJetCountsTemplate::ObjectParameter>* op =
                    jetsTemplate->objectParameter();

            conditionToIntegerMap_[iterCond->first]=counter;
            if ((*op)[0].countIndex==i)
                counter++;

            //msg(int2str((*op)[0].countIndex));

        }

        numberOfConditions_.at(chip-1).push_back(retNumberOfConditionsString("jet_cnts_"
                +int2str(i)+"_cond", counter));

    }

}

void L1GtVhdlWriterCore::writeCond2intMap2File()
{
   
    
    for (unsigned int i=0; i<=1;i++)
    {
        ConditionMap::const_iterator iterCond =
            conditionMap_->at(i).begin();
       
        std::string filename=chip2OutputSubDir(i+1)+"cond_names_integer.txt";
        
        std::ofstream outputFile(filename.c_str());
        
        while (iterCond!=conditionMap_->at(i).end())
        {
            conditionToIntegerMap_[(*iterCond).first];
            
            outputFile<<(iterCond->first)<<": "<<conditionToIntegerMap_[(*iterCond).first]<<std::endl;
            iterCond++;
        }
        
        outputFile.close();
        
        chmod(filename.c_str(), 0666);
          
        
    }
   
    /*
    
    
    const std::vector<ConditionMap> * conditionMap_;
    
    std::string filename=outputDir_+"cond_names_integer.txt";
    std::ofstream outputFile(filename.c_str());

    for (std::map<std::string,int>::const_iterator iterCond =
            conditionToIntegerMap_.begin(); iterCond
            != conditionToIntegerMap_.end(); iterCond++)
    {
        outputFile<<(iterCond->first)<<": "<<(iterCond->second)<<std::endl;
    }
    
    */

}

std:: string L1GtVhdlWriterCore::buildDefValString(const int &conditionIndex,
        const std::vector<int> &values)
{
    // has to produce something like 1 =>  ("00000000", "00000000")

    return "1 =>  (\"00000000\", \"00000000\")";
}

std::string L1GtVhdlWriterCore::getDefValsFromTriggerMenu(
        const L1GtConditionType &type, const L1GtObject &object,
        const VmeRegister &reg)
{
    L1GtConditionCategory category;

    // get condition category from object

    category = getCategoryFromObject(object);

    ConditionMap conditions;
    returnConditionsOfOneClass(type, category, object, (*conditionMap_).at(0),
            conditions);

    std::string result;

    if (category==CondCalo)
    {
        for (ConditionMap::const_iterator iterCond =conditions.begin(); iterCond
                != conditions.end(); iterCond++)
        {
            L1GtCaloTemplate* caloTemplate =
                    static_cast<L1GtCaloTemplate*>(iterCond->second);
            const std::vector<L1GtCaloTemplate::ObjectParameter>* op =
                    caloTemplate->objectParameter();

            unsigned int nObjects = iterCond->second->nrObjects();

            for (unsigned int i =0; i<nObjects; i++)
            {
                if (i==0)
                    result+="(";
                result +="\"";
                result += int2str((*op).at(i).etThreshold);
                result +="\"";
                if (i!=nObjects-1)
                    result +=",";
                else
                    result+=")\n";

            }
        }
    }

    return "-- 0 => (\"00000000\", \"00000000\" ... )";
    return result;
}

void L1GtVhdlWriterCore::writeDefValPkg(
        const std::vector<ConditionMap> &conditionMap, const int &chip)
{
    // open internal template file
    L1GtVhdlTemplateFile internalTemplate;
    internalTemplate.open(vhdlDir_+"InternalTemplates/defvalpkg", true);

    // to write the def_value_pkg file it has to be looped over all POSSIBLE types of conditions
    // POSSIBLE: calo_3 eg. does not exist and therefore makes no sense..

    // Process muon conditions first

    // temporary buffers for the default value blocks (see internal templates) of the different types.
    // in the end those buffers are simply appended to the actual template!
    L1GtVhdlTemplateFile muonDefValuesBuffer, caloDefValuesBuffer,
            jetCountsDefValuesBuffer, esumsDefValBuffer;

    // Get the default value type strings from internal template
    std::vector<std::string> muonDefValTypes;
    muonDefValTypes.push_back(internalTemplate.getInternalParameter(stringConstantPtl_));
    muonDefValTypes.push_back(internalTemplate.getInternalParameter(stringConstantPth_));
    muonDefValTypes.push_back(internalTemplate.getInternalParameter(stringConstantQuality_));
    muonDefValTypes.push_back(internalTemplate.getInternalParameter(substParamCharge_));

    int jetCountsMaxIndex = 11;

    // fill jet counts vector (from 1 to 11)
    std::vector<std::string> jetCountsDefValTypes;
    for (int i = 0; i<=jetCountsMaxIndex; i++)
    {
        jetCountsDefValTypes.push_back(int2str(i));
    }

    // only one default value for calo objects therefore only a empty string
    std::vector<std::string> caloDefValTypes;
    caloDefValTypes.push_back("");

    // get types of esums defvalues (as firmware strings) from the internal templates
    std::vector<std::string> esumsDefValTypes;
    esumsDefValTypes.push_back(internalTemplate.getInternalParameter(stringConstantEsumsLow_));
    esumsDefValTypes.push_back(internalTemplate.getInternalParameter(stringConstantEsumsHigh_));

    std::map<L1GtConditionType,std::string>::iterator typeIter;

    // prepare a map with all muon relevant types by removing 
    // obsolete types from a copy of condType2Str_ map
    // should be improved
    std::map<L1GtConditionType,std::string> muonTypes = condType2Str_;

    typeIter=muonTypes.find(Type2cor);
    muonTypes.erase(typeIter);
    typeIter=muonTypes.find(TypeETM);
    muonTypes.erase(typeIter);
    typeIter=muonTypes.find(TypeETT);
    muonTypes.erase(typeIter);
    typeIter=muonTypes.find(TypeHTT);
    muonTypes.erase(typeIter);
    typeIter=muonTypes.find(TypeJetCounts);
    muonTypes.erase(typeIter);

    std::map<L1GtConditionType,std::string> caloTypes = muonTypes;
    typeIter=caloTypes.find(Type3s);
    caloTypes.erase(typeIter);

    // dummy type in order to be able to use the same code as for calo and muon
    // this map is also used for esums since there is no difference in treatment 
    std::map<L1GtConditionType,std::string> jetCountsTypes;
    jetCountsTypes[TypeJetCounts] = "";

    // here the DefValuesBuffer are build (=objects of the class L1GtVhdlTemplateFile
    // that are containing all default values and finally can be inserted into the
    // def_val_pkg.vhd template

    buildDefValuesBuffer(muonDefValuesBuffer, muonTypes, muonDefValTypes, Mu);

    // loop over all possible calo objects here
    for (unsigned int i=0; i<caloObjects_.size(); i++)
    {
        buildDefValuesBuffer(caloDefValuesBuffer, caloTypes, caloDefValTypes,
                caloObjects_.at(i));
    }

    // loop over all possible esums objects here
    for (unsigned int i=0; i<esumObjects_.size(); i++)
    {
        buildDefValuesBuffer(esumsDefValBuffer, jetCountsTypes,
                esumsDefValTypes, esumObjects_.at(i));
    }

    // same procedure for jet counts
    buildDefValuesBuffer(esumsDefValBuffer, jetCountsTypes,
            jetCountsDefValTypes, JetCounts);

    //----------------------In this section the actual output file is written----------------

    // now open the actual template file:

    std::string filename = vhdlTemplateDefValPkg_;

    std::string outputFile = filename;

    // modify output filename
    L1GtVhdlTemplateFile::findAndReplaceString(outputFile, ".", int2str(chip)
            +".");
    L1GtVhdlTemplateFile defValTemplate = openVhdlFileWithCommonHeader(
            filename, outputFile);

    // insert the temporary buffers to the template

    // muon default values
    defValTemplate.insert(sp(substParamMuonDefVals_), muonDefValuesBuffer);

    // calo default values
    defValTemplate.insert(sp(substParamCaloDefVals_), caloDefValuesBuffer);

    // esums default values
    defValTemplate.insert(sp(substParamEsumsDefVals_), esumsDefValBuffer);

    // jet counts default values
    defValTemplate.insert(sp(substParamJetsDefVals_), jetCountsDefValuesBuffer);

    // close and save the file
    
    L1GtVhdlTemplateFile::findAndReplaceString(outputFile, "1","");
    L1GtVhdlTemplateFile::findAndReplaceString(outputFile, "2","");
    
    defValTemplate.save(chip2OutputSubDir(chip)+outputFile);
    chmod((chip2OutputSubDir(chip)+outputFile).c_str(), 0666);

}

bool L1GtVhdlWriterCore::buildDefValuesBuffer(L1GtVhdlTemplateFile &buffer,
        const std::map<L1GtConditionType,std::string> &typeList,
        const std::vector<std::string> &defValuesList, const L1GtObject &object)
{
    // type iterator
    std::map<L1GtConditionType,std::string>::const_iterator typeIter;

    for (typeIter=typeList.begin(); typeIter!=typeList.end(); typeIter++)
    {

        for (unsigned int i = 0; i<defValuesList.size(); i++)
        {

            // open a new internale template
            L1GtVhdlTemplateFile internalTemplate;
            internalTemplate.open(vhdlDir_+"InternalTemplates/defvalpkg", true);

            std::string
                    idString =
                            internalTemplate.getInternalParameter(stringConstantDefValId_);

            // The following three steps convert "$(particle)_$(type)_$(defvaltype)_def_val"
            // to e.g. "muon_1s_et_def_val"

            // replace the substiution parameter particle with muon - thereby parameterString is a
            // reference and therefore changed.
            L1GtVhdlTemplateFile::findAndReplaceString(idString,
                    sp(substParamParticle_), objType2Str_[object]);

            // replace the substiution parameter particle with muon
            L1GtVhdlTemplateFile::findAndReplaceString(idString,
                    sp(substParamType_), typeIter->second);

            // replace the substiution parameter defvaltype with muon
            L1GtVhdlTemplateFile::findAndReplaceString(idString,
                    sp(substParamDefValType_), defValuesList.at(i));

            // usage of the subsitute routine:
            // first the substitution parameter, then the content
            internalTemplate.substitute(substParamDefValId_, idString);

            // !here the body of the internale template is build - almost done!

            // The body looks as follows $(def_val), $(max_nr), $(content) and
            // $(others) have to be subsituted 

            /*
             * CONSTANT $(def_val) : $(calo_or_muon)_maxnr$(max_nr)vector8_arr := 
             * (
             * $(content)
             * $(others)
             * );
             */

            // substitute the particle
            internalTemplate.substitute(substParamParticle_,
                    objType2Str_[object]);

            // go on with max number - get the correct expression for max number from the internal template
            internalTemplate.substitute(substParamMaxNr_,
                    internalTemplate.getInternalParameter(substParamMaxNr_+"_"
                            +condType2Str_[typeIter->first]));

            // now we have to choose the correct OTHERS string for the
            // internal template. The identifier for the OTHERS string
            // is the condition type (as firmware string) therefore:

            // we get it as follows
            std::string othersString;

            // jet_cnts are a special case
            if (object==JetCounts || object==HTT || object==ETM || object==ETT )
                othersString
                        = internalTemplate.getInternalParameter(objType2Str_[JetCounts]);
            else
                othersString
                        = internalTemplate.getInternalParameter(typeIter->second);

            internalTemplate.substitute(substParamOthers_, othersString);

            // the actual content, taken from the trigger menu

            std::string content = getDefValsFromTriggerMenu(typeIter->first,
                    object, RegEtThreshold);

            internalTemplate.substitute(substParamContent_, content);

            // Finally the parameter $(calo_or_muon) is left open
            std::string caloOrMuonStr = stringConstantCalo_;
           
            if (object == Mu)
                caloOrMuonStr = objType2Str_[Mu];
            
            internalTemplate.substitute(substParamCaloOrMuon_, caloOrMuonStr);
            
            // The internal template has been processed and now can be added to the buffer..
            buffer.append(internalTemplate);

        }

    }

    return true;
}

void L1GtVhdlWriterCore::countCondsAndAdd2NumberVec(
        const L1GtConditionType &type, const L1GtConditionCategory &category,
        const L1GtObject &object, const ConditionMap &map,
        ConditionMap &outputMap, const int &condChip)
{

    // Get the absoloute amount of conditions

    int number;

    if (returnConditionsOfOneClass(type, category, object, map, outputMap))
        number=outputMap.size();
    else
        number=0;

    std::string initstr = retNumberOfConditionsString(objType2Str_[object]+"_"
            +condType2Str_[type], number);

    numberOfConditions_.at(condChip-1).push_back(initstr);

}

void L1GtVhdlWriterCore::initializeDeltaConditions()
{

    for (unsigned int k=0; k<=1; k++)
    {

        // combine muon with calo particles
        for (unsigned int i=0; i<caloObjects_.size(); i++)
        {
            numberOfConditions_.at(k).push_back(retNumberOfConditionsString(objType2Str_[Mu]
                    +"_"+objType2Str_[caloObjects_.at(i)], 0));
        }

        // combine etm with muon
        numberOfConditions_.at(k).push_back(retNumberOfConditionsString(objType2Str_[ETM]+"_"
                +objType2Str_[Mu], 0));

        // combine etm with calo particles
        for (unsigned int i=0; i<caloObjects_.size(); i++)
        {
            numberOfConditions_.at(k).push_back(retNumberOfConditionsString(objType2Str_[ETM]
                    +"_"+objType2Str_[caloObjects_.at(i)], 0));
        }

        std::vector<L1GtObject> caloObjectsCp = caloObjects_;

        while (caloObjectsCp.size()>0)
        {
            std::vector<L1GtObject>::iterator iter=caloObjectsCp.begin();
            L1GtObject firstPartner = (*iter);
            caloObjectsCp.erase(iter);

            iter=caloObjectsCp.begin();
            while (iter!=caloObjectsCp.end())
            {
                numberOfConditions_.at(k).push_back(retNumberOfConditionsString(
                        objType2Str_[firstPartner]+"_"+objType2Str_[(*iter)], 0));
                iter++;
            }

        }

    }
}

L1GtConditionCategory L1GtVhdlWriterCore::getCategoryFromObject(
        const L1GtObject &object)
{

    L1GtConditionCategory category;

    if (object==Mu)
        category = CondMuon;
    else if (object==ETM || object==HTT || object==ETT)
        category = CondEnergySum;
    else if (object==IsoEG|| object==NoIsoEG || object==CenJet || object
            ==ForJet || object==TauJet)
        category = CondCalo;
    else if (object==IsoEG|| object==NoIsoEG || object==CenJet || object
            ==ForJet || object==TauJet)
        category = CondCalo;
    else if (object==JetCounts)
        category = CondJetCounts;
    else 
        category=CondNull;

    return category;

}

void L1GtVhdlWriterCore::printConditionsOfCategory(
        const L1GtConditionCategory &category, const ConditionMap &map)
{
    int counter =0;
    for (ConditionMap::const_iterator iterCond = map.begin(); iterCond
            != map.end(); iterCond++)
    {
        msg(iterCond->first);
        counter++;
    }

    msg("Total Occurances: "+int2str(counter));

}

void L1GtVhdlWriterCore::writeCondChipPkg(const int &chip)
{

    // build filename

    std::string filename;

    if (chip==1)
        filename = vhdlTemplateCondChipPkg1_;
    else if (chip==2)
        filename = vhdlTemplateCondChipPkg2_;

    // write the output
    L1GtVhdlTemplateFile myTemplate = openVhdlFileWithCommonHeader(filename,
            filename);
    
    myTemplate.substitute("version",version_);
    
    myTemplate.insert("$(conditions_nr)", numberOfConditions_.at(chip-1));
   
    L1GtVhdlTemplateFile::findAndReplaceString(filename, "1","");
    L1GtVhdlTemplateFile::findAndReplaceString(filename, "2","");
    
    myTemplate.save(chip2OutputSubDir(chip)+filename);
    chmod((chip2OutputSubDir(chip)+filename).c_str(), 0666);
}

std::string L1GtVhdlWriterCore::retNumberOfConditionsString(
        const std::string &typeStr, const int &number)
{
    std::string initstr=stringConstantConstantNr_+typeStr+" : integer := "+int2str(number)
            +";";

    return initstr;
}

std::map<std::string,int> L1GtVhdlWriterCore::getCond2IntMap()
{
    return conditionToIntegerMap_;
}

std::string L1GtVhdlWriterCore::sp(const std::string &name)
{
    return "$("+name+")";
}

