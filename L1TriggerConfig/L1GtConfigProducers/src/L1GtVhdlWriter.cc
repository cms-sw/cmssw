/**
 * \class L1GtVhdlWriter
 *
 *
 * Description: write VHDL templates for the L1 GT.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author: Philipp Wagner
 *
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriter.h"

// system include files
#include <iomanip>
#include <sstream>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <boost/cstdint.hpp>

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1GtFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtJetCountsTemplate.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterCore.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterBitManager.h"

#include "CondFormats/L1TObjects/interface/L1GtStableParameters.h"

// forward declarations

// constructor(s)
L1GtVhdlWriter::L1GtVhdlWriter(const edm::ParameterSet& parSet)
{

	// directory in /data for the VHDL templates
	vhdlDir_ = parSet.getParameter<std::string>("VhdlTemplatesDir");
	outputDir_ = parSet.getParameter<std::string>("OutputDir");

	if (vhdlDir_[vhdlDir_.length()-1]!= '/') vhdlDir_+="/";
	if (outputDir_[outputDir_.length()-1]!= '/') outputDir_+="/";

	//    // def.xml file
	//    std::string defXmlFileName = parSet.getParameter<std::string>("DefXmlFile");
	//
	//    edm::FileInPath f1("L1TriggerConfig/L1GtConfigProducers/data/" +
	//                       vhdlDir + "/" + defXmlFileName);
	//
	//    m_defXmlFile = f1.fullPath();

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
	condType2Str_[TypeETM]="TypeETM";
	condType2Str_[TypeETT]="TypeETT";
	condType2Str_[TypeHTT]="TypeHTT";
	condType2Str_[TypeJetCounts]="TypeJetCounts";

	caloObjects_.push_back(IsoEG);
	caloObjects_.push_back(NoIsoEG);
	caloObjects_.push_back(CenJet);
	caloObjects_.push_back(ForJet);
	caloObjects_.push_back(TauJet);

	esumObjects_.push_back(HTT);
	esumObjects_.push_back(ETT);
	esumObjects_.push_back(ETM);

	/*
	edm::LogInfo("L1GtConfigProducers")
		<< "\n\nL1 GT VHDL directory: "
		<< vhdlDir
		<< "\n\n"
		<< std::endl;
	*/

}


// destructor
L1GtVhdlWriter::~L1GtVhdlWriter()
{
	// empty
}


bool L1GtVhdlWriter::returnConditionsOfOneClass(const L1GtConditionType &type,  const L1GtConditionCategory &category, const L1GtObject &object, const ConditionMap &map, ConditionMap &outputMap)
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


bool L1GtVhdlWriter::findObjectType(const L1GtObject &object,  ConditionMap &map)
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


void L1GtVhdlWriter::buildMuonParameterMap(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
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

		returnConditionsOfOneClass(muonConditionTypes.at(i),CondMuon,Mu,conditionMap.at(condChip-1), MuonConditions1s);

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


bool L1GtVhdlWriter::buildCaloParameterMap(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
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

		returnConditionsOfOneClass(caloConditionTypes.at(i),CondCalo,caloObject,conditionMap.at(condChip-1), caloConditions);

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


bool L1GtVhdlWriter::buildEnergySumParameter(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
std::string &energySumParameter, const std::vector<ConditionMap> &conditionMap)
{
	unsigned int counter=0;

	for (ConditionMap::const_iterator iterCond = conditionMap.at(condChip-1).begin(); iterCond != conditionMap.at(condChip-1).end(); iterCond++)
	{

		if (iterCond->second->condCategory()==CondEnergySum)
		{
			L1GtEnergySumTemplate* energySumTempl = static_cast<L1GtEnergySumTemplate*>(iterCond->second);
			const std::vector<L1GtEnergySumTemplate::ObjectParameter>* op =  energySumTempl->objectParameter();

			if (bm_.buildPhiEnergySum(op,1,counter)!="") energySumParameter+=bm_.buildPhiEnergySum(op,1,counter);
			counter++;
		}
	}
	return true;
}


bool L1GtVhdlWriter::buildCommonParameter(L1GtVhdlTemplateFile &particle,const L1GtObject &object, const L1GtConditionCategory &category, std::string &parameterStr, const ConditionMap &conditionMap)
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


bool L1GtVhdlWriter::buildCondChipParameters(const unsigned short int &condChip, std::map<std::string,int> &conditionToIntegerMap,
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
				L1GtVhdlTemplateFile muoncopy = muon;

				muoncopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );

				std::ostringstream intVal;

				intVal<<conditionToIntegerMap[(iterCond->first)];

				muoncopy.substitute("ser_no",intVal.str());

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
				muoncopy.findAndReplaceString(tempStr,"$(ser_no)",intVal.str());
				muoncopy.findAndReplaceString(tempStr,"$(type)",condType2Str_[(iterCond->second)->condType()]);

				muoncopy.append(tempStr);

				muoncopy.removeEmptyLines();
				
				// add the processed internal template to parameter map
				templates["muon"].append(muoncopy);

			}
			break;

			case CondCalo:
			{
				L1GtVhdlTemplateFile calocopy = calo;

				std::ostringstream intVal;
				intVal<<conditionToIntegerMap[(iterCond->first)];
				calocopy.substitute("ser_no",intVal.str());

				calocopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );
				calocopy.substitute("particle",objType2Str_[((iterCond->second)->objectType()).at(0)] );
				calocopy.substitute("name",iterCond->first);
				calocopy.substitute("ser_no",intVal.str());
				calocopy.substitute("calo_nr",caloType2Int_[((iterCond->second)->objectType()).at(0)]);

				templates["calo"].append(calocopy);

			}
			break;

			case CondEnergySum:
			{
				L1GtVhdlTemplateFile esumscopy = esums;

				std::ostringstream intVal;
				intVal<<conditionToIntegerMap[(iterCond->first)];
				esumscopy.substitute("ser_no",intVal.str());

				esumscopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );
				esumscopy.substitute("particle",objType2Str_[((iterCond->second)->objectType()).at(0)] );
				esumscopy.substitute("name",iterCond->first);
				esumscopy.substitute("ser_no",intVal.str());

				if (((iterCond->second)->objectType()).at(0)==ETM )
					esumscopy.substitute("if_etm_then_1_else_0","1");
				else
					esumscopy.substitute("if_etm_then_1_else_0","0");

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

				L1GtVhdlTemplateFile jetcopy = jet;

				std::ostringstream intVal;
				intVal<<conditionToIntegerMap[(iterCond->first)];
				jetcopy.substitute("ser_no",intVal.str());

				jetcopy.substitute("type",condType2Str_[(iterCond->second)->condType()] );
				jetcopy.substitute("name",iterCond->first);
												  //templates["muon"].print();
				jetcopy.substitute("ser_no",intVal.str());

				std::string tempStr = jetcopy.returnParameterMap()["PREALGO"];

				jetcopy.findAndReplaceString(tempStr,"$(particle)","jet_cnt");
				jetcopy.findAndReplaceString(tempStr,"$(ser_no)",intVal.str());
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


bool L1GtVhdlWriter::processAlgorithmMap(const AlgorithmMap &algorithmMap, std::vector<ConditionMap> &conditionMap,
std::map<std::string,int> &conditionToIntegerMap, std::map<int, std::string> &algorithmsChip1, std::map<int, std::string> &algorithmsChip2)
{
	AlgorithmMap::const_iterator algoIter=algorithmMap.begin();
	
	// loop over algorithm map
	while (algoIter!=algorithmMap.end())
	{
		L1GtVhdlTemplateFile dummy;
		
		// get the logical expression
		std::string logicalExpr = algoIter->second->algoLogicalExpression();
		std::vector<std::string> conditions;

		dummy.getConditionsFromAlgo(logicalExpr,conditions);

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
		}
		
		L1GtStableParameters stableParams;

		int pin = algoIter->second->algoOutputPin(stableParams.gtNumberConditionChips(),
			stableParams.gtPinsOnConditionChip(),
			stableParams.gtOrderConditionChip());
		if (pin<0) pin*=-1;

		if (algoIter->second->algoChipNumber()==0)
			// add one
			algorithmsChip1[pin+1]=logicalExpr;
		else if (algoIter->second->algoChipNumber()==1)
			algorithmsChip2[pin+1]=logicalExpr;
		else ;

		algoIter++;
	}

	return true;

}


bool L1GtVhdlWriter::makeFirmware(L1GtVhdlWriterCore &core, std::vector<ConditionMap> &conditionMap,const AlgorithmMap &algorithmMap)
{
	std::map<std::string,int> conditionToIntegerMap;

	//--------------------------------------build setups-------------------------------------------------------

	// loop over the two condition chips
	for (unsigned short int i =1; i<=2; i++)
	{
		// ----------------------- muon setup -------------------------------------------------------
		
		std::map<std::string,std::string> muonParameters;
		buildMuonParameterMap(i, conditionToIntegerMap,muonParameters,conditionMap);
		core.buildMuonSetupVhdl(muonParameters,"muon",i);

		// ----------------------- calo setup -------------------------------------------------------

		// map containing all calo object types correlated with the strings
		// under which they appear in the firmware
		std::vector<L1GtObject>::iterator caloObjIter = caloObjects_.begin();

		// loop over all calo objects
		while (caloObjIter != caloObjects_.end())
		{
			std::map<std::string,std::string> caloParameters;
			buildCaloParameterMap(i, conditionToIntegerMap,caloParameters,(*caloObjIter),conditionMap);
			core.buildMuonSetupVhdl(caloParameters,objType2Str_[(*caloObjIter)],i);

			caloObjIter++;
		}

		// ----------------------- etm setup ---------------------------------------------------------

		std::string etmParameter;

		buildEnergySumParameter(i, conditionToIntegerMap,etmParameter, conditionMap);
		core.buildEtmSetup(etmParameter, i);

		// --------------------cond chip setup---------------------------------------------------------

		std::map<std::string, L1GtVhdlTemplateFile> templates;
		std::map<std::string, std::string> common;
		buildCondChipParameters(i, conditionToIntegerMap, conditionMap,templates,common);
		core.buildConditionChipSetup(templates,common,i);

	}

	//-------------------------------process algorithms----------------------------------------------------------

	std::map<int, std::string> algorithmsChip1;
	std::map<int, std::string> algorithmsChip2;

	processAlgorithmMap(algorithmMap,conditionMap,conditionToIntegerMap,algorithmsChip1,algorithmsChip2);
	core.buildPreAlgoAndOrCond(algorithmsChip1, algorithmsChip2);

	return true;
}


// loop over events
void L1GtVhdlWriter::analyze(
const edm::Event& iEvent, const edm::EventSetup& evSetup)
{

	edm::ESHandle< L1GtTriggerMenu > l1GtMenu ;
	evSetup.get< L1GtTriggerMenuRcd >().get( l1GtMenu ) ;

	std::vector<ConditionMap> conditionMap = l1GtMenu->gtConditionMap();
	AlgorithmMap algorithmMap = l1GtMenu->gtAlgorithmMap();

	// print with various level of verbosities
	int printVerbosity = 0;
	l1GtMenu->print(std::cout, printVerbosity);

	//---------------------Here the VHDL files will be created---------------------------------------

	// information that will be delivered by the parser in future
	std::map<std::string,std::string> headerParameters;
	std::vector<std::string> channelVector;

	headerParameters["vhdl_path"]="/vhdllibrarypath";
	headerParameters["designer_date"]="20.05.1986";
	headerParameters["designer_name"]="Philipp Wagner";
	headerParameters["version"]="2.0";
	headerParameters["designer_comments"]="This is a test";
	headerParameters["gtl_setup_name"]="L1Menu2007NovGR";

	channelVector.push_back("-- ca1: ieg");
	channelVector.push_back("-- ca2: eg");
	channelVector.push_back("-- ca3: jet");
	channelVector.push_back("-- ca4: fwdjet");
	channelVector.push_back("-- ca5: tau");
	channelVector.push_back("-- ca6: esums");
	channelVector.push_back("-- ca7: jet_cnts");
	channelVector.push_back("-- ca8: free");
	channelVector.push_back("-- ca9: free");
	channelVector.push_back("-- ca10: free");

	// prepare a core with common header
	L1GtVhdlWriterCore core(vhdlDir_, outputDir_);
	core.buildCommonHeader(headerParameters, channelVector);
	
	// generate the firmware
	L1GtVhdlWriter::makeFirmware(core, conditionMap, algorithmMap);

}
