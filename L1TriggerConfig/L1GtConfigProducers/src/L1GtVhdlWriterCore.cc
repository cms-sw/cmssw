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
 * $Date$
 * $Revision$
 *
 */

// this class header
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlWriterCore.h"

#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"

// system include files
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

L1GtVhdlWriterCore::L1GtVhdlWriterCore(const std::string &templatesDirectory, const std::string &outputDirectory)
{
	templatesPath_=templatesDirectory;
	outputPath_=outputDirectory;
}


L1GtVhdlTemplateFile L1GtVhdlWriterCore::openVhdlFileWithCommonHeader(const std::string &filename, const std::string &outputFilename)
{
	L1GtVhdlTemplateFile commonHeaderCp;
	commonHeaderCp = commonHeader_;

	commonHeaderCp.substitute("vhdl_file_name",outputFilename);

	L1GtVhdlTemplateFile myTemplate(templatesPath_+"Templates/"+filename);
	myTemplate.insert("$(header)", commonHeaderCp);

	return myTemplate;
}


std::string L1GtVhdlWriterCore::int2str(const int &integerValue)
{
	std::ostringstream oss;
	oss<<integerValue;
	return oss.str();

}


std::string L1GtVhdlWriterCore::buildCommonHeader(std::map<std::string,std::string> &headerParameters, const std::vector<std::string> &connectedChannels)
{
	commonHeader_.open(templatesPath_+"InternalTemplates/header");

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

	return "Common Header build successfully!";
}


std::string L1GtVhdlWriterCore::buildMuonSetupVhdl(std::map<std::string,std::string> &muonParameters, const std::string &particle, unsigned short int &condChip)
{
	std::string filename;

	L1GtVhdlTemplateFile muonTemplate,internalTemplate,internalTemplateCopy;

	// choose the correct template file name
	if (particle=="muon")
	{
		filename = "muon_setup.vhd";
		internalTemplate.open(templatesPath_+"InternalTemplates/muonbody",true);
	}
	else
	{
		filename = "calo_setup.vhd";
		internalTemplate.open(templatesPath_+"InternalTemplates/calobody",true);
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
	muonTemplate.save(outputPath_+outputFile);

	return("end");
}


std::string L1GtVhdlWriterCore::buildConditionChipSetup(std::map<std::string, L1GtVhdlTemplateFile> templates, const std::map<std::string, std::string> &common, const unsigned short int &chip)
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

	myTemplate.save(outputPath_+filename);

	return "end";
}


std::string L1GtVhdlWriterCore::buildPreAlgoAndOrCond(std::map<int, std::string> &algorithmsChip1, std::map<int, std::string> &algorithmsChip2)
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

	templ1.save(outputPath_+filename1);
	templ2.save(outputPath_+filename2);

	return "end";
}


std::string L1GtVhdlWriterCore::buildEtmSetup(std::string &etmString,  const int &condChip)
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


	myTemplate.save(outputPath_+outputFile);

	return "end";
}


void L1GtVhdlWriterCore::printCommonHeader()
{
	commonHeader_.print();
}
