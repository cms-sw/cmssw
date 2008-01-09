#ifndef L1GtConfigProducers_L1GtVhdlWriterCore_h
#define L1GtConfigProducers_L1GtVhdlWriterCore_h

/**
 * \class L1GtVhdlWriterCore
 *
 *
 * \Description writes the actual VHDL code and stores global information like the common header.
 *  This class is using parameter maps containing the substitution parameters and their content.
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *
 * \author Philipp Wagner
 *
 * $Date$
 * $Revision$
 *
 */

#include "L1GtVhdlTemplateFile.h"

// system include files

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

class L1GtVhdlWriterCore
{
	private:
		std::string templatesPath_;
		std::string outputPath_;
		L1GtVhdlTemplateFile commonHeader_;

	public:
		L1GtVhdlWriterCore(const std::string &templatesDirectory, const std::string &outputDirectory);
		L1GtVhdlTemplateFile openVhdlFileWithCommonHeader(const std::string &filename, const std::string &outputFilename);
		std::string buildCommonHeader(std::map<std::string,std::string> &headerParameters , const std::vector<std::string> &connectedChannels);
		std::string buildMuonSetupVhdl(std::map<std::string,std::string> &muonParameters, const std::string &particle, unsigned short int &condChip);
		std::string buildConditionChipSetup(std::map<std::string, L1GtVhdlTemplateFile> templates, const std::map<std::string, std::string> &common, const unsigned short int &chip);
		std::string buildPreAlgoAndOrCond(std::map<int, std::string> &algorithmsChip1, std::map<int, std::string> &algorithmsChip2);
		std::string buildEtmSetup(std::string &etmString, const int &condChip);
		std::string int2str(const int &integerValue);
		void printCommonHeader();

};
#endif											  /*L1GtConfigProducers_L1GtVhdlWriterCore_h*/
