#ifndef L1GtConfigProducers_L1GtVhdlWriterBitManager_h
#define L1GtConfigProducers_L1GtVhdlWriterBitManager_h

/**
 * \class L1GtVhdlWriterBitManager
 *
 *
 * \Description This class builds the LUTS for the GT firmware. Furthermore it is providing some helpers
 *  for basic bit operations in binary and hex format.
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

#include "CondFormats/L1TObjects/interface/L1GtCaloTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtMuonTemplate.h"
#include "CondFormats/L1TObjects/interface/L1GtEnergySumTemplate.h"

class L1GtVhdlWriterBitManager
{
    private:
        std::map<std::string, std::string> hex2binMap_ ;

    public:
        L1GtVhdlWriterBitManager();
        std::string bin2hex(std::string binString);
        std::string hex2bin(std::string hexString);
        std::string capitalLetters(std::string hexString);
        std::string mirror(unsigned int offset, std::string hexString, bool hexOutput = true);
        std::string readMapInverse(const std::map<std::string,std::string>& map,std::string value);
        std::string shiftLeft(std::string hexString);
        std::string buildEtaMuon(const std::vector<L1GtMuonTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter);
        std::string buildEtaCalo(const std::vector<L1GtCaloTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter);
        /// \param high is true in order to build phiHigh and false in order to build phiLow
        std::string buildPhiMuon(const std::vector<L1GtMuonTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter, bool high);
        std::string buildPhiCalo(const std::vector<L1GtCaloTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter);
        std::string buildPhiEnergySum(const std::vector<L1GtEnergySumTemplate::ObjectParameter>* op, const unsigned int &num, const unsigned int &counter);
        std::string buildDeltaEtaMuon(const L1GtMuonTemplate::CorrelationParameter* &cp,const unsigned int &counter);
        std::string buildDeltaEtaCalo(const L1GtCaloTemplate::CorrelationParameter* &cp,const unsigned int &counter);
        std::string buildDeltaPhiMuon(const L1GtMuonTemplate::CorrelationParameter* &cp,const unsigned int &counter);
        std::string buildDeltaPhiCalo(const L1GtCaloTemplate::CorrelationParameter* &cp,const unsigned int &counter);

};
#endif                                            /*L1GtConfigProducers_L1GtVhdlWriterBitManager_h*/
