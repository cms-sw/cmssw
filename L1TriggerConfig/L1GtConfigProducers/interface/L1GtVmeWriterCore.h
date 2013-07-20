#ifndef L1GtConfigProducers_L1GtVmeWriterCore_h
#define L1GtConfigProducers_L1GtVmeWriterCore_h

/**
 * \class L1GtVmeWriterCore
 * 
 * 
 * Description: core class to write the VME xml file.  
 *
 * Implementation:
 *    Core class to write the VME xml file: L1GtVmeWriter is an EDM wrapper for this class.
 *    L1GtVmeWriterCore can also be used in L1 Trigger Supervisor framework,  with another 
 *    wrapper - it must be therefore EDM-framework free.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date: 2013/05/23 16:50:08 $
 * $Revision: 1.4 $
 *
 */

// system include files
#include <string>
#include <vector>

// user include files
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtVhdlTemplateFile.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtFwd.h"

// forward declarations

// class declaration
class L1GtVmeWriterCore : public L1GtXmlParserTags
{

public:

    /// constructor
    L1GtVmeWriterCore(const std::string& outputDir,
            const std::string& vmeXmlFile);

    /// destructor
    virtual ~L1GtVmeWriterCore();

    void writeVME(const std::vector<ConditionMap> &conditionMap,
            const std::map<std::string,int>& cond2intMap, const L1GtVhdlTemplateFile& header,  const int spacesPerLevel=2);

    /// opens a new xml tag 
    std::string openTag(const std::string &tag);

    /// closes xml tag
    std::string closeTag(const std::string &tag);

    /// returns a string containing spaces dependant on level
    std::string spaces(const unsigned int &level);

    /// builds a address value block
    std::string vmeAddrValueBlock(const std::string &addr, const int &val,
            const int &spaceLevel, const bool setMsb=false);

    /// conversion algorithm for condition index to hex value 
    /// used to calculate address values
    int condIndex2reg(const unsigned int &index);

    /// calculates address
    std::string calculateAddress(const L1GtObject &obj,
            const L1GtConditionType &type, const std::string &reg,
            const int &index);

    /// calculates addresses for jets counts
    std::string calculateJetsAddress(const int &countIndex, const int &obj,
            const int &index);

private:

    /// output directory
    std::string m_outputDir;

    /// output file
    std::string m_vmeXmlFile;

    std::map<L1GtObject,int> object2reg_;

    std::map<L1GtConditionType,int> type2reg_;

    std::map<std::string,int> reg2hex_;

    std::map<int,int> jetType2reg_;

    std::map<int,int> jetObj2reg_;
    
    int spacesPerLevel_;

};

#endif /*L1GtConfigProducers_L1GtVmeWriterCore_h*/
