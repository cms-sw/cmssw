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
 * $Date$
 * $Revision$
 *
 */

// system include files
#include <string>

// user include files
//   base class
#include "L1TriggerConfig/L1GtConfigProducers/interface/L1GtXmlParserTags.h"


// forward declarations

// class declaration
class L1GtVmeWriterCore : public L1GtXmlParserTags
{

    public:

        /// constructor
        L1GtVmeWriterCore(const std::string& outputDir, const std::string& vmeXmlFile);

        /// destructor
        virtual ~L1GtVmeWriterCore();

        void writeVME();

    private:

        /// output directory
        std::string m_outputDir;
        
        /// output file
        std::string m_vmeXmlFile;

};

#endif /*L1GtConfigProducers_L1GtVmeWriterCore_h*/
