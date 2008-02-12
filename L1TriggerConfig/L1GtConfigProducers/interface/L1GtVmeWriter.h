#ifndef L1GtConfigProducers_L1GtVmeWriter_h
#define L1GtConfigProducers_L1GtVmeWriter_h

/**
 * \class L1GtVmeWriter
 * 
 * 
 * Description: write the VME xml file.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
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
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

// forward declarations

// class declaration
class L1GtVmeWriter : public edm::EDAnalyzer
{

    public:

        /// constructor
        explicit L1GtVmeWriter(const edm::ParameterSet&);

        /// destructor
        virtual ~L1GtVmeWriter();

        virtual void analyze(const edm::Event&, const edm::EventSetup&);

    private:

        /// output directory
        std::string m_outputDir;
        
        /// output file
        std::string m_vmeXmlFile;

};

#endif /*L1GtConfigProducers_L1GtVmeWriter_h*/
