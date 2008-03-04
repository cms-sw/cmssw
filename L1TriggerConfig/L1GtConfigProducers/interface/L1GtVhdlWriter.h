#ifndef L1GtConfigProducers_L1GtVhdlWriter_h
#define L1GtConfigProducers_L1GtVhdlWriter_h

/**
 * \class L1GtVhdlWriter
 * 
 * 
 * Description: write VHDL templates for the L1 GT.  
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

// user include files
//   base class
#include "FWCore/Framework/interface/EDAnalyzer.h"

class Event;
class EventSetup;
class ParameterSet;


// forward declarations


// class declaration
class L1GtVhdlWriter : public edm::EDAnalyzer
{

public:

    // constructor
    explicit L1GtVhdlWriter(const edm::ParameterSet&);

    // destructor
    virtual ~L1GtVhdlWriter();

    virtual void analyze(const edm::Event&, const edm::EventSetup&);

};

#endif /*L1GtConfigProducers_L1GtVhdlWriter_h*/
