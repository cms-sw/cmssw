#ifndef L1GtConfigProducers_L1GtTriggerMenuXmlParser_h
#define L1GtConfigProducers_L1GtTriggerMenuXmlParser_h

/**
 * \class L1GtTriggerMenuXmlParser
 * 
 * 
 * Description: Xerces-C XML parser for the L1 Trigger menu.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date:$
 * $Revision:$
 *
 */

// system include files

#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>

// user include files

// forward declarations

// class declaration
class L1GtTriggerMenuXmlParser
{

public:

    /// constructor
    L1GtTriggerMenuXmlParser();

    /// destructor
    virtual ~L1GtTriggerMenuXmlParser();

private:

    /// error handler for xml-parser
    XERCES_CPP_NAMESPACE::ErrorHandler* m_xmlErrHandler;


};

#endif /*L1GtConfigProducers_L1GtTriggerMenuXmlParser_h*/
