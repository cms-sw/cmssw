// MCDB API: mcdbDOMErrorHandler.cpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Gene Galkin, IHEP, Protvino, Russia
// e-mail: galkine@ihep.ru, 2006-2007
//
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#include <iostream>
#include <cstdlib>
#include "GeneratorInterface/LHEInterface/src/mcdbDOMErrorHandler.h"


namespace mcdb {

namespace parser_xs {


XERCES_CPP_NAMESPACE_USE

class StrX
{
public:
    StrX(const XMLCh* const toTranscode)
    {
        pLocalForm_ = XMLString::transcode(toTranscode);
    }
    ~StrX()
    {
        XMLString::release(&pLocalForm_);
    }
    const char* localForm() const
    {
        return pLocalForm_;
    }
private:
    char* pLocalForm_;
};

inline std::ostream& operator<<(std::ostream& target, const StrX& toDump)
{
    target << toDump.localForm();
    return target;
}

mcdbDOMErrorHandler::mcdbDOMErrorHandler(): fSawErrors(false)
{
}

mcdbDOMErrorHandler::~mcdbDOMErrorHandler()
{
}

bool mcdbDOMErrorHandler::handleError(const DOMError& domError)
{
    using namespace std;

    fSawErrors = true;
    
    if(domError.getSeverity() == DOMError::DOM_SEVERITY_WARNING)
        cerr << endl << "Warning at file ";
    else if (domError.getSeverity() == DOMError::DOM_SEVERITY_ERROR)
        cerr << endl << "Error at file ";
    else cerr << "\nFatal Error at file ";
    
    cerr << StrX(domError.getLocation()->getURI()) << ", line " <<
            domError.getLocation()->getLineNumber() << ", char " <<
            domError.getLocation()->getColumnNumber() << endl <<" Message: " <<
            StrX(domError.getMessage()) << endl;
 return(true);
}

void mcdbDOMErrorHandler::resetErrors()
{
    fSawErrors = false;
}



} //namespace parser_xs

} //namespace mcdb

