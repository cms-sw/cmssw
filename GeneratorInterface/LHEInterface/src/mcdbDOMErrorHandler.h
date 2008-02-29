// MCDB API: mcdbDOMErrorHandler.hpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Gene Galkin, IHEP, Protvino, Russia
// e-mail: galkine@ihep.ru, 2006-2007
//
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#ifndef HEPML_DOM_ERROR_HANDLER_
#define HEPML_DOM_ERROR_HANDLER_ 1

#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>

namespace mcdb {

  namespace parser_xs {

    XERCES_CPP_NAMESPACE_USE

    class mcdbDOMErrorHandler: public DOMErrorHandler
    {
     public:
        mcdbDOMErrorHandler();
        ~mcdbDOMErrorHandler();
        bool getSawErrors()  { return fSawErrors;  }
        bool handleError(const DOMError&);
        void resetErrors();
    private:
        mcdbDOMErrorHandler(const mcdbDOMErrorHandler&);
        void operator=(const mcdbDOMErrorHandler&);
        bool fSawErrors;
    };


  } // namespace parser_xs

} // namespace mcdb

#endif

