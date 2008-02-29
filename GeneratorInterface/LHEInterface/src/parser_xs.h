// MCDB API: parser_xs.hpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#ifndef HEPML_PARSER_XERCES_
#define HEPML_PARSER_XERCES_ 1

#include "GeneratorInterface/LHEInterface/src/mcdbDOMErrorHandler.h"
#include "GeneratorInterface/LHEInterface/src/parser.h"

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/AbstractDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>

namespace mcdb {

  namespace parser_xs {

    using namespace mcdb;
    using std::auto_ptr;

    class HepmlParserXs: public HepmlParser {
     public:
        HepmlParserXs();
        ~HepmlParserXs();
        virtual const Article getArticle(const string& Uri);
        virtual const vector<File> getFiles(const string& Uri);
    };


  } // namespace parser_xs

} // namespace mcdb

#endif

