// MCDB API: mcdb.cpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#include <sstream>
#include "GeneratorInterface/LHEInterface/interface/mcdb.h"
#include "GeneratorInterface/LHEInterface/src/config.h"
#include "GeneratorInterface/LHEInterface/src/parser.h"
#include "GeneratorInterface/LHEInterface/src/macro.h"

namespace mcdb {

using std::string;
using std::vector;


template <class T>
inline std::string to_string(const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}


MCDB::MCDB():
    serverBaseUrl_(MCDB_ARTICLE_BASEURL),
    hepmlProcessNs_(HEPML_PROCESS_NS),
    hepmlReportErrors_(HEPML_REPORT_ERR),
    errorCode_(0) { }

MCDB::MCDB(const string& baseUrl):
    serverBaseUrl_(baseUrl),
    hepmlProcessNs_(HEPML_PROCESS_NS),
    hepmlReportErrors_(HEPML_REPORT_ERR),
    errorCode_(0) { }

MCDB::~MCDB() { }



const Article MCDB::getArticle(const string& Uri)
{
    std::auto_ptr<HepmlParser> p = HepmlParser::newParser();
    p->processNs( hepmlProcessNs() );
    p->reportErrors( hepmlReportErrors() );

    Article a = p->getArticle(Uri);
    errorCode_ = p->errorCode();
    return a;
}

const Article MCDB::getArticle(int id)
{
    return getArticle(serverBaseUrl() + to_string<int>(id));
}


const vector<File> MCDB::getFiles(const string& articleXmlUri)
{
    std::auto_ptr<HepmlParser> p = HepmlParser::newParser();
    p->processNs( hepmlProcessNs() );
    p->reportErrors( hepmlReportErrors() );

    vector<File> f = p->getFiles(articleXmlUri);
    errorCode_ = p->errorCode();
    return f;
}


const vector<File> MCDB::getFiles(int articleId)
{
    return getFiles(serverBaseUrl() + to_string<int>(articleId));
}


FUNC_SETGET(string&, MCDB, serverBaseUrl)
FUNC_SETGET(bool&, MCDB, hepmlProcessNs)
FUNC_SETGET(bool&, MCDB, hepmlReportErrors)
FUNC_GET(int, MCDB, errorCode)


} //namespace mcdb

