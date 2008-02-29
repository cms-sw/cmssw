// MCDB API: parser.hpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
//
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//


#ifndef MCDB_PARSER_HPP_
#define MCDB_PARSER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "GeneratorInterface/LHEInterface/interface/mcdb.h"

namespace mcdb {

using std::string;
using std::vector;
using std::auto_ptr;

class HepmlParser {
public:
    HepmlParser();
    virtual ~HepmlParser();
    virtual const Article getArticle(const string& Uri) = 0;
    virtual const vector<File> getFiles(const string& Uri) = 0;
    static  auto_ptr<HepmlParser> newParser();
    virtual bool& processNs();
    virtual bool& processNs(const bool b);
    virtual bool& reportErrors();
    virtual bool& reportErrors(const bool b);
    int errorCode();
protected:
    void errorCode(int);
private:
    bool processNs_;
    bool reportErrors_;
    int errorCode_;
};

extern auto_ptr<HepmlParser> getHepmlParser();

} // namespace mcdb

#endif // MCDB_PARSER_HPP_

