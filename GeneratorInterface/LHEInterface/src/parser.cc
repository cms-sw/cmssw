// MCDB API: parser.cpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
//
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#include "GeneratorInterface/LHEInterface/src/parser.h"

using namespace mcdb;

using std::string;
using std::vector;
using std::auto_ptr;

HepmlParser::HepmlParser():
    processNs_(false),reportErrors_(false),errorCode_(0)
{}

HepmlParser::~HepmlParser()
{}


int HepmlParser::errorCode()
{
    return errorCode_;
}

void HepmlParser::errorCode(int code)
{
    errorCode_ = code;
}

bool& HepmlParser::processNs()
{
    return processNs_;
}


bool& HepmlParser::processNs(const bool b)
{
    processNs_=b; return processNs_;
}


bool& HepmlParser::reportErrors()
{
    return reportErrors_;
}


bool& HepmlParser::reportErrors(const bool b)
{
    reportErrors_=b; return reportErrors_;
}


auto_ptr<HepmlParser> HepmlParser::newParser()
{
    return getHepmlParser();
}

