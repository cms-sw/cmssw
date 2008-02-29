// MCDB API: file.cpp
// LCG MCDB project, Monte Carlo Data Base
// http://mcdb.cern.ch
// 
// Sergey Belov <Sergey.Belov@cern.ch>, 2007
//

#include "GeneratorInterface/LHEInterface/interface/mcdb.h"
#include "GeneratorInterface/LHEInterface/src/macro.h"

namespace mcdb {

using std::string;
using std::vector;

File::File(): eventsNumber_(0), crossSectionPb_(0),
              csErrorPlusPb_(0), csErrorMinusPb_(0),
	      size_(0), type_(any), id_(0) { }

File::~File() { }

FUNC_SETGET(FileType&, File, type)
FUNC_SETGET(int&, File, id)
FUNC_SETGET(int&, File, eventsNumber)
FUNC_SETGET(unsigned long&, File, size)
FUNC_SETGET(string&, File, checksum)
FUNC_SETGET(float&, File, crossSectionPb)
FUNC_SETGET(float&, File, csErrorPlusPb)
FUNC_SETGET(float&, File, csErrorMinusPb)
FUNC_SETGET(string&, File, comments)
FUNC_SETGET(vector<string>&, File, paths)


vector<string> File::findPaths(const string& substr)
{
    using namespace std;

    vector<string>::iterator p = paths_.begin();
    vector<string> result;
    
    for (; p!=paths_.end(); p++) {
        if ( ((string)*p).find(substr) != string::npos ) {
	    result.push_back((string)*p);
	}
    }

    return result;
}

} //namespace mcdb
