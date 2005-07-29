#ifndef PHYSICSTOOLS_DSTPREOUTPUT_H
#define PHYSICSTOOLS_DSTPREOUTPUT_H
/*----------------------------------------------------------

 $Id$

The module DSTPreOutput is only required in order to 
store copy of DST's as output of an analysis job.

It is probably not suitable for this package, and should be
moved elsewhere.

------------------------------------------------------------*/
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <map>
#include <string>
#include <vector>

namespace phystools {

  class DSTPreOutput : public edm::EDAnalyzer {
  public:
    DSTPreOutput( const edm::ParameterSet & );
  private:
    void analyze( const edm::Event& e, const edm::EventSetup& );
    typedef std::map<std::string, std::vector<std::string> > tagmap;
    tagmap tags;
    const static std::string names[];
    enum { tracks = 0, pixeltracks, basictracks, vertices, muons };
    template<typename T>
    void get( const edm::Event & , const std::vector<std::string> & );
  };

}

#endif
 
