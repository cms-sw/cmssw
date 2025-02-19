#ifndef DTSegment_DTRecSegment4DReader_h
#define DTSegment_DTRecSegment4DReader_h

/** \class DTRecSegment4DReader
 *
 * Description:
 *  
 * detailed description
 *
 * $Date: 2007/03/10 16:14:43 $
 * $Revision: 1.2 $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDAnalyzer.h"

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/Handle.h"

/* C++ Headers */
#include <iostream>

class TFile;
class TH1F;

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

/* Class DTRecSegment4DReader Interface */

class DTRecSegment4DReader : public edm::EDAnalyzer {

  public:

/// Constructor
    DTRecSegment4DReader(const edm::ParameterSet& pset) ;

/// Destructor
    virtual ~DTRecSegment4DReader() ;

/* Operations */ 
    void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  protected:

  private:
    bool debug;
    std::string theRootFileName;
    TFile* theFile;
    //static std::string theAlgoName;
    std::string theRecHits4DLabel;
  
   TH1F *hPositionX;

};
#endif // DTSegment_DTRecSegment4DReader_h

