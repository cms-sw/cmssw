#ifndef DTSegment_DTRecSegment4DReader_h
#define DTSegment_DTRecSegment4DReader_h

/** \class DTRecSegment4DReader
 *
 * Description:
 *  
 * detailed description
 *
 * $Date:  $
 * $Revision: $
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include "FWCore/Framework/interface/EDAnalyzer.h"
namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

/* Collaborating Class Declarations */
#include "FWCore/Framework/interface/Handle.h"
class TFile;
class TH1F;

/* C++ Headers */
#include <iostream>

/* ====================================================================== */

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

