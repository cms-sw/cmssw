#ifndef DTSegment_DTRecSegment2DReader_h
#define DTSegment_DTRecSegment2DReader_h

/** \class DTRecSegment2DReader
 *
 * Description:
 *  
 * detailed description
 *
 * $Date:  $
 * $Revision: $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
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

/* Class DTRecSegment2DReader Interface */

class DTRecSegment2DReader : public edm::EDAnalyzer {

  public:

/// Constructor
    DTRecSegment2DReader(const edm::ParameterSet& pset) ;

/// Destructor
    virtual ~DTRecSegment2DReader() ;

/* Operations */ 
    void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  protected:

  private:
    bool debug;
    std::string theRootFileName;
    TFile* theFile;
    //static std::string theAlgoName;
    std::string theRecHits2DLabel;
  
   TH1F *hPositionX;

};
#endif // DTSegment_DTRecSegment2DReader_h

