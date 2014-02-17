#ifndef DTSegment_DTRecSegment2DReader_h
#define DTSegment_DTRecSegment2DReader_h

/** \class DTRecSegment2DReader
 *
 * Description:
 *  
 * detailed description
 *
 * $Date: 2007/03/10 16:14:43 $
 * $Revision: 1.2 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
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

