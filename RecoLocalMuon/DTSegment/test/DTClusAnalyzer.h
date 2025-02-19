#ifndef DTCLUSANALYZER_H
#define DTCLUSANALYZER_H

/** \class DTClusAnalyzer
 *
 * Description:
 *  
 *  detailed description
 *
 * \author : Stefano Lacaprara - INFN LNL <stefano.lacaprara@pd.infn.it>
 * $date   : 06/05/2008 16:33:43 CEST $
 *
 * Modification:
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
#include "DataFormats/Common/interface/Handle.h"
class TFile;
class TH1F;
class TH2F;

/* C++ Headers */
#include <iosfwd>

/* ====================================================================== */

/* Class DTClusAnalyzer Interface */

class DTClusAnalyzer : public edm::EDAnalyzer {

  public:

/* Constructor */ 
    DTClusAnalyzer(const edm::ParameterSet& pset) ;

/* Destructor */ 
    ~DTClusAnalyzer() ;

/* Operations */ 
    void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  private:
    TH1F* histo(const std::string& name) const;
    TH2F* histo2d(const std::string& name) const;

  private:
    bool debug;
    int _ev;
    std::string theRootFileName;
    TFile* theFile;

    std::string theRecClusLabel;     
    std::string theRecHits2DLabel;     
    std::string theRecHits1DLabel;     
  protected:

};
#endif // DTCLUSANALYZER_H

