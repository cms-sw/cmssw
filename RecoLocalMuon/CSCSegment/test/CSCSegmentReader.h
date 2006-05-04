#ifndef RecoLocalMuon_CSCSegmentReader_H
#define RecoLocalMuon_CSCSegmentReader_H

/** \class CSCSegmentReader
 *  Basic analyzer class which accesses CSCSegment
 *  and plot efficiency of the builder
 *
 *  $Date: 2006/04/20 16:15:36 $
 *  $Revision: 1.3 $
 *  \author M. Sani - FNAL
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>

#include <vector>
#include <map>
#include <string>

#include "TFile.h"
#include "TH1F.h"

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class CSCSegmentReader : public edm::EDAnalyzer {
public:
  /// Constructor
  CSCSegmentReader(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~CSCSegmentReader();

  // Operations

  /// Perform the real analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);


protected:

private: 
    std::string label1;
    std::string filename;
    TH1F* h, *h2;
    TH1I* h3;
    
    TFile* file;  
    std::map<std::string, int> segMap;
    std::map<std::string, int> chaMap;
    std::map<std::string, int> recMap;
    int minRechitChamber;
    int minRechitSegment;
};

#endif
