#ifndef HLTGetRaw_h
#define HLTGetRaw_h

/** \class HLTGetRaw
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality for RAW, to simulate online FF running/timimg.
 *
 *  $Date: 2007/04/20 06:58:26 $
 *  $Revision: 1.1 $
 *
 *  \author various
 *
 */

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// class declaration
//

class HLTGetRaw : public edm::EDAnalyzer {

 public:
  explicit HLTGetRaw(const edm::ParameterSet&);
  ~HLTGetRaw();
  void analyze(const edm::Event&, const edm::EventSetup&);
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
 private:
  edm::InputTag RawDataCollection_;
  edm::EDGetTokenT<FEDRawDataCollection> RawDataToken_;
};

#endif //HLTGetRaw_h
