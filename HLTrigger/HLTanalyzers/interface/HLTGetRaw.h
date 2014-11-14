#ifndef HLTGetRaw_h
#define HLTGetRaw_h

/** \class HLTGetRaw
 *
 *  
 *  This class is an EDAnalyzer implementing a "get data into RAM"
 *  functionality for RAW, to simulate online FF running/timimg.
 *
 *
 *  \author various
 *
 */

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

//
// class declaration
//

class HLTGetRaw : public edm::global::EDAnalyzer<> {

 public:
  explicit HLTGetRaw(const edm::ParameterSet&);
  ~HLTGetRaw();
  virtual void analyze(edm::StreamID, edm::Event const& , edm::EventSetup const&) const override final;
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

 private:
  edm::InputTag                          rawDataCollection_;
  edm::EDGetTokenT<FEDRawDataCollection> rawDataToken_;
};

#endif //HLTGetRaw_h
