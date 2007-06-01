/** \class SiPixelRawDumper_H
 *  Plug-in module that dump raw data file 
 *  for pixel subdetector
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/SiPixelRawToDigi/interface/PixelDataFormatter.h"


class SiPixelRawDumper : public edm::EDAnalyzer {
public:

  /// ctor
  explicit SiPixelRawDumper( const edm::ParameterSet& cfg) : theConfig(cfg) {} 

  /// dtor
  virtual ~SiPixelRawDumper() {}

  virtual void beginJob( const edm::EventSetup& ) {}

  /// dummy end of job 
  virtual void endJob() {}

  /// get data, convert to digis attach againe to Event
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

private:
  edm::ParameterSet theConfig;
};

void SiPixelRawDumper::analyze(const  edm::Event& ev, const edm::EventSetup& es) 
{
  edm::Handle<FEDRawDataCollection> buffers;
  static std::string label = theConfig.getUntrackedParameter<std::string>("InputLabel","source");
  static std::string instance = theConfig.getUntrackedParameter<std::string>("InputInstance","");
  ev.getByLabel( label, instance, buffers);

  FEDNumbering fednum;
  std::pair<int,int> fedIds = fednum.getSiPixelFEDIds();

  PixelDataFormatter formatter(0);

  for (int fedId = fedIds.first; fedId <= fedIds.second; fedId++) {
    LogDebug("SiPixelRawDumper")<< " GET DATA FOR FED: " <<  fedId ;
    PixelDataFormatter::Digis digis;

    //get event data for this fed
    const FEDRawData& fedRawData = buffers->FEDData( fedId );

    //convert data to digi
    formatter.interpretRawData( fedId, fedRawData, digis);
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPixelRawDumper);

