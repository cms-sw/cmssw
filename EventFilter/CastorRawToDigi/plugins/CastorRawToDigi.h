#ifndef CastorRawToDigi_h
#define CastorRawToDigi_h

/** \class CastorRawToDigi
 *
 * CastorRawToDigi is the EDProducer subclass which runs 
 * the Hcal Unpack algorithm.
 *
 * \author Alan Campbell
      
 *
 * \version   1st Version April 18, 2008  

 *
 ************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/CastorRawToDigi/interface/CastorUnpacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCtdcUnpacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorDataFrameFilter.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "EventFilter/CastorRawToDigi/interface/ZdcUnpacker.h"
#include "CondFormats/DataRecord/interface/HcalAllRcds.h"
#include <map>
//#include "Geometry/Records/interface/HcalRecNumberingRecord.h"


class CastorRawToDigi : public edm::stream::EDProducer<>
{
public:
  explicit CastorRawToDigi(const edm::ParameterSet& ps);
  ~CastorRawToDigi() override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;

private:
  edm::InputTag dataTag_;
  CastorUnpacker unpacker_;
  ZdcUnpacker zdcunpacker_;
  CastorCtdcUnpacker ctdcunpacker_;
  CastorDataFrameFilter filter_;
  std::vector<int> fedUnpackList_;
  int firstFED_;
  bool complainEmptyData_;
  bool usingctdc_;
  bool unpackTTP_;
  bool unpackZDC_;
  bool silent_;
  bool usenominalOrbitMessageTime_;
  int expectedOrbitMessageTime_;
  std::unique_ptr<HcalElectronicsMap> myEMap;
  edm::EDGetTokenT<FEDRawDataCollection> tok_input_;
  edm::ParameterSet zdcemap;
};

#endif
