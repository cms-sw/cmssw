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

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "EventFilter/CastorRawToDigi/interface/CastorUnpacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorCtdcUnpacker.h"
#include "EventFilter/CastorRawToDigi/interface/CastorDataFrameFilter.h"

class CastorRawToDigi : public edm::EDProducer
{
public:
  explicit CastorRawToDigi(const edm::ParameterSet& ps);
  virtual ~CastorRawToDigi();
  virtual void produce(edm::Event& e, const edm::EventSetup& c) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;

private:
  edm::InputTag dataTag_;
  CastorUnpacker unpacker_;
  CastorCtdcUnpacker ctdcunpacker_;
  CastorDataFrameFilter filter_;
  std::vector<int> fedUnpackList_;
  int firstFED_;
  // bool unpackCalib_;
  bool complainEmptyData_;
  bool usingctdc_;
  bool unpackTTP_;
  bool silent_;
  bool usenominalOrbitMessageTime_;
  int expectedOrbitMessageTime_;

};

#endif
