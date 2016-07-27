/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*   Jan Kašpar (jan.kaspar@gmail.com)
*
****************************************************************************/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/TotemDigi/interface/TotemTriggerCounters.h"

#include <string>
#include <iostream>

//----------------------------------------------------------------------------------------------------

class TotemTriggerRawToDigi : public edm::stream::EDProducer<>
{
  public:
    explicit TotemTriggerRawToDigi(const edm::ParameterSet&);
    ~TotemTriggerRawToDigi();

    virtual void produce(edm::Event&, const edm::EventSetup&) override;

  private:
    unsigned int fedId;

    edm::EDGetTokenT<FEDRawDataCollection> fedDataToken;

    /// Process one LoneG frame.
    int ProcessLoneGFrame(uint64_t *oBuf, unsigned long size, TotemTriggerCounters &data);
};

//----------------------------------------------------------------------------------------------------

using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------

TotemTriggerRawToDigi::TotemTriggerRawToDigi(const edm::ParameterSet &conf):
  fedId(conf.getParameter<unsigned int>("fedId"))
{
  fedDataToken = consumes<FEDRawDataCollection>(conf.getParameter<edm::InputTag>("rawDataTag"));

  if (fedId == 0)
    fedId = FEDNumbering::MINTotemTriggerFEDID;

  produces<TotemTriggerCounters>();
}

//----------------------------------------------------------------------------------------------------

TotemTriggerRawToDigi::~TotemTriggerRawToDigi()
{
}

//----------------------------------------------------------------------------------------------------

void TotemTriggerRawToDigi::produce(edm::Event& event, const edm::EventSetup &es)
{
  // raw data handle
  edm::Handle<FEDRawDataCollection> rawData;
  event.getByToken(fedDataToken, rawData);

  // book output products
  TotemTriggerCounters totemTriggerCounters;

  // unpack trigger data
  const FEDRawData &data = rawData->FEDData(fedId);
  uint64_t *buf = (uint64_t *) data.data();
  unsigned int sizeInWords = data.size() / 8; // bytes -> words
  if (data.size() > 0)
    ProcessLoneGFrame(buf + 2, sizeInWords - 4, totemTriggerCounters);

  // commit products to event
  event.put(make_unique<TotemTriggerCounters>(totemTriggerCounters));
}

//----------------------------------------------------------------------------------------------------

int TotemTriggerRawToDigi::ProcessLoneGFrame(uint64_t *oBuf, unsigned long size, TotemTriggerCounters &td)
{
  if (size != 20)
  {
    LogProblem("Totem") << "Error in TotemTriggerRawToDigi::ProcessLoneGFrame > " <<
      "Wrong LoneG frame size: " << size << " (shall be 20)." << endl;
    return 1;
  }

  // buffer mapping: OptoRx buffer --> LoneG buffer
  uint64_t buf[5];
  for (unsigned int i = 0; i < 5; i++)
    buf[i] = 0;

  for (unsigned int i = 0; i < 20; i++)
  {
      int row = i / 4;
      int col = i % 4;
      buf[row] |= (oBuf[i] & 0xFFFF) << (col * 16);
  }

  td.type = (buf[0] >> 56) & 0xF;
  td.event_num = (buf[0] >> 32) & 0xFFFFFF;
  td.bunch_num = (buf[0] >> 20) & 0xFFF;
  td.src_id = (buf[0] >> 8) & 0xFFF;

  td.orbit_num = (buf[1] >> 32) & 0xFFFFFFFF;
  td.revision_num = (buf[1] >> 24) & 0xFF;

  td.run_num = (buf[2] >> 32) & 0xFFFFFFFF;
  td.trigger_num = (buf[2] >> 0) & 0xFFFFFFFF;

  td.inhibited_triggers_num = (buf[3] >> 32) & 0xFFFFFFFF;
  td.input_status_bits = (buf[3] >> 0) & 0xFFFFFFFF;

#ifdef DEBUG
  printf(">> RawDataUnpacker::ProcessLoneGFrame > size = %li\n", size);
  printf("\ttype = %x, event number = %x, bunch number = %x, id = %x\n",
    td.type, td.event_num, td.bunch_num, td.src_id);
  printf("\torbit number = %x, revision = %x\n",
    td.orbit_num, td.revision_num);
  printf("\trun number = %x, trigger number = %x\n",
    td.run_num, td.trigger_num);
  printf("\tinhibited triggers = %x, input status bits = %x\n",
    td.inhibited_triggers_num, td.input_status_bits);
#endif

  return 0;
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(TotemTriggerRawToDigi);
