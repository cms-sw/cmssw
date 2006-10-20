#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GlobalTriggerRawToDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

L1GlobalTriggerRawToDigi::L1GlobalTriggerRawToDigi(const edm::ParameterSet& ps) {
  //  produces<L1GlobalTriggerReadoutRecord>();
  produces<L1MuGMTReadoutCollection>();
}


L1GlobalTriggerRawToDigi::~L1GlobalTriggerRawToDigi() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1GlobalTriggerRawToDigi::produce(edm::Event& e, const edm::EventSetup& es) {
  edm::Handle<FEDRawDataCollection> fed_handle;
  e.getByType(fed_handle);
  const FEDRawData& raw = (fed_handle.product())->FEDData(FEDNumbering::getTriggerGTPFEDIds().first);

  // get pointer to the GMT block (tmp solution)
  const unsigned char* p = raw.data();
  p+=8; // skip header
  p+=16; // skip GTFE block (later need to determine from here the active boards)

  auto_ptr<L1MuGMTReadoutCollection> gmtrc(new L1MuGMTReadoutCollection);
  unpackGMT(p,gmtrc);
  e.put(gmtrc);
}



void L1GlobalTriggerRawToDigi::unpackGMT(const unsigned char* chp, auto_ptr<L1MuGMTReadoutCollection>& gmtrc) {

  const unsigned* p = (const unsigned*) chp;

  // for the moment assume 3 bx's
  for(int ib=-1; ib<=1; ib++) {
    
    L1MuGMTReadoutRecord gmtrr(ib);

    gmtrr.setEvNr((*p)&0xffffff);
    gmtrr.setBCERR(((*p)>>24)&0xff);
    p++;

    gmtrr.setBxNr((*p)&0xfff);
    gmtrr.setBxInEvent(((*p)>>12)&0xf);
    // to do: check here the block length and the board id
    p++;

    for(int im=0; im<16; im++) {
      gmtrr.setInputCand(im,*p++);
    }

    unsigned char* prank = (unsigned char*) (p+12);

    for(int im=0; im<4; im++) {
      gmtrr.setGMTBrlCand(im, *p++, *prank++);
    }

    for(int im=0; im<4; im++) {
      gmtrr.setGMTFwdCand(im, *p++, *prank++);
    }

    for(int im=0; im<4; im++) {
      gmtrr.setGMTCand(im, *p++);
    }

    // skip the two sort rank words
    p+=2;

    gmtrc->addRecord(gmtrr);
    
  }
}

