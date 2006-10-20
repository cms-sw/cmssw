#include "EventFilter/L1GlobalTriggerRawToDigi/interface/L1GTDigiToRaw.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTExtendedCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

L1GTDigiToRaw::L1GTDigiToRaw(const edm::ParameterSet& ps) {
  produces<FEDRawDataCollection>();
}


L1GTDigiToRaw::~L1GTDigiToRaw() {}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
L1GTDigiToRaw::produce(edm::Event& e, const edm::EventSetup& es) {
  edm::Handle<L1MuGMTReadoutCollection> gmtrc_handle;
  e.getByType(gmtrc_handle);
  L1MuGMTReadoutCollection const* gmtrc = gmtrc_handle.product();

  auto_ptr<FEDRawDataCollection> gtrd(new FEDRawDataCollection);
  pack(gmtrc,gtrd);
  e.put(gtrd);
}


void L1GTDigiToRaw::pack(L1MuGMTReadoutCollection const* digis, auto_ptr<FEDRawDataCollection>& rawCol){

  FEDRawData& raw = rawCol->FEDData(FEDNumbering::getTriggerGTPFEDIds().first);
  raw.resize(1480);
  unsigned char* p = raw.data();
  // to do: set eventNr, bxNr., version in the generated header
  FEDHeader::set(p, 1, 0, 0, FEDNumbering::getTriggerGTPFEDIds().first, 0, false);
  p+=8;
  unsigned gtfesize = 0;
  gtfesize = packGTFE(p);
  p+=gtfesize;

  unsigned gmtsize = 0;
  for(int ibx=-1; ibx<=1; ibx++) {
    L1MuGMTReadoutRecord const& gmtrr = digis->getRecord(ibx);
    gmtsize = packGMT(gmtrr,p);
    p+=gmtsize;
  }

  // to do: add crc, evt_stat...
  FEDTrailer::set(p, (8+gtfesize+3*gmtsize+8)/8, 0, 0, 0, false);

}

unsigned L1GTDigiToRaw::packGTFE(unsigned char* chp) {
  const unsigned SIZE = 16;
  unsigned* p = (unsigned*) chp;
  // setup version
  *p++ = 0;
  // bxnr(0), length, boardID(?)
  *p++ = (SIZE<<16);
  // total triggenr(0)
  *p++ = 0;
  // active boards (GMT)
  *p++ = (8<<16);

  return SIZE;
}

unsigned L1GTDigiToRaw::packGMT(L1MuGMTReadoutRecord const& gmtrr, unsigned char* chp) {
  const unsigned SIZE=128;
  memset(chp,0,SIZE);
  unsigned* p = (unsigned*) chp;
  // event number + bcerr
  *p++ = (gmtrr.getEvNr()&0xffffff) | ((gmtrr.getBCERR()&0xff)<<24);
  // bx number, bx in event, length(?), board-id(?)
  *p++ = (gmtrr.getBxNr()&0xfff) | ((gmtrr.getBxInEvent()&0xf)<<12);

  vector<L1MuRegionalCand> vrc;
  vector<L1MuRegionalCand>::const_iterator irc;
  unsigned* pp = p;

  vrc = gmtrr.getDTBXCands();
  pp = p;
  for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
    *pp++ = (*irc).getDataWord();
  }
  p+=4;

  vrc = gmtrr.getBrlRPCCands();
  pp = p;
  for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
    *pp++ = (*irc).getDataWord();
  }
  p+=4;

  vrc = gmtrr.getCSCCands();
  pp = p;
  for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
    *pp++ = (*irc).getDataWord();
  }
  p+=4;

  vrc = gmtrr.getFwdRPCCands();
  pp = p;
  for(irc=vrc.begin(); irc!=vrc.end(); irc++) {
    *pp++ = (*irc).getDataWord();
  }
  p+=4;

  vector<L1MuGMTExtendedCand> vgc;
  vector<L1MuGMTExtendedCand>::const_iterator igc;

  vgc = gmtrr.getGMTBrlCands();
  pp = p;
  for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
    *pp++ = (*igc).getDataWord();
  }
  p+=4;

  vgc = gmtrr.getGMTFwdCands();
  pp = p;
  for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
    *pp++ = (*igc).getDataWord();
  }
  p+=4;

  vgc = gmtrr.getGMTCands();
  pp = p;
  for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
    *pp++ = (*igc).getDataWord();
  }
  p+=4;

  unsigned char* chpp;

  vgc = gmtrr.getGMTBrlCands();
  chpp = (unsigned char*) p;
  for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
    *chpp++ = (*igc).rank();
  }
  p++;
  
  vgc = gmtrr.getGMTFwdCands();
  chpp = (unsigned char*) p;
  for(igc=vgc.begin(); igc!=vgc.end(); igc++) {
    *chpp++ = (*igc).rank();
  }
  p++;

  return SIZE;
}

