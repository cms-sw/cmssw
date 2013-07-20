//-------------------------------------------------
//
//   \class L1MuGMTHWFileReader
//
//   Description: Puts the GMT input information from 
//                a GMT ascii HW testfile into the Event
//
//
//   $Date: 2012/11/19 21:03:41 $
//   $Revision: 1.5 $
//
//   Author :
//   Tobias Noebauer                 HEPHY Vienna
//   Ivan Mikulec                    HEPHY Vienna
//
//--------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "L1Trigger/GlobalMuonTrigger/src/L1MuGMTHWFileReader.h"

//---------------
// C++ Headers --
//---------------
#include <stdexcept>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//----------------
// Constructors --
//----------------
L1MuGMTHWFileReader::L1MuGMTHWFileReader(edm::ParameterSet const& ps,
                                         edm::InputSourceDescription const& desc) :
                                         ProducerSourceFromFiles(ps, desc, true) {

  produces<std::vector<L1MuRegionalCand> >("DT");
  produces<std::vector<L1MuRegionalCand> >("CSC");
  produces<std::vector<L1MuRegionalCand> >("RPCb");
  produces<std::vector<L1MuRegionalCand> >("RPCf");

  produces<L1CaloRegionCollection>();

  if(!fileNames().size()) {
    throw std::runtime_error("L1MuGMTHWFileReader: no input file");
  }
  edm::LogInfo("GMT_HWFileReader_info") << "opening file " << fileNames()[0];
  m_in.open((fileNames()[0].substr(fileNames()[0].find(":")+1)).c_str());
  if(!m_in) {
    throw std::runtime_error("L1MuGMTHWFileReader: file " + fileNames()[0]
			+ " could not be openned");
  }

}

//--------------
// Destructor --
//--------------
L1MuGMTHWFileReader::~L1MuGMTHWFileReader() {
  m_in.close();
}

//--------------
// Operations --
//--------------
bool L1MuGMTHWFileReader::setRunAndEventInfo(edm::EventID& id, edm::TimeValue_t& time) {
  readNextEvent();
  if(!m_evt.getRunNumber() && !m_evt.getEventNumber()) return false;
  id = edm::EventID(m_evt.getRunNumber(), id.luminosityBlock(), m_evt.getEventNumber());

  edm::LogInfo("GMT_HWFileReader_info") << "run: " << m_evt.getRunNumber() << 
          "   evt: " << m_evt.getEventNumber();
  return true;
}

void L1MuGMTHWFileReader::produce(edm::Event& e) {
  L1MuRegionalCand empty_mu;


  std::auto_ptr<std::vector<L1MuRegionalCand> > DTCands(new std::vector<L1MuRegionalCand>);
  for (unsigned i = 0; i < 4; i++) {
    const L1MuRegionalCand *mu = m_evt.getInputMuon("IND", i);
    if (!mu) mu = &empty_mu;
    DTCands->push_back(*mu);
  }
  e.put(DTCands,"DT");

  std::auto_ptr<std::vector<L1MuRegionalCand> > CSCCands(new std::vector<L1MuRegionalCand>);
  for (unsigned i = 0; i < 4; i++) {
    const L1MuRegionalCand *mu = m_evt.getInputMuon("INC", i);
    if (!mu) mu = &empty_mu;
    CSCCands->push_back(*mu);
  }
  e.put(CSCCands,"CSC");

  std::auto_ptr<std::vector<L1MuRegionalCand> > RPCbCands(new std::vector<L1MuRegionalCand>);
  for (unsigned i = 0; i < 4; i++) {
    const L1MuRegionalCand *mu = m_evt.getInputMuon("INB", i);
    if (!mu) mu = &empty_mu;
    RPCbCands->push_back(*mu);
  }
  e.put(RPCbCands,"RPCb");

  std::auto_ptr<std::vector<L1MuRegionalCand> > RPCfCands(new std::vector<L1MuRegionalCand>);
  for (unsigned i = 0; i < 4; i++) {
    const L1MuRegionalCand *mu = m_evt.getInputMuon("INF", i);
    if (!mu) mu = &empty_mu;
    RPCfCands->push_back(*mu);
  }
  e.put(RPCfCands,"RPCf");

  std::auto_ptr<L1CaloRegionCollection> rctRegions (new L1CaloRegionCollection);
  for(int ieta = 4; ieta < 18; ieta++) {
    for(int iphi = 0; iphi < 18; iphi++) {
      rctRegions->push_back(L1CaloRegion(0,false,true,m_evt.getMipBit(ieta-4,iphi),m_evt.getIsoBit(ieta-4,iphi),ieta,iphi));
    }
  }

  e.put(rctRegions);
}

void L1MuGMTHWFileReader::readNextEvent() {
  m_evt.reset();

  std::string line_id;
  do {
    int bx = 0;

    m_in >> line_id;
    if (line_id == "--") continue;

    if (line_id == "RUN") {
      unsigned long runnr;
      m_in >> runnr;
      m_evt.setRunNumber(runnr);
    }

    if (line_id == "EVT") {
      unsigned long evtnr;
      m_in >> evtnr;
      m_evt.setEventNumber(evtnr);
    }


    if (line_id == "DT" || line_id == "CSC" || line_id == "BRPC" || line_id == "FRPC")
    {

      // decode input muon

      unsigned inpmu = 0;
      unsigned val;
      m_in >> val; inpmu |= (val & 0x01) << 24; // valid charge
      m_in >> val; inpmu |= (val & 0x01) << 23; // charge
      m_in >> val; inpmu |= (val & 0x01) << 22; // halo / fine
      m_in >> val; inpmu |= (val & 0x3f) << 16; // eta
      m_in >> val; inpmu |= (val & 0x07) << 13; // quality
      m_in >> val; inpmu |= (val & 0x1f) <<  8; // pt
      m_in >> val; inpmu |= (val & 0xff)      ; // phi

      std::string chipid("IN");
      chipid += line_id[0];

      int type=0;
      if (line_id == "DT") type = 0;
      if (line_id == "CSC") type = 2;
      if (line_id == "BRPC") type = 1;
      if (line_id == "FRPC") type = 3;


      L1MuRegionalCand cand(inpmu);
      cand.setType(type);
      cand.setBx(bx);
      m_evt.addInputMuon(chipid, cand);
    }

    if (line_id == "MIP") {
      int nPairs;
      m_in >> nPairs;
      for (int i=0; i<nPairs; i++) {
        unsigned eta;
        unsigned phi;
        m_in >> eta;
        m_in >> phi;
        if (phi >= 9) phi-=9;
        else phi+=9;
        m_evt.setMipBit(eta, phi, true);
      }
    }

    if (line_id == "NQ") {
      int nPairs;
      m_in >> nPairs;
      for (int i=0; i<nPairs; i++) {
        unsigned eta;
        unsigned phi;
        m_in >> eta;
        m_in >> phi;
        if (phi >= 9) phi-=9;
        else phi+=9;
        m_evt.setIsoBit(eta, phi, false);
      }
    }

    //read the rest of the line
    const int sz=4000; char buf[sz];
    m_in.getline(buf, sz);

  } while (line_id != "NQ" && !m_in.eof());

}
