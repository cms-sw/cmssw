// system include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <bitset>
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"



#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"

#include "EventFilter/L1TRawToDigi/interface/AMC13Spec.h"
#include "EventFilter/L1TRawToDigi/interface/AMCSpec.h"

#include "CondFormats/RPCObjects/interface/RPCReadOutMapping.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "CondFormats/RPCObjects/interface/RPCEMap.h"
#include "CondFormats/DataRecord/interface/RPCEMapRcd.h"
#include "CondFormats/RPCObjects/interface/DccSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardElectronicIndex.h"
#include "CondFormats/RPCObjects/interface/LinkBoardPackedStrip.h"
#include "CondFormats/RPCObjects/interface/TriggerBoardSpec.h"
#include "CondFormats/RPCObjects/interface/LinkBoardSpec.h"

#include "EventFilter/RPCRawToDigi/interface/RPCRecordFormatter.h"
#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include "DataFormats/RPCDigi/interface/ReadoutError.h"
#include "DataFormats/RPCDigi/interface/RPCRawSynchro.h"
#include "EventFilter/RPCRawToDigi/interface/EventRecords.h"
#include "EventFilter/RPCRawToDigi/interface/DebugDigisPrintout.h"

#include "CondFormats/DataRecord/interface/RPCOMTFLinkMapRcd.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLinkMap.h"
#include "CondFormats/RPCObjects/interface/RPCAMCLink.h"
#include "CondFormats/RPCObjects/interface/RPCLBLink.h"

#include "DataFormats/RPCDigi/interface/DataRecord.h"
#include "DataFormats/RPCDigi/interface/RecordBX.h"
#include "DataFormats/RPCDigi/interface/RecordSLD.h"
#include "DataFormats/RPCDigi/interface/RecordCD.h"

#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"


typedef uint64_t Word64;

namespace DataWord64 {
  enum Type { csc=0xC, dt= 0xD, rpc=0xE, omtf=0xF };
  template <typename T> Type type(const T&);
  template <> Type type<Word64>(const Word64 & data) { return static_cast<Type> (data>>60); }
  template <> Type type<unsigned int>(const unsigned int & intType) { return static_cast<Type> (intType); }
  std::ostream & operator<< (std::ostream &out, const DataWord64::Type &o) {
    switch(o) {
      case(csc)  : out <<"csc "; break;
      case(rpc)  : out <<"rpc "; break;
      case(dt)   : out <<"dt  "; break;
      case(omtf) : out <<"omtf"; break;
      default    : out <<"unkn"; break; 
    }
    out<<"(0x"<<std::hex<<static_cast<int>(o)<<std::dec<<")";
    return out;
  }
};

class RpcDataWord64 {
public: 
  RpcDataWord64(Word64 data=0) : rawData(data) {}
  unsigned int frame1() const { return frame1_;}
  unsigned int frame2() const { return frame2_;}
  unsigned int frame3() const { return frame3_;}
  unsigned int empty() const { return empty_;}
  unsigned int linkNum() const { return linkNum_;}
  unsigned int bxNum() const { return bxNum_; }
  unsigned int type() const { return type_;}
  friend std::ostream & operator<< (std::ostream &out, const RpcDataWord64 &o) {
    out << "RpcDataWord64: "
        <<" type: "<< std::bitset<4>(o.type_) 
        << " bx: "<<o.bxNum_
        << " lnk: "<< o.linkNum_;
    out << std::hex;
    out << " frame1: 0x"<< o.frame1_; if (o.frame1_ != 0) out <<" ("<< std::bitset<16>(o.frame1_)<<")";
    out << " frame2: 0x"<< o.frame2_; if (o.frame2_ != 0) out <<" ("<< std::bitset<16>(o.frame2_)<<")";
    out << " frame3: 0x"<< o.frame3_; if (o.frame3_ != 0) out <<" ("<< std::bitset<16>(o.frame3_)<<")";
    out << std::dec;
    return out;
  }
private:
  union {
    uint64_t rawData;
    struct {
      uint64_t frame3_  : 16;
      uint64_t frame2_  : 16;
      uint64_t frame1_  : 16;
      uint64_t empty_   : 4;
      uint64_t linkNum_ : 5;
      uint64_t bxNum_   : 3;
      uint64_t type_    : 4;
    };
  };
};

class OmtfDataWord64 {
public:
  OmtfDataWord64(Word64 data=0) : rawData(data) {}
  unsigned int weight_lowBits() const { return weight_; } 
  unsigned int layers() const { return layers_; }
  unsigned int ch() const { return ch_; } 
  unsigned int vch() const { return vch_; } 
           int phi() const { return phi_; }
           int eta() const { return eta_; }
  unsigned int pT() const { return pT_; }
  unsigned int quality() const { return quality_; }
  unsigned int bxNum() const { return bxNum_; }
  unsigned int type() const { return type_;}
  friend std::ostream & operator<< (std::ostream &out, const OmtfDataWord64 &o) {
    out << "OmtfDataWord64: "
        <<" type: "<< DataWord64::type(o.type())
        << " bx: "<<o.bxNum()
        << " pT: "<<o.pT()
        << " eta: "<<o.eta()
        << " phi: "<<o.phi()
        << " quality: "<<o.quality()
        << " layers: "<< std::bitset<18>(o.layers())
        << "";
    return out;
  }
private:
  union {
    uint64_t rawData;
    struct {
      uint64_t pT_ : 9;
      uint64_t quality_ : 4;
       int64_t eta_ : 9;
      uint64_t empty_   : 1; //not used, orig h/f
       int64_t phi_ : 8;
      uint64_t bc0_  : 1;
      uint64_t ch_ : 1;
      uint64_t vch_ : 1;
      uint64_t layers_ : 18;
      uint64_t weight_ : 5;
      uint64_t bxNum_   : 3;
      uint64_t type_    : 4;
    };
  };
};

class DtDataWord64 {
public:
  DtDataWord64(Word64 data=0) : rawData(data) {}
  int phi() const { return st_phi_; }
  int phiB() const { return st_phib_; }
  unsigned int quality() const { return st_q_; }
  unsigned int eta() const { return eta_hit_; }
  unsigned int etaQuality() const { return eta_qbit_; }
  unsigned int bcnt_st() const { return bcnt_st_; }
  unsigned int bcnt_e0() const { return bcnt_e0_; }
  unsigned int bcnt_e1() const { return bcnt_e1_; }
  unsigned int valid() const { return valid_; }
  unsigned int station() const { return st_; }
  unsigned int fiber() const { return fiber_; }
  unsigned int sector() const { return sector_; }
  unsigned int bxNum() const { return bxNum_; }
  unsigned int type() const { return type_;}
  friend std::ostream & operator<< (std::ostream &out, const DtDataWord64 &o) {
    out << "DtDataWord64: "
        <<" type: "<< DataWord64::type(o.type())
        << " bx: "<<o.bxNum()
        << " station: " << o.station()
        << " sector: " << o.sector()
        << " fiber: " << o.fiber()
        << " phi: "<<o.phi()
        << " quality: "<<o.quality()
        << " eta: "<<o.eta()
        << " etaQ: "<<o.etaQuality()
        << " bcnt: "<<o.bcnt_st()<<"_"<<o.bcnt_e0()<<"_"<<o.bcnt_e1()
        << "";
    return out;
  }
private:
  union {
    uint64_t rawData;
    struct {
      int64_t st_phi_ : 12;
      int64_t st_phib_ : 10;
      uint64_t st_q_     : 5;
      uint64_t st_cal_   : 1;
      uint64_t eta_qbit_ : 7;
      uint64_t eta_hit_ : 7;
      uint64_t dummy1_  : 1;
      uint64_t bcnt_st_ : 2;
      uint64_t bcnt_e0_ : 2;
      uint64_t bcnt_e1_ : 2;
      uint64_t valid_   : 3;
      uint64_t st_      : 2;
      uint64_t fiber_   : 1;
      uint64_t sector_  : 2;
      uint64_t bxNum_   : 3;
      uint64_t type_    : 4;
    };
  };
};

class CscDataWord64 {
public:
  CscDataWord64(const Word64 & data=0) : rawData(data) {}  

  unsigned int type() const { return type_;}
  unsigned int bxNum() const { return bxNum_; }
  unsigned int hitNum() const { return hitNum_; }
  unsigned int wireGroup() const { return keyWG_; }
  unsigned int quality() const { return quality_; }
  unsigned int clctPattern() const { return clctPattern_; }
  unsigned int cscID() const { return cscID_; }
  unsigned int halfStrip() const { return halfStrip_; }
  unsigned int linkNum() const { return linkNum_;}
  unsigned int station() const { return station_; }
  unsigned int bend() const { return lr_; }
  unsigned int valid() const { return vp_; }

  friend std::ostream & operator<< (std::ostream &out, const CscDataWord64 &o) {
    out << "CscDataWord64: "
        <<" type: "<< DataWord64::type(o.type())
        << " val: "<< o.valid()
        << " bx: "<<o.bxNum()
        << " lnk: "<< o.linkNum() 
        << " stat: "<<o.station()
        << " cscId: " << o.cscID() 
        << " hit: "<< o.hitNum()
        << " qual: "<< o.quality()
        << " patt: " << o.clctPattern()
        << " bending: " << o.bend()
        << " hs: "<<o.halfStrip()
        << " wg: "<< o.wireGroup();
    return out;
  }
private:
  union {
    uint64_t rawData;
    struct {               // watch out - bit fields are  is implementation-defined
      uint64_t dummy7_      : 3; //2:0   unused, oryginalnie TBIN Num  
      uint64_t vp_          : 1; //3:3   VP      
      uint64_t station_     : 3; //6:4   Station                      
      uint64_t af_          : 1; //7:7   AF       
      uint64_t dummy6_      : 4; //11:8  unused, oryginalnie EPC     
      uint64_t sm_          : 1; //12:12 unused, oryginalnie SM     
      uint64_t se_          : 1; //13:13 SE     
      uint64_t dummy5_      : 1; //14:14 unused, oryginalnie AFEF   
      uint64_t dummy4_      : 2; //16:15 unused, oryginalnie ME BXN [11:0] 
      uint64_t nit_         : 1; //17:17 NIT   
      uint64_t cik_         : 1; //18:18 CIK    
      uint64_t dummy3_      : 1; //19:19 unused, oryginalnie AFFF  
      uint64_t linkNum_     : 6; //25:20 numer linku CSC               
      uint64_t halfStrip_   : 8; //33:26 CLCT key half-strip [7:0]      
      uint64_t cscID_       : 4; //37:34 CSC ID [3:0]                    
      uint64_t lr_          : 1; //38:38 L/R
      uint64_t dummy2_      : 1; //39:39 unused, oryginalnie BXE   
      uint64_t dummy1_      : 1; //40:40 unused, oryginalnie BC0    
      uint64_t clctPattern_ : 4; //44:41 CLCT pattern [3:0]              4b
      uint64_t quality_     : 4; //48:45 Quality [3:0]                   
      uint64_t keyWG_       : 7; //55:49 Key wire group [6:0]            
      uint64_t hitNum_      : 1; //56:56 int in chamber 0 or 1
      uint64_t bxNum_       : 3; //59:57 SBXN
      uint64_t type_        : 4; //63:60 CSC identifier 0xC        
    };
  };
};



class OmtfEleIndex {
public:
  OmtfEleIndex() : packed_(0) {}
  OmtfEleIndex(const std::string & board, unsigned int link) {
    unsigned int fed = 0;
    if (board.substr(4,1)=="n") fed = 1380; else if (board.substr(4,1)=="p") fed = 1381;
    unsigned int amc = std::stoi( board.substr(5,1) );
    packed_ = fed*1000+amc*100+link; 
  }
  OmtfEleIndex(unsigned int fed, unsigned int amc, unsigned int link) { packed_ = fed*1000+amc*100+link; }
  unsigned int fed() const  { return packed_/1000; }
  unsigned int amc() const  { return ( (packed_ /100) %10); }
  unsigned int link() const { return packed_ % 100; } 
  friend std::ostream & operator<< (std::ostream &out, const OmtfEleIndex&o) {
    out << "OMTF";
    if (o.fed()==1380) out <<"n";
    if (o.fed()==1381) out <<"p";
    out << o.amc();
    out <<" (fed: "<<o.fed()<<"), ln: " << o.link();
    return out;
  }
  inline bool operator< (const OmtfEleIndex& o) const { return this->packed_ < o.packed_; }

private:
  uint32_t packed_;

};


class RpcLinkMap {
public:
  RpcLinkMap() { }

  void init(const RPCAMCLinkMap::map_type & amcMap) {

    for (const auto & item : amcMap ) {
      unsigned int fedId = item.first.getFED();
      unsigned int amcSlot = item.first.getAMCNumber();
      unsigned int link = item.first.getAMCInput();
      std::string lbName =  item.second.getName();

      std::ostringstream processorNameStr; processorNameStr<<"OMTF";;
      if (fedId==1380) processorNameStr<< "n"; else processorNameStr<< "p";
      processorNameStr<< amcSlot/2+1;
      std::string processorName(processorNameStr.str());

      std::map< unsigned int, std::string > & li2lb = link2lbName[processorName];
      std::map< std::string, unsigned int > & lb2li = lbName2link[processorName];
      li2lb[link] = lbName;
      lb2li[lbName] = link;
      OmtfEleIndex ele(processorName, link);
      lbName2OmtfIndex[lbName].push_back(ele);
    } 
  }

  void init( const std::string& fName) {
    std::ifstream inFile;
    inFile.open(fName);
    if (inFile) {
      LogTrace("")<<" reading OmtfRpcLinksMap from: "<<fName;
    } else {
      LogTrace("")<<" Unable to open file "<<fName;

      throw std::runtime_error("Unable to open OmtfRpcLinksMap file " + fName);
    }

    std::string line;
    while (std::getline(inFile, line)) {
      line.erase(0, line.find_first_not_of(" \t\r\n"));      //cut first character
      if (line.empty() || !line.compare(0,2,"--")) continue; // empty or comment line
      std::stringstream ss(line);
      std::string processorName, lbName;
      unsigned int link, dbId;
      if (ss >> processorName >> link >> lbName >> dbId) {
          std::map< unsigned int, std::string > & li2lb = link2lbName[processorName];
          std::map< std::string, unsigned int > & lb2li = lbName2link[processorName];
          li2lb[link] = lbName;
          lb2li[lbName] = link;
          OmtfEleIndex ele(processorName, link);
          lbName2OmtfIndex[lbName].push_back(ele);
      }
    }
    inFile.close();
  }

  const std::string & lbName(const std::string& board, unsigned int link) const {
    return link2lbName.at(board).at(link);
  }

  unsigned int        link(const std::string& board, const std::string& lbName) const {
    return lbName2link.at(board).at(lbName);
  }

  const std::vector<OmtfEleIndex> & omtfEleIndex ( const std::string& lbName) const {
     return lbName2OmtfIndex.at(lbName);
  }

private:
    std::map<std::string, std::map<unsigned int, std::string> > link2lbName; //[processorName][rpcRxNum] - lbName
    std::map<std::string, std::map<std::string, unsigned int> > lbName2link; //[processorName][lbName] - rpcRxNum
    std::map<std::string, std::vector<OmtfEleIndex> > lbName2OmtfIndex; //[lbName] - vector of {board,rpcRxNum}

};


class OmtfUnpacker: public edm::stream::EDProducer<> {
public:

    ///Constructor
    OmtfUnpacker(const edm::ParameterSet& pset);

    ~OmtfUnpacker() {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    void produce(edm::Event & ev, const edm::EventSetup& es) override;

    void beginRun(const edm::Run &run, const edm::EventSetup& es) override;

private:

  edm::InputTag dataLabel_;
  unsigned long eventCounter_;

  edm::EDGetTokenT<FEDRawDataCollection> fedToken_;

  std::map<OmtfEleIndex, LinkBoardElectronicIndex> omtf2rpc_;
  std::map<OmtfEleIndex, CSCDetId> omtf2csc_;
  
  const RPCReadOutMapping* theCabling;

  edm::ParameterSet theConfig;

};

OmtfUnpacker::OmtfUnpacker(const edm::ParameterSet& pset) : theConfig(pset) {
  produces<RPCDigiCollection>("OmtfUnpack");
  produces<CSCCorrelatedLCTDigiCollection>("OmtfUnpack");
  produces<l1t::RegionalMuonCandBxCollection >("OmtfUnpack");
  produces<L1MuDTChambPhContainer>("OmtfUnpack");
  produces<L1MuDTChambThContainer>("OmtfUnpack");

  fedToken_ = consumes<FEDRawDataCollection>(pset.getParameter<edm::InputTag>("InputLabel"));
}

void OmtfUnpacker::beginRun(const edm::Run &run, const edm::EventSetup& es) {
  edm::ESTransientHandle<RPCEMap> readoutMapping;
  es.get<RPCEMapRcd>().get(readoutMapping);
  const RPCReadOutMapping * cabling= readoutMapping->convert();
  theCabling = cabling;
  LogDebug("OmtfUnpacker") <<" Has readout map, VERSION: " << cabling->version() << std::endl;


  RpcLinkMap omtfLink2Ele;

  if (theConfig.getParameter<bool>("useRpcConnectionFile")) {
    edm::FileInPath fip(theConfig.getParameter<string>("rpcConnectionFile"));
    omtfLink2Ele.init(fip.fullPath());
  } else {
    edm::ESHandle<RPCAMCLinkMap> amcMapping;
    es.get<RPCOMTFLinkMapRcd>().get(amcMapping);
    omtfLink2Ele.init(amcMapping->getMap());
  }

  std::vector<const DccSpec*> dccs = cabling->dccList();
  for (std::vector<const DccSpec*>::const_iterator it1= dccs.begin(); it1!= dccs.end(); ++it1) {
    const std::vector<TriggerBoardSpec> & rmbs = (*it1)->triggerBoards();
    for (std::vector<TriggerBoardSpec>::const_iterator it2 = rmbs.begin(); it2 != rmbs.end(); ++it2) {
      const  std::vector<LinkConnSpec> & links = it2->linkConns();
      for (std::vector<LinkConnSpec>::const_iterator it3 = links.begin(); it3 != links.end(); ++it3) {
        const  std::vector<LinkBoardSpec> & lbs = it3->linkBoards();
        for (std::vector<LinkBoardSpec>::const_iterator it4=lbs.begin(); it4 != lbs.end(); ++it4) {
          
          try {
            std::string lbNameCH = it4->linkBoardName();
            std::string lbName = lbNameCH.substr(0,lbNameCH.size()-4);
            const std::vector<OmtfEleIndex> & omtfEles = omtfLink2Ele.omtfEleIndex(lbName); 
//          std::cout <<"  isOK ! " <<  it4->linkBoardName() <<" has: " << omtfEles.size() << " first: "<< omtfEles[0] << std::endl;
            LinkBoardElectronicIndex rpcEle = { (*it1)->id(), it2->dccInputChannelNum(), it3->triggerBoardInputNumber(), it4->linkBoardNumInLink()};
            for ( const auto & omtfEle : omtfEles ) omtf2rpc_[omtfEle]= rpcEle; 
          } 
          catch(...) { ; } // std::cout << "exception! "<<it4->linkBoardName()<< std::endl; }
        }
      }
    }
  }
  LogTrace(" ") << " SIZE OF OMTF to RPC map  is: " << omtf2rpc_.size() << std::endl;


  //
  // init CSC Link map
  //
  omtf2csc_.clear();
  for (unsigned int fed=1380; fed<=1381; fed++) {
    //Endcap label. 1=forward (+Z); 2=backward (-Z)
    unsigned int endcap = (fed==1380) ? 2 : 1;
    for (unsigned int amc=1;    amc<=6; amc++) {
      for (unsigned int link=0; link <=34; link++) {
        unsigned int stat=0;
        unsigned int ring=0;
        unsigned int cham=0;
        switch (link) {
          case ( 0) : { stat=1; ring=2; cham=3; break;} //  (0,  9, 2, 3 ), --channel_0  OV1A_4 chamber_ME1/2/3  layer_9 input 2, 3
          case ( 1) : { stat=1; ring=2; cham=4; break;} //  (1,  9, 4, 5 ), --channel_1  OV1A_5 chamber_ME1/2/4  layer_9 input 4, 5
          case ( 2) : { stat=1; ring=2; cham=5; break;} //  (2,  9, 6, 7 ), --channel_2  OV1A_6 chamber_ME1/2/5  layer_9 input 6, 7
          case ( 3) : { stat=1; ring=3; cham=3; break;} //  (3,  6, 2, 3 ), --channel_3  OV1A_7 chamber_ME1/3/3  layer_6 input 2, 3 
          case ( 4) : { stat=1; ring=3; cham=4; break;} //  (4,  6, 4, 5 ), --channel_4  OV1A_8 chamber_ME1/3/4  layer_6 input 4, 5
          case ( 5) : { stat=1; ring=3; cham=5; break;} //  (5,  6, 6, 7 ), --channel_5  OV1A_9 chamber_ME1/3/5  layer_6 input 6, 7
          case ( 6) : { stat=1; ring=2; cham=6; break;} //  (6,  9, 8, 9 ), --channel_6  OV1B_4 chamber_ME1/2/6  layer_9 input 8, 9
          case ( 7) : { stat=1; ring=2; cham=7; break;} //  (7,  9, 10,11), --channel_7  OV1B_5 chamber_ME1/2/7  layer_9 input 10,11
          case ( 8) : { stat=1; ring=2; cham=8; break;} //  (8,  9, 12,13), --channel_8  OV1B_6 chamber_ME1/2/8  layer_9 input 12,13
          case ( 9) : { stat=1; ring=3; cham=6; break;} //  (9,  6, 8, 9 ), --channel_9  OV1B_7 chamber_ME1/3/6  layer_6 input 8, 9 
          case (10) : { stat=1; ring=3; cham=7; break;} //  (10, 6, 10,11), --channel_10 OV1B_8 chamber_ME1/3/7  layer_6 input 10,11
          case (11) : { stat=1; ring=3; cham=8; break;} //  (11, 6, 12,13), --channel_11 OV1B_9 chamber_ME1/3/8  layer_6 input 12,13
          case (12) : { stat=2; ring=2; cham=3; break;} //  (12, 7, 2, 3 ), --channel_0  OV2_4  chamber_ME2/2/3  layer_7 input 2, 3
          case (13) : { stat=2; ring=2; cham=4; break;} //  (13, 7, 4, 5 ), --channel_1  OV2_5  chamber_ME2/2/4  layer_7 input 4, 5
          case (14) : { stat=2; ring=2; cham=5; break;} //  (14, 7, 6, 7 ), --channel_2  OV2_6  chamber_ME2/2/5  layer_7 input 6, 7
          case (15) : { stat=2; ring=2; cham=6; break;} //  (15, 7, 8, 9 ), --channel_3  OV2_7  chamber_ME2/2/6  layer_7 input 8, 9 
          case (16) : { stat=2; ring=2; cham=7; break;} //  (16, 7, 10,11), --channel_4  OV2_8  chamber_ME2/2/7  layer_7 input 10,11
          case (17) : { stat=2; ring=2; cham=8; break;} //  (17, 7, 12,13), --channel_5  OV2_9  chamber_ME2/2/8  layer_7 input 12,13
          case (18) : { stat=3; ring=2; cham=3; break;} //  (18, 8, 2, 3 ), --channel_6  OV3_4  chamber_ME3/2/3  layer_8 input 2, 3 
          case (19) : { stat=3; ring=2; cham=4; break;} //  (19, 8, 4, 5 ), --channel_7  OV3_5  chamber_ME3/2/4  layer_8 input 4, 5 
          case (20) : { stat=3; ring=2; cham=5; break;} //  (20, 8, 6, 7 ), --channel_8  OV3_6  chamber_ME3/2/5  layer_8 input 6, 7 
          case (21) : { stat=3; ring=2; cham=6; break;} //  (21, 8, 8, 9 ), --channel_9  OV3_7  chamber_ME3/2/6  layer_8 input 8, 9 
          case (22) : { stat=3; ring=2; cham=7; break;} //  (22, 8, 10,11), --channel_10 OV3_8  chamber_ME3/2/7  layer_8 input 10,11
          case (23) : { stat=3; ring=2; cham=8; break;} //  (23, 8, 12,13), --channel_11 OV3_9  chamber_ME3/2/8  layer_8 input 12,13
          case (24) : { stat=4; ring=2; cham=3; break;} //--(24,  ,      ), --channel_3  OV4_4  chamber_ME4/2/3  layer   input       
          case (25) : { stat=4; ring=2; cham=4; break;} //--(25,  ,      ), --channel_4  OV4_5  chamber_ME4/2/4  layer   input       
          case (26) : { stat=4; ring=2; cham=5; break;} //--(26,  ,      ), --channel_5  OV4_6  chamber_ME4/2/5  layer   input       
          case (27) : { stat=4; ring=2; cham=6; break;} //--(27,  ,      ), --channel_7  OV4_7  chamber_ME4/2/6  layer   input       
          case (28) : { stat=4; ring=2; cham=7; break;} //--(28,  ,      ), --channel_8  OV4_8  chamber_ME4/2/7  layer   input      
          case (29) : { stat=4; ring=2; cham=8; break;} //--(29,  ,      ), --channel_9  OV4_9  chamber_ME4/2/8  layer   input      
          case (30) : { stat=1; ring=2; cham=2; break;} //  (30, 9, 0, 1 ), --channel_0  OV1B_6 chamber_ME1/2/2  layer_9 input 0, 1 
          case (31) : { stat=1; ring=3; cham=2; break;} //  (31, 6, 0, 1 ), --channel_1  OV1B_9 chamber_ME1/3/2  layer_6 input 0, 1 
          case (32) : { stat=2; ring=2; cham=2; break;} //  (32, 7, 0, 1 ), --channel_2  OV2_9  chamber_ME2/2/2  layer_7 input 0, 1 
          case (33) : { stat=3; ring=2; cham=2; break;} //  (33, 8, 0, 1 ), --channel_3  ON3_9  chamber_ME3/2/2  layer_8 input 0, 1 
          case (34) : { stat=4; ring=2; cham=2; break;} //--(34,  ,      ), --channel_4  ON4_9  chamber_ME4/2/2  layer   input      
          default   : { stat=0; ring=0; cham=0; break;}
        }
        if (ring !=0) {
          int chamber = cham+(amc-1)*6; 
          if (chamber > 36) chamber -= 36;
          CSCDetId cscDetId(endcap, stat, ring, chamber);
//          std::cout <<" INIT CSC DET ID: "<< cscDetId << std::endl;
          OmtfEleIndex omtfEle(fed, amc, link);
          omtf2csc_[omtfEle]=cscDetId;
        }
      }
    }
  }
  LogTrace(" ") << " SIZE OF OMTF to CSC map  is: " << omtf2csc_.size() << std::endl;

}

void OmtfUnpacker::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("InputLabel",edm::InputTag("rawDataCollector"));
  desc.add<bool>("useRpcConnectionFile",bool(false));
  desc.add<std::string>("rpcConnectionFile",std::string("EventFilter/L1TRawToDigi/data/OmtfRpcLinksMap.txt"));
  descriptions.add("omtfUnpacker",desc);
}

void OmtfUnpacker::produce(edm::Event& event, const edm::EventSetup& setup)
{
  bool debug = edm::MessageDrop::instance()->debugEnabled;
  eventCounter_++;
  if (debug) LogDebug ("OmtfUnpacker::produce") <<"Beginning To Unpack Event: "<<eventCounter_;

  edm::Handle<FEDRawDataCollection> allFEDRawData;
  event.getByToken(fedToken_,allFEDRawData);

  auto producedRPCDigis = std::make_unique<RPCDigiCollection>();
  auto producedCscLctDigis = std::make_unique<CSCCorrelatedLCTDigiCollection>();
  auto producedMuonDigis = std::make_unique<l1t::RegionalMuonCandBxCollection>(); 
  producedMuonDigis->setBXRange(-3,3);
  auto producedDTPhDigis = std::make_unique<L1MuDTChambPhContainer>();
  auto producedDTThDigis = std::make_unique<L1MuDTChambThContainer>();
  std::vector<L1MuDTChambPhDigi> phi_Container;
  std::vector<L1MuDTChambThDigi> the_Container;

  for (int fedId= 1380; fedId<= 1381; ++fedId) {

    const FEDRawData & rawData = allFEDRawData->FEDData(fedId);
    int nWords = rawData.size()/sizeof(Word64);
    LogTrace("") <<"FED : " << fedId <<" words: " << nWords;
    if (nWords==0) continue;

    int triggerBX =0;
   
    //
    // FED header
    // Expect one FED header + AMC13 headers.
    const Word64* header = reinterpret_cast<const Word64* >(rawData.data());
    FEDHeader fedHeader( reinterpret_cast<const unsigned char*>(header));
    if (!fedHeader.check()) { LogTrace("") <<" ** PROBLEM **, header.check() failed, break"; break; }
    if ( fedHeader.sourceID() != fedId) {
      LogTrace("") <<" ** PROBLEM **, fedHeader.sourceID() != fedId"
          << "fedId = " << fedId<<" sourceID="<<fedHeader.sourceID();
    }
    triggerBX = fedHeader.bxID();
    if (debug) {
      std::ostringstream str;
      str <<"  header: "<< *reinterpret_cast<const std::bitset<64>*> (header) << std::endl;
      str <<"  header triggerType: " << fedHeader.triggerType()<< std::endl;
      str <<"  header lvl1ID:      " << fedHeader.lvl1ID() << std::endl;
      str <<"  header bxID:        " << fedHeader.bxID() << std::endl;
      str <<"  header sourceID:    " << fedHeader.sourceID() << std::endl;
      str <<"  header version:     " << fedHeader.version() << std::endl;
      str <<"  header more :     " << fedHeader.moreHeaders() << std::endl;
      str << " triggerBx "<< triggerBX << std::endl;
      LogTrace("") << str.str();
    }
    // AMC13 headers


    // 
    // FED trailed
    //
    const Word64* trailer=reinterpret_cast<const Word64* >(rawData.data())+(nWords-1); trailer++;
    bool moreTrailers = true;
    while (moreTrailers) {
      trailer--;
      FEDTrailer fedTrailer(reinterpret_cast<const unsigned char*>(trailer));
      if ( !fedTrailer.check()) {
        if (debug) LogTrace("") <<" ** PROBLEM **, trailer.check() failed, break";
        break;
      }
      if ( fedTrailer.lenght()!= nWords) {
        if (debug) LogTrace("")<<" ** PROBLEM **, fedTrailer.lenght()!= nWords, break";
        break;
      }
      moreTrailers = fedTrailer.moreTrailers();
      if (debug) {
        std::ostringstream str;
        str <<" trailer: "<<  *reinterpret_cast<const std::bitset<64>*> (trailer) << std::endl;
        str <<"  trailer lenght:    "<<fedTrailer.lenght()<< std::endl;
        str <<"  trailer crc:       "<<fedTrailer.crc()<< std::endl;
        str <<"  trailer evtStatus: "<<fedTrailer.evtStatus()<< std::endl;
        str <<"  trailer ttsBits:   "<<fedTrailer.ttsBits()<< std::endl;
        LogTrace("") << str.str();
      }
    }

    if (debug) {
      std::ostringstream str;
      for (const Word64* word = header+1; word != trailer; word++) {
        str<<"    data: "<<*reinterpret_cast<const std::bitset<64>*>(word) << std::endl;
      }
      LogTrace("") << str.str();
    }

    //
    // AMC13 header
    //
    const Word64* headerAmc13raw = header+1;
    amc13::Header headerAmc13(headerAmc13raw);
    if (debug) {
      std::ostringstream str;
      str <<" headerAMC13:  "<<  *reinterpret_cast<const std::bitset<64>*> (headerAmc13raw) << std::endl;
      str <<" amc13 check:  "<< headerAmc13.check() << std::endl;
      str <<" amc13 format: "<< headerAmc13.getFormatVersion() << std::endl;
      str <<" amc13 nAMCs:  "<< headerAmc13.getNumberOfAMCs() << std::endl;
      str <<" amc13 orbit:  "<< headerAmc13.getOrbitNumber() << std::endl;
      LogTrace("") << str.str();
    }


    //unsigned int nAMCs = headerAmc13.getNumberOfAMCs();
    //for (unsigned int iAMC = 0; iAMC <nAMCs; iAMC++) {
    // const Word64* raw = header+1 +(iAMC+1);
    //}
   

    //
    // AMC13 trailer
    //
    const Word64* trailerAmc13raw = trailer-1;
    amc13::Trailer trailerAmc13(trailerAmc13raw);
    if (debug) {
      std::ostringstream str;
      str <<" trailerAMC13:  "<<  *reinterpret_cast<const std::bitset<64>*> (trailerAmc13raw) << std::endl;
      str <<" crc:  "<< trailerAmc13.getCRC() << std::endl;
      str <<" block: "<< trailerAmc13.getBlock() << std::endl;
      str <<" LV1ID:  "<< trailerAmc13.getLV1ID() << std::endl;
      str <<" BX:  "<< trailerAmc13.getBX() << std::endl;
      LogTrace("") << str.str();
    }

    amc13::Packet packetAmc13;
    if (!packetAmc13.parse( header, header+1, nWords-2, fedHeader.lvl1ID(), fedHeader.bxID(), 1, 0)) {
      edm::LogError("OMTF") << "Could not extract AMC13 Packet.";
      return;
    } 
//    std::cout <<"AMC13 Packet: "<< packetAmc13.blocks() << " size "<<packetAmc13.size() << std::endl;  
    unsigned int blockNum=0;
    for (auto amc: packetAmc13.payload()) {
//      amc.finalize(fedHeader.lvl1ID(), fedHeader.bxID(), true, false);
      amc::BlockHeader bh =  amc.blockHeader(); 
      if (debug) {
        std::ostringstream str;
        str <<" ----------- #"<<blockNum++ << std::endl;
        str <<" blockheader:  "<<  std::bitset<64>(bh.raw()) << std::endl;
        str <<" boardID:  "<< bh.getBoardID() << std::endl;
        str <<" amcNumber:  "<< bh.getAMCNumber() << std::endl;
        str <<" size:  "<< bh.getSize(); // << std::endl;
        LogTrace("") << str.str();
      }

      //
      // AMC header
      //
      amc::Header headerAmc = amc.header(); 
      if (debug) {
        std::ostringstream str;
        str <<" AMC header[0]:  "<<  std::bitset<64>(headerAmc.raw()[0]) << std::endl;
        str <<" AMC header[1]:  "<<  std::bitset<64>(headerAmc.raw()[1]) << std::endl;
        str <<" AMC number:     "<< headerAmc.getAMCNumber(); // << std::endl;
        LogTrace("") << str.str();
      }


      //
      // AMC trailer
      //
      //amc::Trailer trailerAmc = amc.trailer();              //this is the expected way but does not work 
      amc::Trailer trailerAmc(amc.data().get()+amc.size()-1); //FIXME: the above is prefered but this works (CMSSW900)
      if (debug) {
        std::ostringstream str;
        str <<" AMC trailer:  "<<  std::bitset<64>(trailerAmc.raw()) << std::endl;
        str <<" getLV1ID:     "<< trailerAmc.getLV1ID() << std::endl;
        str <<" size:         "<< trailerAmc.getSize()  << std::endl;
        LogTrace("") << str.str();
      }

      //
      // AMC payload
      //
      const auto & payload64 = amc.data();
      const Word64* word = payload64.get();
      for (unsigned int iWord= 1; iWord<= amc.size(); iWord++, word++) {
        if (iWord<=2 ) continue; // two header words for each AMC
        if (iWord==amc.size() ) continue; // trailer for each AMC 

        LogTrace("") <<" payload: " <<  *reinterpret_cast<const std::bitset<64>*>(word);
        DataWord64::Type recordType = DataWord64::type(*word); 


        //
        // RPC data
        // 
        if (DataWord64::rpc==recordType) {
          RpcDataWord64 data(*word);
          LogTrace("") << data;
  
          OmtfEleIndex omtfEle(fedHeader.sourceID(), bh.getAMCNumber()/2+1, data.linkNum());
          LinkBoardElectronicIndex rpcEle = omtf2rpc_.at(omtfEle);
          RPCRecordFormatter formater(rpcEle.dccId, theCabling);
  
  
          rpcrawtodigi::EventRecords records(triggerBX);
          rpcrawtodigi::RecordBX recordBX(triggerBX+data.bxNum()-3);
          records.add(recordBX);   // warning: event records must be added in right order
          rpcrawtodigi::RecordSLD recordSLD(rpcEle.tbLinkInputNum, rpcEle.dccInputChannelNum);
          records.add(recordSLD); // warning: event records must be added in right order
  
          for (unsigned int iframe=1; iframe <=3; iframe++) {
  
            uint16_t frame = (iframe==1) ?  data.frame1() : ( (iframe==2) ?  data.frame2() : data.frame3() );
            if (frame==0) continue;
            rpcrawtodigi::RecordCD recordCD(frame);
            records.add(recordCD);
  
            LogTrace("") <<"OMTF->RPC Event isComplete: "<<records.complete() <<records.print(rpcrawtodigi::DataRecord::StartOfBXData); // << std::endl; 
            LogTrace("") <<"OMTF->RPC Event:             "<<records.print(rpcrawtodigi::DataRecord::StartOfTbLinkInputNumberData) << std::endl; 
            LogTrace("") <<"OMTF->RPC Event:             "<<records.print(rpcrawtodigi::DataRecord::ChamberData)
                      <<" lb:"<< recordCD.lbInLink() 
                      <<" part: "<< recordCD.partitionNumber() 
                      <<" partData: "<<recordCD.partitionData() 
                      << std::endl << std::endl;
  
            if (records.complete()) formater.recordUnpack( records,  producedRPCDigis.get(), 0,0);
          }
        }



        //
        // CSC data
        //
        if (DataWord64::csc==recordType) {
          CscDataWord64   data(*word);
          OmtfEleIndex omtfEle(fedHeader.sourceID(), bh.getAMCNumber()/2+1, data.linkNum());
          std::map<OmtfEleIndex,CSCDetId>::const_iterator icsc = omtf2csc_.find(omtfEle);
          if (icsc==omtf2csc_.end()) {LogTrace(" ") <<" CANNOT FIND key: " << omtfEle << std::endl; continue; }
          CSCDetId cscId = omtf2csc_[omtfEle];
          LogTrace("") <<"OMTF->CSC "<<cscId << std::endl; 
          LogTrace("") << data << std::endl;
          CSCCorrelatedLCTDigi digi(data.hitNum(), //trknmb
                                    data.valid(), 
                                    data.quality(),
                                    data.wireGroup(),
                                    data.halfStrip(),
                                    data.clctPattern(),
                                    data.bend(),
                                    data.bxNum()+3) ;
          LogTrace("") << digi << std::endl;
          producedCscLctDigis->insertDigi( cscId, digi); 

        } 

        //
        // OMTF (muon) data
        //
        if (DataWord64::omtf==recordType) {
          OmtfDataWord64   data(*word);
          LogTrace("") <<"OMTF->MUON " << std::endl;
          LogTrace("") << data << std::endl;
          l1t::tftype  overlap = (fedId==1380) ? l1t::tftype::omtf_neg :  l1t::tftype::omtf_pos;
          unsigned int iProcessor = bh.getAMCNumber()/2;   //0-5 
          l1t::RegionalMuonCand digi;
          digi.setHwPt(data.pT());
          digi.setHwEta(data.eta());
          digi.setHwPhi(data.phi());
          digi.setHwSign(data.ch());
          digi.setHwSignValid(data.vch());
          digi.setHwQual(data.quality());
          std::map<int, int> trackAddr;
          trackAddr[0]=data.layers();
          trackAddr[1]=0;
          trackAddr[2]=data.weight_lowBits();
          digi.setTrackAddress(trackAddr);
          digi.setTFIdentifiers(iProcessor, overlap);
          int bx = data.bxNum()-3;
          LogTrace("")  <<"OMTF Muon, BX="<<bx<<", hwPt="<<digi.hwPt()<< std::endl;
          if(std::abs(bx) <= 3) producedMuonDigis->push_back(bx,digi);
        }


        //
        // DT data
        //
        if (DataWord64::dt==recordType) {
          DtDataWord64   data(*word);
          LogTrace("") <<"OMTF->DT " << std::endl;
          LogTrace("") << data << std::endl;
          int bx = data.bxNum()-3;
          int whNum = (fedId==1380) ? -2 : 2;
          int sector =   (bh.getAMCNumber()/2)*2 + data.sector();
          if (sector==12) sector=0;
          int station =  data.station()+1;
//  unsigned int valid() const { return valid_; }
//  unsigned int fiber() const { return fiber_; }
//          if(data.valid())
           phi_Container.push_back( L1MuDTChambPhDigi( bx, whNum, sector, station,
                                    data.phi(), data.phiB(), data.quality(), 
                                    0,                // utag/Ts2Tag 
                                    data.bcnt_st())); //ucnt/BxCnt  
          int pos[7];
          int posQual[7];
          for (unsigned int i=0; i<7; i++) { pos[i] = (data.eta() >> i & 1); posQual[i] = (data.etaQuality() >> i & 1); }
            if(data.eta())the_Container.push_back(L1MuDTChambThDigi(bx,whNum, sector, station, pos, posQual)); 
        }

      }
    }         

  } 
  event.put(std::move(producedRPCDigis),"OmtfUnpack");
  event.put(std::move(producedCscLctDigis),"OmtfUnpack");
  event.put(std::move(producedMuonDigis),"OmtfUnpack"); 
  producedDTPhDigis->setContainer(phi_Container);
  event.put(std::move(producedDTPhDigis),"OmtfUnpack");
  producedDTThDigis->setContainer(the_Container);
  event.put(std::move(producedDTThDigis),"OmtfUnpack");


}

DEFINE_FWK_MODULE(OmtfUnpacker);
