/** \file
 * 
 * 
 * \author N. Amapane - S. Argiro'
 *
*/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/FED1024.h"

#include <iostream>
#include <iomanip>

#include <time.h>


namespace test{

  static const unsigned int GTEVMId= 812;
  static const unsigned int GTPEId= 814;
  class GlobalNumbersAnalysis: public edm::EDAnalyzer {
    private:
    edm::EDGetTokenT<FEDRawDataCollection> m_fedRawDataCollectionToken;
    public:
    GlobalNumbersAnalysis(const edm::ParameterSet& pset):
      m_fedRawDataCollectionToken( consumes<FEDRawDataCollection>( pset.getUntrackedParameter<edm::InputTag>( "inputTag", edm::InputTag( "source" ) ) ) ) {
    }
 
    void analyze(const edm::Event & e, const edm::EventSetup& c) {
      std::cout << "--- Run: " << e.id().run()
                << " LS: " << e.luminosityBlock() 
                << " Event: " << e.id().event() 
                << " Type: " << e.experimentType() << std::endl;
      edm::Handle<FEDRawDataCollection> rawdata;
      e.getByToken(m_fedRawDataCollectionToken,rawdata);
      const FEDRawData& data = rawdata->FEDData(GTEVMId);
      size_t size=data.size();

      if( size>0 ) {
        std::cout << "FED# " << std::setw(4) << GTEVMId << " " << std::setw(8) << size << " bytes " << std::endl;
	      if( evf::evtn::evm_board_sense( data.data(), size ) ) {
	        std::cout << "FED# " << std::setw(4) << GTEVMId << " is the real GT EVM block " << std::endl;
	        std::cout << "Event # " << evf::evtn::get(data.data(),true) << std::endl;
	        std::cout << "LS # " << evf::evtn::getlbn(data.data()) << std::endl;
	        std::cout << "ORBIT # " << evf::evtn::getorbit(data.data()) << std::endl;
	        std::cout << "GPS LOW # " << evf::evtn::getgpslow(data.data()) << std::endl;
	        std::cout << "GPS HI # " << evf::evtn::getgpshigh(data.data()) << std::endl;
	        std::cout << "BX FROM FDL 0-xing # " << evf::evtn::getfdlbx(data.data()) << std::endl;
	        std::cout << "PRESCALE INDEX FROM FDL 0-xing # " << evf::evtn::getfdlpsc(data.data()) << std::endl;
	      }
	    } else {
        std::cout << "FED# " << std::setw(4) << GTEVMId << " not read out." << std::endl;
      }

      const FEDRawData& data2 = rawdata->FEDData(GTPEId);
      size=data2.size();

      if( size>0 ) {
        std::cout << "FED# " << std::setw(4) << GTPEId << " " << std::setw(8) << size << " bytes " << std::endl;
	      if( evf::evtn::gtpe_board_sense( data2.data() ) ) {
	        std::cout << "FED# " << std::setw(4) << GTPEId << " is the real GTPE block " << std::endl;
	        std::cout << "Event # " << evf::evtn::gtpe_get(data2.data()) << std::endl;
	        std::cout << "LS # " << evf::evtn::gtpe_getlbn(data2.data()) << std::endl;
	        std::cout << "ORBIT # " << evf::evtn::gtpe_getorbit(data2.data()) << std::endl;
	        std::cout << "BX # " << evf::evtn::gtpe_getbx(data2.data()) << std::endl;
	      }
	    } else {
        std::cout << "FED# " << std::setw(4) << GTPEId << " not read out." << std::endl;
      }

      const FEDRawData& data3 = rawdata->FEDData(FEDNumbering::MINTCDSuTCAFEDID);
      size=data3.size();

      if( size>0 ) {
        evf::evtn::TCDSRecord record(data3.data());
	      std::cout << "FED# " << std::setw(4) << FEDNumbering::MINTCDSuTCAFEDID << " " 
	                << std::setw(8) << size << " bytes " << std::endl;
        std::cout << "sizes: " 
	                << " BGOSize " << std::hex << (unsigned int) record.getHeader().getSizes().size.BGOSize  
                  << "  reserved2;" << std::hex << (unsigned int)record.getHeader().getSizes().size.reserved2
	                << "  reserved1;" << std::hex << (unsigned int) record.getHeader().getSizes().size.reserved1
	                << "  reserved0;" << std::hex << (unsigned int)record.getHeader().getSizes().size.reserved0
	                << "  BSTSize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.BSTSize
	                << "  L1AhistSize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.L1AhistSize
	                << "  summarySize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.summarySize
	                << "  headerSize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.headerSize
                  << std::endl;
	      std::cout << "macAddress;          " << std::hex << (uint64_t)record.getHeader().getData().header.macAddress;
        std::cout << "\n";
        std::cout << "sw;		   " << std::hex << (unsigned int)record.getHeader().getData().header.sw;
	      std::cout << "\n";
	      std::cout << "fw;		   " << std::hex <<(unsigned int)record.getHeader().getData().header.fw;
	      std::cout << "\n";
	      std::cout << "reserved0;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved0;	
	      std::cout << "\n";
	      std::cout << "format;		   " << std::hex <<(unsigned int)record.getHeader().getData().header.format;
	      std::cout << "\n";
	      std::cout << "runNumber;	   " << std::dec << (unsigned int)record.getHeader().getData().header.runNumber;
	      std::cout << "\n";
	      std::cout << "reserved1;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved1;	
	      std::cout << "\n";
	      std::cout << "activePartitions2;   " << std::hex <<(unsigned int)record.getHeader().getData().header.activePartitions2; 
	      std::cout << "\n";
	      std::cout << "reserved2;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved2;	 
	      std::cout << "\n";
	      std::cout << "activePartitions0;   " << std::hex << (unsigned int)record.getHeader().getData().header.activePartitions0;
	      std::cout << "\n";
	      std::cout << "activePartitions1;   " << std::hex <<(unsigned int)record.getHeader().getData().header.activePartitions1;
	      std::cout << "\n"; 
	      std::cout << "nibble;		   " << std::dec << (unsigned int)record.getHeader().getData().header.nibble;
	      std::cout << "\n";
	      std::cout << "lumiSection;	   " << std::dec << (unsigned int)record.getHeader().getData().header.lumiSection;
	      std::cout << "\n";
	      std::cout << "nibblesPerLumiSection;" << std::hex <<(unsigned int)record.getHeader().getData().header.nibblesPerLumiSection;
        std::cout << "\n";
        std::cout << "triggerTypeFlags;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.triggerTypeFlags;
	      std::cout << "\n";
        std::cout << "reserved5;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved5;	
        std::cout << "\n";
        std::cout << "inputs;		   " << std::hex <<(unsigned int)record.getHeader().getData().header.inputs;
	      std::cout << "\n";
        std::cout << "bcid;		   " << std::dec << (unsigned int)record.getHeader().getData().header.bcid;
        std::cout << "\n";
	      std::cout << "orbitLow;		   " << std::dec << (unsigned int)record.getHeader().getData().header.orbitLow;	
	      std::cout << "\n";
        std::cout << "orbitHigh;	   " << std::dec << (unsigned int)record.getHeader().getData().header.orbitHigh;
        std::cout << "\n";
        std::cout << "triggerCount;	   " << std::dec << (uint64_t)record.getHeader().getData().header.triggerCount;
	      std::cout << "\n";
        std::cout << "eventNumber;         " << std::dec << (uint64_t)record.getHeader().getData().header.eventNumber;  
	      std::cout << "\n";
	      std::cout << std::endl;

	      std::cout << "====================l1a history===================" << std::endl;
	      const evf::evtn::TCDSL1AHistory::l1a *history = record.getHistory().history().hist;
	      for(unsigned int i = 0; i < 16; i++){
	        std::cout << i << " " << std::hex << history[i].bxid << std::endl;
	        std::cout << i << " " << std::hex << history[i].orbitlow << std::endl;
	        std::cout << i << " " << std::hex << history[i].orbithigh << std::endl;
	        std::cout << i << " " << std::hex << (unsigned int)history[i].eventtype << std::endl;
	      }

        std::cout << "====================BST record===================" << std::endl;
	      std::cout << "gpstimehigh; " << std::hex << record.getBST().getBST().gpstimehigh;
        std::cout << "\n";
	      std::cout << "gpstimelow; " << std::hex << record.getBST().getBST().gpstimelow;
	      std::cout << "\n";
        std::cout << "low0;   	" << record.getBST().getBST().low0;
	      std::cout << "\n";
        std::cout << "high0;  	" << record.getBST().getBST().high0;
	      std::cout << "\n";
        std::cout << "low1;   	" << record.getBST().getBST().low1;
	      std::cout << "\n";
        std::cout << "high1;  	" << record.getBST().getBST().high1;
        std::cout << "\n";
        std::cout << "low2;   	" << record.getBST().getBST().low2;
	      std::cout << "\n";
        std::cout << "high2;  	" << record.getBST().getBST().high2;
	      std::cout << "\n";
        std::cout << "low3;   	" << record.getBST().getBST().low3;
	      std::cout << "\n";
        std::cout << "high3;  	" << record.getBST().getBST().high3;
	      std::cout << "\n";
        std::cout << "low4;   	" << record.getBST().getBST().low4;
	      std::cout << "\n";
        std::cout << "high4;  	" << record.getBST().getBST().high4;
	      std::cout << "\n";
        std::cout << "low5;   	" << record.getBST().getBST().low5;
	      std::cout << "\n";
        std::cout << "status;  " << record.getBST().getBST().status;
	      std::cout << std::endl;

	      time_t nowtime = (time_t)record.getBST().getBST().gpstimehigh;
	      std::cout << "value of nowtime: hex " << std::hex << nowtime << std::dec << ", dec " << nowtime << std::endl; 
	      std::cout << "GPS time " << ctime(&nowtime) << "plus microseconds: " << std::dec << record.getBST().getBST().gpstimelow << std::endl;
        std::cout << "B1 intensity: (10E10 charges) " << record.getBST().getBST().low3  << ", B2 intensity: (10E10 charges) " << record.getBST().getBST().high3 << std::endl;
        uint32_t beamMomentum = record.getBST().getBST().high2 >> 16;
        uint32_t particleType2 = (record.getBST().getBST().high2 >> 8 ) & 0xFF;
        uint32_t particleType1 = 0xFF & record.getBST().getBST().high2;  
        std::cout << "Beam Momentum: " << beamMomentum << std::endl;
        std::cout << "particleType1: " << particleType1 << ", particleType2: " << particleType2 << std::endl;
        uint32_t beamMode = record.getBST().getBST().low2 >> 16;
        uint32_t fill = ( ( record.getBST().getBST().low2 & 0xFFFF ) << 16 ) | ( record.getBST().getBST().high1 >> 16 );
        std::cout << "Beam Mode: " << beamMode << std::endl;
        std::cout << "Fill: " << fill << std::endl;
        uint32_t turnCount = ( ( record.getBST().getBST().high1 & 0xFFFF ) << 16 ) | ( record.getBST().getBST().low1 >> 16 );
        uint32_t bstMaster = ( record.getBST().getBST().low1 >> 8 ) & 0xFF;
        std::cout << "Beam " << bstMaster << " master sent turn count " << turnCount << std::endl;
      } else {
        std::cout << "FED# " << std::setw(4) << FEDNumbering::MINTCDSuTCAFEDID << " not read out." << std::endl;
      }


// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
    }
  };
DEFINE_FWK_MODULE(GlobalNumbersAnalysis);
}

