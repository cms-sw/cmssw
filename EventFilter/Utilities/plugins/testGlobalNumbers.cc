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
        std::cout << "FED# " << std::setw(4) << FEDNumbering::MINTCDSuTCAFEDID << " " << std::setw(8) << size << " bytes " << std::endl;
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
        std::cout << "gpstimehigh;" << std::setw(19) << std::hex << record.getBST().getBST().bst.gpstimehigh;
        std::cout << "\n";
        std::cout << "gpstimelow;" << std::setw(20) << std::hex << record.getBST().getBST().bst.gpstimelow;
        std::cout << "\n";
        std::cout << "bireserved8_11;" << std::setw(16) << record.getBST().getBST().bst.bireserved8_11;
        std::cout << "\n";
        std::cout << "bireserved12_15;" << std::setw(15) << record.getBST().getBST().bst.bireserved12_15;
        std::cout << "\n";
        std::cout << "bstMaster_bireserved16;" << std::setw(8) << record.getBST().getBST().bst.bstMaster_bireserved16;
        std::cout << "\n";
        std::cout << "turnCountLow;" << std::setw(18) << record.getBST().getBST().bst.turnCountLow;
        std::cout << "\n";
        std::cout << "turnCountHigh;" << std::setw(17) << record.getBST().getBST().bst.turnCountHigh;
        std::cout << "\n";
        std::cout << "lhcFillLow;" << std::setw(20) << record.getBST().getBST().bst.lhcFillLow;
        std::cout << "\n";
        std::cout << "lhcFillHigh;" << std::setw(19) << record.getBST().getBST().bst.lhcFillHigh;
        std::cout << "\n";
        std::cout << "beamMode;" << std::setw(22) << record.getBST().getBST().bst.beamMode;
        std::cout << "\n";
        std::cout << "particleTypes;" << std::setw(17) << record.getBST().getBST().bst.particleTypes;
        std::cout << "\n";
        std::cout << "beamMomentum;" << std::setw(18) << record.getBST().getBST().bst.beamMomentum;
        std::cout << "\n";
        std::cout << "intensityBeam1;" << std::setw(16) << record.getBST().getBST().bst.intensityBeam1;
        std::cout << "\n";
        std::cout << "intensityBeam2;" << std::setw(16) << record.getBST().getBST().bst.intensityBeam2;
        std::cout << "\n";
        std::cout << "bireserved40_43;" << std::setw(15) << record.getBST().getBST().bst.bireserved40_43;
        std::cout << "\n";
        std::cout << "bireserved44_47;" << std::setw(15) << record.getBST().getBST().bst.bireserved44_47;
        std::cout << "\n";
        std::cout << "bireserved48_51;" << std::setw(15) << record.getBST().getBST().bst.bireserved48_51;
        std::cout << "\n";
        std::cout << "bireserved52_55;" << std::setw(15) << record.getBST().getBST().bst.bireserved52_55;
        std::cout << "\n";
        std::cout << "bireserved56_59;" << std::setw(15) << record.getBST().getBST().bst.bireserved56_59;
        std::cout << "\n";
        std::cout << "bireserved60_63;" << std::setw(15) << record.getBST().getBST().bst.bireserved60_63;
        std::cout << std::endl;

        time_t nowtime = (time_t)record.getBST().getBST().bst.gpstimehigh;
        std::cout << "value of nowtime: hex " << std::hex << nowtime << std::dec << ", dec " << nowtime << std::endl; 
        std::cout << "GPS time " << ctime(&nowtime) << "plus microseconds: " << std::dec << record.getBST().getBST().bst.gpstimelow << std::endl;
        uint32_t turnCountHigh = record.getBST().getBST().bst.turnCountHigh;
        uint16_t turnCountLow = record.getBST().getBST().bst.turnCountLow;
        std::cout << "Beam " << (record.getBST().getBST().bst.bstMaster_bireserved16 >> 8)
                  << " master sent turn count " << (uint32_t)((turnCountHigh << 16) + turnCountLow) << std::endl;
        uint32_t lhcFillHigh = record.getBST().getBST().bst.lhcFillHigh;
        uint16_t lhcFillLow = record.getBST().getBST().bst.lhcFillLow;
        std::cout << "Fill: " << (uint32_t)((lhcFillHigh << 16) + lhcFillLow) << std::endl;
        std::cout << "Beam Mode: " << record.getBST().getBST().bst.beamMode << std::endl;
        std::cout << "particleType1: " << (record.getBST().getBST().bst.particleTypes & 0xFF)
                  << ", particleType2: " << (record.getBST().getBST().bst.particleTypes >> 8) << std::endl;
        std::cout << "Beam Momentum: " << record.getBST().getBST().bst.beamMomentum << " GeV/c" << std::endl;
        std::cout << "B1 intensity: (10E10 charges) " << record.getBST().getBST().bst.intensityBeam1
                  << ", B2 intensity: (10E10 charges) " << record.getBST().getBST().bst.intensityBeam2 << std::endl;
      } else {
        std::cout << "FED# " << std::setw(4) << FEDNumbering::MINTCDSuTCAFEDID << " not read out." << std::endl;
      }


// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
    }
  };
DEFINE_FWK_MODULE(GlobalNumbersAnalysis);
}

