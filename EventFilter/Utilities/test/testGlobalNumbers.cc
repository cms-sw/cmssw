/** \file
 * 
 * 
 * \author N. Amapane - S. Argiro'
 *
*/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/FED1024.h"

#include "boost/date_time/posix_time/posix_time.hpp"

#include <iomanip>
#include <sstream>
#include <string>
#include <time.h>


namespace test{

  static const unsigned int GTEVMId= 812;
  static const unsigned int GTPEId= 814;
  class GlobalNumbersAnalysis: public edm::one::EDAnalyzer<> {
    private:
    edm::EDGetTokenT<FEDRawDataCollection> m_fedRawDataCollectionToken;
    public:
    GlobalNumbersAnalysis(const edm::ParameterSet& pset):
      m_fedRawDataCollectionToken( consumes<FEDRawDataCollection>( pset.getUntrackedParameter<edm::InputTag>( "inputTag", edm::InputTag( "source" ) ) ) ) {
    }
 
    void analyze(const edm::Event & e, const edm::EventSetup& c) {
      boost::posix_time::ptime event_timestamp;
      event_timestamp =  boost::posix_time::from_time_t( (time_t)e.time().unixTime() );
      event_timestamp += boost::posix_time::microseconds( e.time().microsecondOffset() );
      edm::LogInfo( "GlobalNumbersAnalysis" ) << "--- Run: " << e.id().run()
                                              << " LS: " << e.luminosityBlock() 
                                              << " Event: " << e.id().event()
                                              << " Timestamp: " << e.time().unixTime() << "." << e.time().microsecondOffset()
                                              << " (" << boost::posix_time::to_iso_extended_string( event_timestamp ) << " UTC)"
                                              << " Type: " << e.experimentType() << std::endl;
      edm::Handle<FEDRawDataCollection> rawdata;
      e.getByToken(m_fedRawDataCollectionToken,rawdata);
      const FEDRawData& data = rawdata->FEDData(GTEVMId);
      size_t size=data.size();

      if( size>0 ) {
        edm::LogInfo( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << GTEVMId << " " << std::setw(8) << size << " bytes " << std::endl;
        if( evf::evtn::evm_board_sense( data.data(), size ) ) {
          edm::LogInfo( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << GTEVMId << " is the real GT EVM block"
                                                 << "\nEvent # " << evf::evtn::get(data.data(),true)
                                                 << "\nLS # " << evf::evtn::getlbn(data.data())
                                                 << "\nORBIT # " << evf::evtn::getorbit(data.data())
                                                 << "\nGPS LOW # " << evf::evtn::getgpslow(data.data())
                                                 << "\nGPS HI # " << evf::evtn::getgpshigh(data.data())
                                                 << "\nBX FROM FDL 0-xing # " << evf::evtn::getfdlbx(data.data())
                                                 << "\nPRESCALE INDEX FROM FDL 0-xing # " << evf::evtn::getfdlpsc(data.data()) << std::endl;
	  }
      } else {
        edm::LogWarning( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << GTEVMId << " not read out." << std::endl;
      }

      const FEDRawData& data2 = rawdata->FEDData(GTPEId);
      size=data2.size();

      if( size>0 ) {
        edm::LogInfo( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << GTPEId << " " << std::setw(8) << size << " bytes " << std::endl;
        if( evf::evtn::gtpe_board_sense( data2.data() ) ) {
          edm::LogInfo( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << GTPEId << " is the real GTPE block"
                                                 << "\nEvent # " << evf::evtn::gtpe_get(data2.data())
                                                 << "\nLS # " << evf::evtn::gtpe_getlbn(data2.data())
                                                 << "\nORBIT # " << evf::evtn::gtpe_getorbit(data2.data())
                                                 << "\nBX # " << evf::evtn::gtpe_getbx(data2.data()) << std::endl;
        }
      } else {
        edm::LogWarning( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << GTPEId << " not read out." << std::endl;
      }

      const FEDRawData& data3 = rawdata->FEDData(FEDNumbering::MINTCDSuTCAFEDID);
      size=data3.size();

      if( size>0 ) {
        evf::evtn::TCDSRecord record(data3.data());
        edm::LogInfo( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << FEDNumbering::MINTCDSuTCAFEDID << " " << std::setw(8) << size << " bytes " << std::endl;
        edm::LogInfo( "GlobalNumberAnalysis" ) << "sizes: "
                                               << " BGOSize " << std::hex << (unsigned int) record.getHeader().getSizes().size.BGOSize  
                                               << "  reserved2;" << std::hex << (unsigned int)record.getHeader().getSizes().size.reserved2
                                               << "  reserved1;" << std::hex << (unsigned int) record.getHeader().getSizes().size.reserved1
                                               << "  reserved0;" << std::hex << (unsigned int)record.getHeader().getSizes().size.reserved0
                                               << "  BSTSize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.BSTSize
                                               << "  L1AhistSize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.L1AhistSize
                                               << "  summarySize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.summarySize
                                               << "  headerSize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.headerSize
                                               << std::endl;
        edm::LogInfo( "GlobalNumberAnalysis" ) << "macAddress;          " << std::hex << (uint64_t)record.getHeader().getData().header.macAddress
                                               << "\nsw;		   " << std::hex << (unsigned int)record.getHeader().getData().header.sw
                                               << "\nfw;		   " << std::hex <<(unsigned int)record.getHeader().getData().header.fw
                                               << "\nreserved0;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved0	
                                               << "\nformat;		   " << std::hex <<(unsigned int)record.getHeader().getData().header.format
                                               << "\nrunNumber;	   " << std::dec << (unsigned int)record.getHeader().getData().header.runNumber
                                               << "\nreserved1;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved1	
                                               << "\nactivePartitions2;   " << std::hex <<(unsigned int)record.getHeader().getData().header.activePartitions2 
                                               << "\nreserved2;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved2
                                               << "\nactivePartitions0;   " << std::hex << (unsigned int)record.getHeader().getData().header.activePartitions0
                                               << "\nactivePartitions1;   " << std::hex <<(unsigned int)record.getHeader().getData().header.activePartitions1
                                               << "\nnibble;		   " << std::dec << (unsigned int)record.getHeader().getData().header.nibble
                                               << "\nlumiSection;	   " << std::dec << (unsigned int)record.getHeader().getData().header.lumiSection
                                               << "\nnibblesPerLumiSection;" << std::hex <<(unsigned int)record.getHeader().getData().header.nibblesPerLumiSection
                                               << "\ntriggerTypeFlags;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.triggerTypeFlags
                                               << "\nreserved5;	   " << std::hex <<(unsigned int)record.getHeader().getData().header.reserved5
                                               << "\ninputs;		   " << std::hex <<(unsigned int)record.getHeader().getData().header.inputs
                                               << "\nbcid;		   " << std::dec << (unsigned int)record.getHeader().getData().header.bcid
                                               << "\norbitLow;		   " << std::dec << (unsigned int)record.getHeader().getData().header.orbitLow
                                               << "\norbitHigh;	   " << std::dec << (unsigned int)record.getHeader().getData().header.orbitHigh
                                               << "\ntriggerCount;	   " << std::dec << (uint64_t)record.getHeader().getData().header.triggerCount
                                               << "\neventNumber;         " << std::dec << (uint64_t)record.getHeader().getData().header.eventNumber
                                               << std::endl;

        std::ostringstream osl1ah;
        osl1ah << "====================l1a history===================";
        const evf::evtn::TCDSL1AHistory::l1a *history = record.getHistory().history().hist;
        for(unsigned int i = 0; i < 16; i++){
          osl1ah << "\n" << i << " " << std::hex << history[i].bxid
                 << "\n" << i << " " << std::hex << history[i].orbitlow
                 << "\n" << i << " " << std::hex << history[i].orbithigh
                 << "\n" << i << " " << std::hex << (unsigned int)history[i].eventtype;
        }
        edm::LogInfo( "GlobalNumberAnalysis" ) << osl1ah.str();

        edm::LogInfo( "GlobalNumberAnalysis" ) << "====================BST record==================="
                                               << "\ngpstimehigh;" << std::setw(19) << std::hex << record.getBST().getBST().bst.gpstimehigh
                                               << "\ngpstimelow;" << std::setw(20) << std::hex << record.getBST().getBST().bst.gpstimelow
                                               << "\nbireserved8_11;" << std::setw(16) << record.getBST().getBST().bst.bireserved8_11
                                               << "\nbireserved12_15;" << std::setw(15) << record.getBST().getBST().bst.bireserved12_15
                                               << "\nbstMaster_bireserved16;" << std::setw(8) << record.getBST().getBST().bst.bstMaster_bireserved16
                                               << "\nturnCountLow;" << std::setw(18) << record.getBST().getBST().bst.turnCountLow
                                               << "\nturnCountHigh;" << std::setw(17) << record.getBST().getBST().bst.turnCountHigh
                                               << "\nlhcFillLow;" << std::setw(20) << record.getBST().getBST().bst.lhcFillLow
                                               << "\nlhcFillHigh;" << std::setw(19) << record.getBST().getBST().bst.lhcFillHigh
                                               << "\nbeamMode;" << std::setw(22) << record.getBST().getBST().bst.beamMode
                                               << "\nparticleTypes;" << std::setw(17) << record.getBST().getBST().bst.particleTypes
                                               << "\nbeamMomentum;" << std::setw(18) << record.getBST().getBST().bst.beamMomentum
                                               << "\nintensityBeam1;" << std::setw(16) << record.getBST().getBST().bst.intensityBeam1
                                               << "\nintensityBeam2;" << std::setw(16) << record.getBST().getBST().bst.intensityBeam2
                                               << "\nbireserved40_43;" << std::setw(15) << record.getBST().getBST().bst.bireserved40_43
                                               << "\nbireserved44_47;" << std::setw(15) << record.getBST().getBST().bst.bireserved44_47
                                               << "\nbireserved48_51;" << std::setw(15) << record.getBST().getBST().bst.bireserved48_51
                                               << "\nbireserved52_55;" << std::setw(15) << record.getBST().getBST().bst.bireserved52_55
                                               << "\nbireserved56_59;" << std::setw(15) << record.getBST().getBST().bst.bireserved56_59
                                               << "\nbireserved60_63;" << std::setw(15) << record.getBST().getBST().bst.bireserved60_63
                                               << std::endl;

        time_t nowtime = (time_t)record.getBST().getBST().bst.gpstimehigh;
        uint32_t turnCountHigh = record.getBST().getBST().bst.turnCountHigh;
        uint16_t turnCountLow = record.getBST().getBST().bst.turnCountLow;
        uint32_t lhcFillHigh = record.getBST().getBST().bst.lhcFillHigh;
        uint16_t lhcFillLow = record.getBST().getBST().bst.lhcFillLow;
        edm::LogInfo( "GlobalNumberAnalysis" ) << "value of nowtime: hex " << std::hex << nowtime << std::dec << ", dec " << nowtime 
                                               << "\nGPS time " << ctime(&nowtime) << "plus microseconds: " << std::dec << record.getBST().getBST().bst.gpstimelow
                                               << "\nBeam " << (record.getBST().getBST().bst.bstMaster_bireserved16 >> 8)
                                               << " master sent turn count " << (uint32_t)((turnCountHigh << 16) + turnCountLow)
                                               << "\nFill: " << (uint32_t)((lhcFillHigh << 16) + lhcFillLow)
                                               << "\nBeam Mode: " << record.getBST().getBST().bst.beamMode
                                               << "\nparticleType1: " << (record.getBST().getBST().bst.particleTypes & 0xFF)
                                               << ", particleType2: " << (record.getBST().getBST().bst.particleTypes >> 8)
                                               << "\nBeam Momentum: " << record.getBST().getBST().bst.beamMomentum << " GeV/c"
                                               << "\nB1 intensity: (10E10 charges) " << record.getBST().getBST().bst.intensityBeam1
                                               << ", B2 intensity: (10E10 charges) " << record.getBST().getBST().bst.intensityBeam2 << std::endl;
      } else {
        edm::LogInfo( "GlobalNumberAnalysis" ) << "FED# " << std::setw(4) << FEDNumbering::MINTCDSuTCAFEDID << " not read out." << std::endl;
      }


// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
    }
  };
DEFINE_FWK_MODULE(GlobalNumbersAnalysis);
}

