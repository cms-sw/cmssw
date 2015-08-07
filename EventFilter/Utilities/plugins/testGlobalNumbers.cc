/** \file
 * 
 * 
 * \author N. Amapane - S. Argiro'
 *
*/

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "EventFilter/FEDInterface/interface/GlobalEventNumber.h"
#include "EventFilter/FEDInterface/interface/FED1024.h"

#include <iostream>
#include <iomanip>

#include <time.h>


using namespace edm;
using namespace std;

namespace test{

  static const unsigned int GTEVMId= 812;
  static const unsigned int GTPEId= 814;
  class GlobalNumbersAnalysis: public EDAnalyzer{
  private:
  public:
    GlobalNumbersAnalysis(const ParameterSet& pset){
    }

 
    void analyze(const Event & e, const EventSetup& c){
      cout << "--- Run: " << e.id().run()
	   << " LS: " << e.luminosityBlock() 
	   << " Event: " << e.id().event() 
	   << " Type: " << e.experimentType() << endl;
      Handle<FEDRawDataCollection> rawdata;
      e.getByLabel("source",rawdata);
      const FEDRawData& data = rawdata->FEDData(GTEVMId);
      size_t size=data.size();

      if (size>0 ) {
	  cout << "FED# " << setw(4) << GTEVMId << " " << setw(8) << size << " bytes " << endl;
	  if(evf::evtn::evm_board_sense(data.data(),size))
	    {
	      cout << "FED# " << setw(4) << GTEVMId << " is the real GT EVM block " << endl;
	      cout << "Event # " << evf::evtn::get(data.data(),true) << endl;
	      cout << "LS # " << evf::evtn::getlbn(data.data()) << endl;
	      cout << "ORBIT # " << evf::evtn::getorbit(data.data()) << endl;
	      cout << "GPS LOW # " << evf::evtn::getgpslow(data.data()) << endl;
	      cout << "GPS HI # " << evf::evtn::getgpshigh(data.data()) << endl;
	      cout << "BX FROM FDL 0-xing # " << evf::evtn::getfdlbx(data.data()) << endl;
	      cout << "PRESCALE INDEX FROM FDL 0-xing # " << evf::evtn::getfdlpsc(data.data()) << endl;
	    }
	  }

      const FEDRawData& data2 = rawdata->FEDData(GTPEId);
      size=data2.size();

      if (size>0 ) {
	  cout << "FED# " << setw(4) << GTPEId << " " << setw(8) << size << " bytes " << endl;
	  if(evf::evtn::gtpe_board_sense(data2.data()))
	    {
	      cout << "FED# " << setw(4) << GTPEId << " is the real GTPE block " << endl;
	      cout << "Event # " << evf::evtn::gtpe_get(data2.data()) << endl;
	      cout << "LS # " << evf::evtn::gtpe_getlbn(data2.data()) << endl;
	      cout << "ORBIT # " << evf::evtn::gtpe_getorbit(data2.data()) << endl;
	      cout << "BX # " << evf::evtn::gtpe_getbx(data2.data()) << endl;
	    }
	  }

      const FEDRawData& data3 = rawdata->FEDData(FEDNumbering::MINTCDSuTCAFEDID);
      size=data3.size();

      if (size>0 ) {
	evf::evtn::TCDSRecord record(data3.data());
	cout << "FED# " << setw(4) << FEDNumbering::MINTCDSuTCAFEDID << " " 
	     << setw(8) << size << " bytes " << endl;
	cout << "sizes: " 
	     << " BGOSize " << std::hex << (unsigned int) record.getHeader().getSizes().size.BGOSize  
	     << "  reserved2;" <<  std::hex <<(unsigned int)record.getHeader().getSizes().size.reserved2
	     << "  reserved1;" << std::hex <<(unsigned int) record.getHeader().getSizes().size.reserved1
	     << "  reserved0;" <<  std::hex <<(unsigned int)record.getHeader().getSizes().size.reserved0
	     << "  BSTSize;" <<  std::hex <<(unsigned int)record.getHeader().getSizes().size.BSTSize
	     << "  L1AhistSize;" <<  std::hex <<(unsigned int)record.getHeader().getSizes().size.L1AhistSize
	     << "  summarySize;" <<  std::hex <<(unsigned int)record.getHeader().getSizes().size.summarySize
	     << "  headerSize;" << std::hex << (unsigned int)record.getHeader().getSizes().size.headerSize
	     << std::endl;

	std::cout << "macAddress;          "
		  << hex << (uint64_t)record.getHeader().getData().header.macAddress;
	std::cout << "\n";
	std::cout << "sw;		   "
		  << hex << (unsigned int)record.getHeader().getData().header.sw;
	std::cout << "\n";		 
	std::cout << "fw;		   "
		  << hex <<(unsigned int)record.getHeader().getData().header.fw;
	std::cout << "\n";		 
	std::cout << "reserved0;	   "
		  << hex <<(unsigned int)record.getHeader().getData().header.reserved0;	
	std::cout << "\n"; 
	std::cout << "format;		   "
		  << hex <<(unsigned int)record.getHeader().getData().header.format;
	std::cout << "\n";		 
	std::cout << "runNumber;	   "
		  << dec << (unsigned int)record.getHeader().getData().header.runNumber;
	std::cout << "\n";	 
	std::cout << "reserved1;	   "
		  << hex <<(unsigned int)record.getHeader().getData().header.reserved1;	
	std::cout << "\n"; 
	std::cout << "activePartitions2;   "
		  << hex <<(unsigned int)record.getHeader().getData().header.activePartitions2; 
	std::cout << "\n";
	std::cout << "reserved2;	   "
		  << hex <<(unsigned int)record.getHeader().getData().header.reserved2;	 
	std::cout << "\n";
	std::cout << "activePartitions0;   "
		  << hex << (unsigned int)record.getHeader().getData().header.activePartitions0;
	std::cout << "\n"; 
	std::cout << "activePartitions1;   "
		  << hex <<(unsigned int)record.getHeader().getData().header.activePartitions1;
	std::cout << "\n"; 
	std::cout << "nibble;		   "
		  << dec << (unsigned int)record.getHeader().getData().header.nibble;
	std::cout << "\n";		 
	std::cout << "lumiSection;	   "
		  << dec << (unsigned int)record.getHeader().getData().header.lumiSection;
	std::cout << "\n";	 
	std::cout << "nibblesPerLumiSection;"
		  << hex <<(unsigned int)record.getHeader().getData().header.nibblesPerLumiSection;
	std::cout << "\n"; 
	std::cout << "triggerTypeFlags;	   "
		  << hex <<(unsigned int)record.getHeader().getData().header.triggerTypeFlags;
	std::cout << "\n";	 
	std::cout << "reserved5;	   "
		  << hex <<(unsigned int)record.getHeader().getData().header.reserved5;	
	std::cout << "\n"; 
	std::cout << "inputs;		   "
		  << hex <<(unsigned int)record.getHeader().getData().header.inputs;
	std::cout << "\n";		 
	std::cout << "bcid;		   "
		  << dec << (unsigned int)record.getHeader().getData().header.bcid;
	std::cout << "\n";		 
	std::cout << "orbitLow;		   "
		  << dec << (unsigned int)record.getHeader().getData().header.orbitLow;	
	std::cout << "\n";	 
	std::cout << "orbitHigh;	   "
		  << dec << (unsigned int)record.getHeader().getData().header.orbitHigh;
	std::cout << "\n";	 
	std::cout << "triggerCount;	   "
		  << dec << (uint64_t)record.getHeader().getData().header.triggerCount;
	std::cout << "\n";	 
	std::cout << "eventNumber;         "
		  << dec << (uint64_t)record.getHeader().getData().header.eventNumber;  
	std::cout << "\n";     
	std::cout << std::endl;

	std::cout << "====================l1a history===================" << std::endl;
	const evf::evtn::TCDSL1AHistory::l1a *history = record.getHistory().history().hist;
	for(unsigned int i = 0; i < 16; i++){
	  std::cout << i << " " << hex << history[i].bxid << std::endl;
	  std::cout << i << " " << hex << history[i].orbitlow << std::endl;
	  std::cout << i << " " << hex << history[i].orbithigh << std::endl;
	  std::cout << i << " " << hex << (unsigned int)history[i].eventtype << std::endl;
	}

	std::cout << " gpstimehigh; " << hex << record.getBST().getBST().gpstimehigh;
	std::cout << " gpstimelow; " << hex << record.getBST().getBST().gpstimelow;
	std::cout << " low0;   	" << record.getBST().getBST().low0;   
	std::cout << " high0;  	" << record.getBST().getBST().high0;  
	std::cout << " low1;   	" << record.getBST().getBST().low1;   
	std::cout << " high1;  	" << record.getBST().getBST().high1;  
	std::cout << " low2;   	" << record.getBST().getBST().low2;   
	std::cout << " high2;  	" << record.getBST().getBST().high2;  
	std::cout << " low3;   	" << record.getBST().getBST().low3;   
	std::cout << " high3;  	" << record.getBST().getBST().high3;  
	std::cout << " low4;   	" << record.getBST().getBST().low4;   
	std::cout << " high4;  	" << record.getBST().getBST().high4;  
	std::cout << " low5;   	" << record.getBST().getBST().low5;   
	std::cout << " status;  " << record.getBST().getBST().status; 
	std::cout << std::endl;

	//	char tmbuf[64], buf[64];
	time_t nowtime = (time_t)record.getBST().getBST().gpstimehigh;
	//	uint32_t nowusec = record.getBST().getBST().gpstimelow;
	std::cout << " value of nowtime " << hex << nowtime << dec <<std::endl; 
	std::cout << "GPS time " << ctime(&nowtime) << "."
		  << dec <<record.getBST().getBST().gpstimelow << std::endl;
      }


// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
    }
  };
DEFINE_FWK_MODULE(GlobalNumbersAnalysis);
}

