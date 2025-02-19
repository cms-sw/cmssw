/** \file
 * 
 * 
 * $Date: 2010/04/26 16:10:23 $
 * $Revision: 1.5 $
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

#include <iostream>
#include <iomanip>

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

// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
    }
  };
DEFINE_FWK_MODULE(GlobalNumbersAnalysis);
}

