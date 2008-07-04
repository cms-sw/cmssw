/** \file
 * 
 * 
 * $Date: 2006/10/27 01:35:23 $
 * $Revision: 1.6 $
 * \author N. Amapane - S. Argiro'
 *
*/

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>

#include "EventFilter/Utilities/interface/GlobalEventNumber.h"

#include <iostream>
#include <iomanip>

using namespace edm;
using namespace std;

namespace test{

  static const unsigned int GTEVMId= 812;
  class GlobalNumbersAnalysis: public EDAnalyzer{
  private:
  public:
    GlobalNumbersAnalysis(const ParameterSet& pset){
    }

 
    void analyze(const Event & e, const EventSetup& c){
      cout << "--- Run: " << e.id().run()
	   << " Event: " << e.id().event() << endl;
      Handle<FEDRawDataCollection> rawdata;
      e.getByType(rawdata);
      const FEDRawData& data = rawdata->FEDData(GTEVMId);
      size_t size=data.size();

      if (size>0 ) {
	  cout << "FED# " << setw(4) << GTEVMId << " " << setw(8) << size << " bytes " << endl;
	  if(evf::evtn::evm_board_sense(data.data()))
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

// 	  CPPUNIT_ASSERT(trailer.check()==true);
// 	  CPPUNIT_ASSERT(trailer.lenght()==(int)data.size()/8);
    }
  };
DEFINE_FWK_MODULE(GlobalNumbersAnalysis);
}

