#include "OnlineDB/CSCCondDB/interface/CSCCableReadTest.h"
#include "string"

CSCCableReadTest::CSCCableReadTest(const edm::ParameterSet&){}
CSCCableReadTest::~CSCCableReadTest(){}

void CSCCableReadTest::analyze (const edm::Event&, const edm::EventSetup&)
{

  int i;

  i=system("date");
  csccableread *cable = new csccableread ();
  std::cout << " Connected cscr for cables ... " << std::endl;

  // Get information by chamber index.
   int chamberindex = 480;
   std::cout<<std::endl;
   std::cout<<std::endl;
   std::cout<<"Method cable_read, input: chamber index  "<<chamberindex<<std::endl;
   std::cout<<std::endl;
   std::string chamber_label, cfeb_rev, alct_rev;
   int cfeb_length, alct_length, cfeb_tmb_skew_delay, cfeb_timing_corr;
   cable->cable_read(chamberindex, &chamber_label, &cfeb_length, &cfeb_rev,
    &alct_length, &alct_rev, &cfeb_tmb_skew_delay, &cfeb_timing_corr);

     std::cout<<"chamber_label  "<<"  "<<chamber_label <<std::endl;
     std::cout<<"cfeb_length  "<<"  "<<cfeb_length <<std::endl;
     std::cout<<"cfeb_rev  "<<"  "<<cfeb_rev <<std::endl;
     std::cout<<"alct_length  "<<"  "<<alct_length <<std::endl;
     std::cout<<"alct_rev  "<<"  "<<alct_rev <<std::endl;
     std::cout<<"cfeb_tmb_skew_delay  "<<"  "<<cfeb_tmb_skew_delay <<std::endl;
     std::cout<<"cfeb_timing_corr  "<<"  "<<cfeb_timing_corr <<std::endl;
}
void CSCCableReadTest::beginJob(){
  std::cout << "Here is the start" << std::endl;
  std::cout << "-----------------" << std::endl;
}
void CSCCableReadTest::endJob() {
  std::cout << "---------------" << std::endl;
  std::cout << "Here is the end" << std::endl;
}
