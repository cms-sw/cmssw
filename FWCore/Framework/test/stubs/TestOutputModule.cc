
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"

#include <string>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <iterator>

using namespace std;
using namespace edm;

namespace {

 void printBits(unsigned char c){

         //cout << "HEX: "<< "0123456789ABCDEF"[((c >> 4) & 0xF)] <<endl;

        for (int i = 7; i >= 0; --i) {
            int bit = ((c >> i) & 1);
            cout << " "<<bit;
        }
 }

 void packIntoString(vector<unsigned char> const& source,
                    vector<unsigned char>& package)
 {
 unsigned int packInOneByte = 4;
 unsigned int sizeOfPackage = 1 + 
           ((source.size()-1)/packInOneByte); //Two bits per HLT
 if (source.size() == 0) sizeOfPackage = 0;
 
 package.resize(sizeOfPackage);
 memset(&package[0], 0x00, sizeOfPackage);
    
 for (unsigned int i=0; i != source.size() ; ++i)
   { 
      unsigned int whichByte = i/packInOneByte;
      unsigned int indxWithinByte = i % packInOneByte;
      package[whichByte] = package[whichByte] | 
                            (source[i] << (indxWithinByte*2));
   }
  //for (unsigned int i=0; i !=package.size() ; ++i)
  //   printBits(package[i]);
   cout<<endl;

 }

}
namespace edmtest
{

  class TestOutputModule : public edm::OutputModule
  {
  public:
    explicit TestOutputModule(edm::ParameterSet const&);
    virtual ~TestOutputModule();

  private:
    virtual void write(edm::EventPrincipal const& e);
    virtual void endLuminosityBlock(edm::LuminosityBlockPrincipal const&){}
    virtual void endRun(edm::RunPrincipal const&){}
    virtual void endJob();

    string name_;
    int bitMask_;
    std::vector<unsigned char> hltbits_;
  };

  // -----------------------------------------------------------------

  TestOutputModule::TestOutputModule(edm::ParameterSet const& ps):
    edm::OutputModule(ps),
    name_(ps.getParameter<string>("name")),
    bitMask_(ps.getParameter<int>("bitMask")),
    hltbits_(0)
  {
  }
    
  TestOutputModule::~TestOutputModule()
  {
  }

  void TestOutputModule::write(edm::EventPrincipal const& e)
  {
    assert(currentContext() != 0);

    Trig prod;

    prod = getTriggerResults(e);

    vector<unsigned char> vHltState;

    std::vector<std::string> hlts = getAllTriggerNames();
    unsigned int hltSize = hlts.size(); 

    for(unsigned int i=0; i != hltSize ; ++i) {
      vHltState.push_back(((prod->at(i)).state()));
    }

    //Pack into member hltbits_
    packIntoString(vHltState, hltbits_);

    //This is Just a printing code.
    cout <<"Size of hltbits:"<<hltbits_.size()<<endl;

    char* intp = (char*)&bitMask_;
    bool matched = false;

    for(int i = hltbits_.size() - 1; i != -1 ; --i) {
      cout<<endl<<"Current Bits Mask byte:";printBits(hltbits_[i]);
      unsigned char tmp = static_cast<unsigned char>(*(intp+i));
      cout<<endl<<"Original Byte:";printBits(tmp);cout<<endl;

      if (tmp == hltbits_[i]) matched = true;
    }
    cout<<"\n";

    if ( !matched && hltSize > 0)
    {
       cerr << "\ncfg bitMask is different from event..aborting."<<endl;

       abort();
    }
    else cout <<"\nSUCCESS: Found Matching Bits"<<endl;
  }

  void TestOutputModule::endJob()
  {
    assert( currentContext() == 0 );
  }
}

using edmtest::TestOutputModule;

DEFINE_FWK_MODULE(TestOutputModule);
