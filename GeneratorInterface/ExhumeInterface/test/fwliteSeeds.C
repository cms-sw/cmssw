#include <string>
#include <iostream>

#if !defined(__CINT__) && !defined(__MAKECINT__)
//Headers for the data items
#include "DataFormats/FWLite/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "SimDataFormats/RandomEngine/interface/RandomEngineState.h"
#endif

template<typename T>
class PrintToStdOut{
   public: 
      PrintToStdOut(const std::string& prefix = ""): prefix_(prefix) {}
      void operator()(const T& item){std::cout << prefix_ << item << std::endl;}
   private:
      std::string prefix_;
};

void printSeeds(const std::vector<std::string>& fileNames, int maxEvents = -1, const char* processName = "HLT"){

   std::cout << ">>> Reading files: " << std::endl;
   for(std::vector<std::string>::const_iterator it = fileNames.begin(); it != fileNames.end(); ++it) std::cout << "  " << *it << std::endl; 
   // Chain the input files
   fwlite::ChainEvent ev(fileNames);

   // Loop over the events
   int nEvts = 0;
   std::vector<std::string> labelNames;
   labelNames.push_back("@source");
   labelNames.push_back("generator");
   for( ev.toBegin(); ! ev.atEnd(); ++ev) {

     if((maxEvents > 0)&&(nEvts == maxEvents)) break;

     ++nEvts;
     std::cout << ">>> Event number: " << nEvts << endl;

     fwlite::Handle<std::vector<RandomEngineState> > rndmEngineStateCollection;
     rndmEngineStateCollection.getByLabel(ev,"randomEngineStateProducer","",processName);

     const std::vector<RandomEngineState>& randomEngineStates = *rndmEngineStateCollection;

     std::vector<RandomEngineState>::const_iterator it_rndm = randomEngineStates.begin();
     std::vector<RandomEngineState>::const_iterator it_rndm_end = randomEngineStates.end();
     for(; it_rndm != it_rndm_end; ++it_rndm){
        if(std::find(labelNames.begin(),labelNames.end(),it_rndm->getLabel()) == labelNames.end()) continue;

        //std::cout << "   label " << it_rndm->getLabel() << std::endl;
        //const std::vector<unsigned int>& states = it_rndm->getState();
        const std::vector<unsigned int>& seeds = it_rndm->getSeed();
        //std::for_each(states.begin(),states.end(),PrintToStdOut<unsigned int>());
        std::for_each(seeds.begin(),seeds.end(),PrintToStdOut<unsigned int>(it_rndm->getLabel() + "  "));
     }
   }
}
