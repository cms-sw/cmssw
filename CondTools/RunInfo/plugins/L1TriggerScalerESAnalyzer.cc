#include <stdexcept>
#include <string>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/RunInfo/interface/L1TriggerScaler.h"

#include "CondFormats/DataRecord/interface/L1TriggerScalerRcd.h"

using namespace std;





namespace edmtest
{
  class L1TriggerScalerESAnalyzer : public edm::EDAnalyzer
  {
  public:
    explicit  L1TriggerScalerESAnalyzer(edm::ParameterSet const& p) 
    { 
      std::cout<<"L1TriggerScalerESAnalyzer"<<std::endl;
    }
    explicit  L1TriggerScalerESAnalyzer(int i) 
    { std::cout<<"L1TriggerScalerESAnalyzer "<<i<<std::endl; }
    virtual ~L1TriggerScalerESAnalyzer() {  
      std::cout<<"~L1TriggerScalerESAnalyzer "<<std::endl;
    }
     virtual void beginJob();
     virtual void beginRun(const edm::Run&, const edm::EventSetup& context);
    virtual void analyze(const edm::Event& e, const edm::EventSetup& c);
  private:
  };
   
  
 void
  L1TriggerScalerESAnalyzer::beginRun(const edm::Run&, const edm::EventSetup& context){
    std::cout<<"###L1TriggerScalerESAnalyzer::beginRun"<<std::endl;
    edm::ESHandle<L1TriggerScaler> L1TriggerScaler_lumiarray;
    std::cout<<"got eshandle"<<std::endl;
    context.get<L1TriggerScalerRcd>().get(L1TriggerScaler_lumiarray);
    std::cout<<"got data"<<std::endl;
  }
  
  void
  L1TriggerScalerESAnalyzer::beginJob(){
    std::cout<<"###L1TriggerScalerESAnalyzer::beginJob"<<std::endl;
   
  }
 
 
  void
   L1TriggerScalerESAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& context){
    using namespace edm::eventsetup;
    std::cout<<"###L1TriggerScalerESAnalyzer::analyze"<<std::endl;
     
    // Context is not used.
    std::cout <<" I AM IN RUN NUMBER "<<e.id().run() <<std::endl;
    std::cout <<" ---EVENT NUMBER "<<e.id().event() <<std::endl;
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("L1TriggerScalerRcd"));
    if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag()) {
      //record not found
      std::cout <<"Record \"L1TriggerScalerRcd"<<"\" does not exist "<<std::endl;
    }
    edm::ESHandle<L1TriggerScaler> l1tr;
    std::cout<<"got eshandle"<<std::endl;
    context.get<L1TriggerScalerRcd>().get(l1tr);
    std::cout<<"got context"<<std::endl;
    const L1TriggerScaler* l1lumiscaler=l1tr.product();
    std::cout<<"got L1TriggerScaler* "<< std::endl;

    /* let's user the printValues method
    std::cout<< "print result" << std::endl;
    l1lumiscaler->printAllValues();
    std::cout<< "print finished" << std::endl;
    */
std::cout<< "print  result" << std::endl;
    l1lumiscaler->printRunValue();
    l1lumiscaler->printLumiSegmentValues();
    l1lumiscaler->printFormat();
    l1lumiscaler->printGTAlgoCounts();
    l1lumiscaler->printGTAlgoRates();
    l1lumiscaler->printGTAlgoPrescaling();
    l1lumiscaler->printGTTechCounts();
    l1lumiscaler->printGTTechRates();
    l1lumiscaler->printGTTechPrescaling();
    l1lumiscaler->printGTPartition0TriggerCounts();
    l1lumiscaler->printGTPartition0TriggerRates();
    l1lumiscaler->printGTPartition0DeadTime();
    l1lumiscaler->printGTPartition0DeadTimeRatio();
    std::cout<<  "print  finished" << std::endl;
 


   /*
    for(std::vector<L1TriggerScaler::Lumi>::const_iterator it=l1lumiscaler->m_run.begin(); it!=l1lumiscaler->m_run.end(); ++it){
      std::cout << "  run:  " <<it->m_rn<<
	"\nlumisegment: "  << it->m_lumisegment<<std::endl;  



      for(size_t i=0; i<it->m_GTAlgoRates.size(); i++ ){ 
	std::cout << "m_GTAlgoRates["<<i<<"] = "<< it->m_GTAlgoRates[i]<<std::endl;  
      }
     for(size_t i=0; i<it->m_GTAlgoPrescaling.size(); i++ ){ 
       std::cout << "m_GTAlgoPrescaling["<<i<<"] = "<< it->m_GTAlgoPrescaling[i]<<std::endl;  
      } 
for(size_t i=0; i<it->m_GTTechCounts.size(); i++ ){ 
       std::cout << " m_GTTechCounts["<<i<<"] = "<< it->m_GTTechCounts[i]<<std::endl;  
      } 

for(size_t i=0; i<it->m_GTTechRates.size(); i++ ){ 
       std::cout << " m_GTTechRates["<<i<<"] = "<< it->m_GTTechRates[i]<<std::endl;  
      } 
for(size_t i=0; i<it->m_GTTechPrescaling.size(); i++ ){ 
       std::cout << " m_GTTechPrescaling["<<i<<"] = "<< it->m_GTTechPrescaling[i]<<std::endl;  
      } 
for(size_t i=0; i<it->m_GTPartition0TriggerCounts.size(); i++ ){ 
       std::cout << " m_GTPartition0TriggerCounts["<<i<<"] = "<< it->m_GTPartition0TriggerCounts[i]<<std::endl;  
      } 
for(size_t i=0; i<it->m_GTPartition0TriggerRates.size(); i++ ){ 
       std::cout << " m_GTPartition0TriggerRates["<<i<<"] = "<< it->m_GTPartition0TriggerRates[i]<<std::endl;  
      } 

for(size_t i=0; i<it->m_GTPartition0DeadTime.size(); i++ ){ 
       std::cout << " m_GTPartition0DeadTime["<<i<<"] = "<< it->m_GTPartition0DeadTime[i]<<std::endl;  
      }
for(size_t i=0; i<it->m_GTPartition0DeadTimeRatio.size(); i++ ){ 
       std::cout << " m_GTPartition0DeadTimeRatio["<<i<<"] = "<< it->m_GTPartition0DeadTimeRatio[i]<<std::endl;  
      }
   }
    */
  }
  DEFINE_FWK_MODULE(L1TriggerScalerESAnalyzer);
}


