/**
 * module  for displaying unpacked DCCHeader information
 *   
 * \author A. Ghezzi
 * \author S. Cooper
 * \author G. Franzoni
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <iostream>
#include <vector>



class EcalDCCHeaderDisplay: public edm::EDAnalyzer{
  
public:
  EcalDCCHeaderDisplay(const edm::ParameterSet& ps);
  
protected:

  void analyze( const edm::Event & e, const  edm::EventSetup& c);

private:
  edm::InputTag EcalDCCHeaderCollection_;

};

EcalDCCHeaderDisplay::EcalDCCHeaderDisplay(const edm::ParameterSet& iConfig){
  EcalDCCHeaderCollection_ = iConfig.getParameter<edm::InputTag>("EcalDCCHeaderCollection");
}


void EcalDCCHeaderDisplay::analyze( const edm::Event & e, const  edm::EventSetup& c){
    
  edm::Handle<EcalRawDataCollection> DCCHeaders;
  e.getByLabel(EcalDCCHeaderCollection_, DCCHeaders);
    

  std::cout << "\n\n ^^^^^^^^^^^^^^^^^^ [EcalDCCHeaderDisplay]  DCCHeaders collection size " << DCCHeaders->size() << std::endl;
  std::cout << "          [EcalDCCHeaderDisplay]  the Header(s)\n"  << std::endl;
  //short dumpConter =0;      

  for ( EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();headerItr != DCCHeaders->end(); 
	++headerItr ) {
    //      int nevt =headerItr->getLV1(); 
    bool skip = false;

    if(skip){continue;}
    std::cout<<"###################################################################### \n";
    std::cout << "FedId: "<< headerItr->fedId() <<"\n";
      
    std::cout << "DCCErrors: "<<headerItr->getDCCErrors()<<"\n";
    std::cout<<"Run Number: "<<headerItr->getRunNumber()<<"\n";
    std::cout<<"Event number (LV1): "<<headerItr->getLV1()<<"\n";
    std::cout<<"BX: "<<headerItr->getBX()<<"\n";
    std::cout<<"TRIGGER TYPE: "<< headerItr->getBasicTriggerType()<<"\n";
      
    std::cout<<"RUNTYPE: "<< headerItr->getRunType()<<"\n";
    std::cout<<"Half: "<<headerItr->getRtHalf()<<"\n";
    std::cout<<"MGPA gain: "<<headerItr->getMgpaGain()<<"\n";
    std::cout<<"MEM gain: "<<headerItr->getMemGain()<<"\n";
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
    std::cout<<"LaserPower: "<<  settings.LaserPower<<"\n";
    std::cout <<"LAserFilter: "<<settings.LaserFilter<<"\n";
    std::cout<<"Wavelenght: "<<settings.wavelength<<"\n";
    std::cout<<"delay: "<<settings.delay<<"\n";
    std::cout<<"MEM Vinj: "<< settings.MEMVinj<<"\n";
    std::cout<<"MGPA content: "<<settings.mgpa_content<<"\n";
    std::cout<<"Ped offset dac: "<<settings.ped_offset<<"\n";

    std::cout<<"Selective Readout: "<<headerItr->getSelectiveReadout()<<"\n";
    std::cout<<"ZS: "<<headerItr->getZeroSuppression()<<"\n";
    std::cout <<"TZS: "<<headerItr->getTestZeroSuppression()<<"\n";
    std::cout<<"SRStatus: "<<headerItr->getSrpStatus()<<"\n";

    std::vector<short> TCCStatus = headerItr->getTccStatus();
    std::cout<<"TCC Status size: "<<TCCStatus.size()<<std::endl;
    std::cout<<"TCC Status: ";
    for(unsigned u =0;u<TCCStatus.size();u++){
      std::cout<<TCCStatus[u]<<" ";
    }
    std::cout<<std::endl;
      
    std::vector<short> TTStatus = headerItr->getFEStatus();
    std::cout<<"TT Status size: "<<TTStatus.size()<<std::endl;
    std::cout<<"TT Statuses: ";
    for(unsigned u =0;u<TTStatus.size();u++){
      if ( !(u%14) ) std::cout<<std::endl; // TODO: add space after first six in a row
      std::cout<<TTStatus[u]<<" ";
    }
    std::cout<<std::endl;
    std::cout<<"######################################################################"<<std::endl;;

  }	
     
} 



DEFINE_FWK_MODULE(EcalDCCHeaderDisplay);
