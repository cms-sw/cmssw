/**
 * \file EcalDigiDumperModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2009/09/03 22:47:50 $
 * $Revision: 1.11 $
 *
 * \author A. Ghezzi
 *
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalRawData/interface/EcalRawDataCollections.h>
#include <iostream>
#include <vector>



class EcalDCCHeaderDumperModule: public edm::EDAnalyzer{
  
 public:
  EcalDCCHeaderDumperModule(const edm::ParameterSet& ps){   }
  
 protected:
  
  void analyze( const edm::Event & e, const  edm::EventSetup& c){
    
    edm::Handle<EcalRawDataCollection> DCCHeaders;
    e.getByLabel("ecalEBunpacker", DCCHeaders);
    

    std::cout << "\n\n ^^^^^^^^^^^^^^^^^^ [EcalDCCHeaderDumperModule]  DCCHeaders collection size " << DCCHeaders->size() << std::endl;
    std::cout << "          [EcalDCCHeaderDumperModule]  the Header(s)\n"  << std::endl;
    //short dumpConter =0;      

    for ( EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();headerItr != DCCHeaders->end(); 
	  ++headerItr ) {
      //      int nevt =headerItr->getLV1(); 
      bool skip = false;
      //LASER
//       bool skip = nevt > 3 && nevt < 620;
//       skip = skip || (nevt > 623 && nevt <1230);
//       skip = skip || (nevt > 1233 && nevt <1810);
//       skip = skip || (nevt > 1813);

 //    MGPA test pulse
 //      bool skip = nevt > 3 && nevt <53;
//       skip = skip || (nevt > 56 && nevt <102);
//       skip = skip || (nevt > 105);

 //    PEDESTAL
 //    bool skip = nevt > 3 && nevt <153;
//     skip = skip || (nevt > 156 && nevt <302);
//     skip = skip || (nevt > 305);


    if(skip){continue;}
      // if(nevt > 60 ){break;}
      std::cout<<"###################################################################### \n";
      std::cout << "DCCid: "<< headerItr->id()<<"\n";
      
      std::cout << "DCCErrors: "<<headerItr->getDCCErrors()<<"\n";
      std::cout<<"Run Number: "<<headerItr->getRunNumber()<<"\n";
      std::cout<<"Event number (LV1): "<<headerItr->getLV1()<<"\n";
      // this requires DataFormats/EcalRawData V01-01-12
      //std::cout << "Orbit: " << headerItr->getOrbit () << "\n";
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
      
      // this requires DataFormats/EcalRawData V01-01-12
      //std::vector<short> TTStatus = headerItr->getFEStatus();
      std::vector<short> TTStatus = headerItr->getFEStatus();
      std::cout<<"TT Status size: "<<TTStatus.size()<<std::endl;
      std::cout<<"TT Status: ";
      for(unsigned u =0;u<TTStatus.size();u++){
	std::cout<<TTStatus[u]<<" ";
      }
      std::cout<<std::endl;
      std::cout<<"######################################################################"<<std::endl;;
      //if( (dumpConter++) > 10) break;

    }	
    
    
    
  } // produce

};// class EcalDigiDumperModule

DEFINE_FWK_MODULE(EcalDCCHeaderDumperModule);
