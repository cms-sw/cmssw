/**
 * \file EcalDigiDumperModule.h 
 * dummy module  for the test of  DaqFileInputService
 *   
 * 
 * $Date: 2006/05/05 09:09:31 $
 * $Revision: 1.4 $
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


using namespace cms;
using namespace std;


class EcalDCCHeaderDumperModule: public edm::EDAnalyzer{
  
 public:
  EcalDCCHeaderDumperModule(const edm::ParameterSet& ps){   }
  
 protected:
  
  void analyze( const edm::Event & e, const  edm::EventSetup& c){
    
    edm::Handle<EcalRawDataCollection> DCCHeaders;
    e.getByLabel("ecalEBunpacker", DCCHeaders);
    

    cout << "\n\n ^^^^^^^^^^^^^^^^^^ [EcalDCCHeaderDumperModule]  DCCHeaders collection size " << DCCHeaders->size() << endl;
    cout << "          [EcalDCCHeaderDumperModule]  the Header(s)\n"  << endl;
    //short dumpConter =0;      

    for ( EcalRawDataCollection::const_iterator headerItr= DCCHeaders->begin();headerItr != DCCHeaders->end(); 
	  ++headerItr ) {
      int nevt =headerItr->getLV1(); 
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
      cout<<"###################################################################### \n";
      cout << "DCCid: "<< headerItr->id()<<"\n";
      
      cout << "DCCErrors: "<<headerItr->getDCCErrors()<<"\n";
      cout<<"Run Number: "<<headerItr->getRunNumber()<<"\n";
      cout<<"Event number (LV1): "<<headerItr->getLV1()<<"\n";
      cout<<"BX: "<<headerItr->getBX()<<"\n";
      cout<<"TRIGGER TYPE: "<< headerItr->getBasicTriggerType()<<"\n";
      
      cout<<"RUNTYPE: "<< headerItr->getRunType()<<"\n";
      cout<<"Half: "<<headerItr->getRtHalf()<<"\n";
      cout<<"MGPA gain: "<<headerItr->getMgpaGain()<<"\n";
      cout<<"MEM gain: "<<headerItr->getMemGain()<<"\n";
      EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
      cout<<"LaserPower: "<<  settings.LaserPower<<"\n";
      cout <<"LAserFilter: "<<settings.LaserFilter<<"\n";
      cout<<"Wavelenght: "<<settings.wavelength<<"\n";
      cout<<"delay: "<<settings.delay<<"\n";
      cout<<"MEM Vinj: "<< settings.MEMVinj<<"\n";
      cout<<"MGPA content: "<<settings.mgpa_content<<"\n";
      cout<<"Ped offset dac: "<<settings.ped_offset<<"\n";

      cout<<"Selective Readout: "<<headerItr->getSelectiveReadout()<<"\n";
      cout<<"ZS: "<<headerItr->getZeroSuppression()<<"\n";
      cout <<"TZS: "<<headerItr->getTestZeroSuppression()<<"\n";
      cout<<"SRStatus: "<<headerItr->getSrpStatus()<<"\n";

      std::vector<short> TCCStatus = headerItr->getTccStatus();
      cout<<"TCC Status size: "<<TCCStatus.size()<<endl;
      cout<<"TCC Status: ";
      for(unsigned u =0;u<TCCStatus.size();u++){
	cout<<TCCStatus[u]<<" ";
      }
      cout<<endl;
      
      std::vector<short> TTStatus = headerItr->getTriggerTowerStatus();
      cout<<"TT Status size: "<<TTStatus.size()<<endl;
      cout<<"TT Status: ";
      for(unsigned u =0;u<TTStatus.size();u++){
	cout<<TTStatus[u]<<" ";
      }
      cout<<endl;
      cout<<"######################################################################"<<endl;;
      //if( (dumpConter++) > 10) break;

    }	
    
    
    
  } // produce

};// class EcalDigiDumperModule

DEFINE_FWK_MODULE(EcalDCCHeaderDumperModule);
  


