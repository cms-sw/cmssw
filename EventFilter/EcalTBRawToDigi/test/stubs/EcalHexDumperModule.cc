/**
 *   
 * \author G. Franzoni
 *
 */


#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Handle.h>
#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <DataFormats/EcalDigi/interface/EcalDigiCollections.h>

#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDNumbering.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>

#include <iostream>
#include <vector>

#include <iomanip> 

using namespace cms;
using namespace std;


class EcalHexDumperModule: public edm::EDAnalyzer{
  
 public:

  EcalHexDumperModule(const edm::ParameterSet& ps){  
    verbosity= ps.getUntrackedParameter<int>("verbosity",1);
    event_ =0;
  }

  
 protected:
  int verbosity;
  int event_;

  void analyze( const edm::Event & e, const  edm::EventSetup& c);


};// class EcalHexDumperModule



void EcalHexDumperModule::analyze( const edm::Event & e, const  edm::EventSetup& c){
  
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);  
  
  event_++;

  for (int id= 0; id<=FEDNumbering::lastFEDId(); ++id){ 

    const FEDRawData& data = rawdata->FEDData(id);

    if (data.size()>4){      
 
      cout << "\n\n\n[EcalHexDumperModule] Event: " 
	   << dec << event_ 
	   << " fed: " << id 
	   << " size_fed: " << data.size() << "\n"<< endl;
      
      if ( ( data.size() %16 ) !=0)
	{
	  cout << "***********************************************" << endl;
	  cout<< "Fed size in bits not multiple of 64, strange." << endl;
	  cout << "***********************************************" << endl;
	}


      int length = data.size();
      const ulong * pData = ( reinterpret_cast<ulong*>(const_cast<unsigned char*> ( data.data())));


      cout << setfill('0');
      for (int words=0; words < length/8; (words+=2)  )
	{
	  cout << setw(8)   << hex << pData[words+1] << " ";
	  cout << setw(8)   << hex << pData[words] << endl;
	}

      cout << "\n";
            
    }

  }


}



DEFINE_FWK_MODULE(EcalHexDumperModule)
