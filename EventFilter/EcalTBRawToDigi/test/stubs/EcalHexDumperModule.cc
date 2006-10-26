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
#include <string>

#include <stdio.h>
#include <fstream>

#include <iomanip> 

using namespace cms;
using namespace std;


class EcalHexDumperModule: public edm::EDAnalyzer{
  
 public:

  EcalHexDumperModule(const edm::ParameterSet& ps){  
    verbosity_= ps.getUntrackedParameter<int>("verbosity",1);
    
    beg_DCC_fed_id_= ps.getUntrackedParameter<int>("beg_DCC_fed_id",600);
    end_DCC_fed_id_= ps.getUntrackedParameter<int>("end_DCC_fed_id",670);

    writeDcc_ =ps.getUntrackedParameter<bool>("writeDCC",false);
    filename_  =ps.getUntrackedParameter<string>("filename","dump.bin");
    event_ =0;
  }

  
 protected:
  int       verbosity_;
  bool   writeDcc_;
  int      beg_DCC_fed_id_;
  int      end_DCC_fed_id_;
  string filename_;
  int      event_;

  void analyze( const edm::Event & e, const  edm::EventSetup& c);

};



void EcalHexDumperModule::analyze( const edm::Event & e, const  edm::EventSetup& c){
  
  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByType(rawdata);  

  ofstream dumpFile (filename_.c_str(),ios::app );
  
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
      const ulong               * pData     = ( reinterpret_cast<ulong*>(const_cast<unsigned char*> ( data.data())));
      cout << setfill('0');
      for (int words=0; words < length/4; (words+=2)  )
	{
	  cout << setw(8)   << hex << pData[words+1] << " ";
	  cout << setw(8)   << hex << pData[words] << endl;
	}

      cout << "\n";


      if (beg_DCC_fed_id_ <= id && id <= end_DCC_fed_id_ && writeDcc_)
	{
	  dumpFile.write( reinterpret_cast <const char *> (pData), length);
	}
    }
    
  }
  dumpFile.close();    
  if (! writeDcc_) remove(filename_.c_str()); 
}


DEFINE_FWK_MODULE(EcalHexDumperModule);
