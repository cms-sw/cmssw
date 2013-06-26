/**
 *   
 * \author G. Franzoni
 *
 */

#include "CaloOnlineTools/EcalTools/plugins/EcalHexDisplay.h"

EcalHexDisplay::EcalHexDisplay(const edm::ParameterSet& ps) :
    verbosity_ (ps.getUntrackedParameter<int>("verbosity",1)),
    beg_fed_id_ (ps.getUntrackedParameter<int>("beg_fed_id",0)),
    end_fed_id_ (ps.getUntrackedParameter<int>("end_fed_id",654)),
    first_event_ (ps.getUntrackedParameter<int>("first_event",1)),
    last_event_  (ps.getUntrackedParameter<int>("last_event",9999999)),
    event_ (0),
    writeDcc_ (ps.getUntrackedParameter<bool>("writeDCC",false)),
    filename_ (ps.getUntrackedParameter<std::string>("filename","dump.bin")),
    fedRawDataCollectionTag_(ps.getParameter<edm::InputTag>("fedRawDataCollectionTag"))
{  
}

void EcalHexDisplay::analyze( const edm::Event & e, const  edm::EventSetup& c){
  
  event_++;
  if (event_ < first_event_ || last_event_ < event_) return;
  

  edm::Handle<FEDRawDataCollection> rawdata;
  e.getByLabel(fedRawDataCollectionTag_, rawdata);  

  std::ofstream dumpFile (filename_.c_str(),std::ios::app );
  
  for (int id= 0; id<=FEDNumbering::MAXFEDID; ++id){ 
    
    if (id < beg_fed_id_ || end_fed_id_ < id) continue;

    const FEDRawData& data = rawdata->FEDData(id);
    
    if (data.size()>4){      
      
      std::cout << "\n\n\n[EcalHexDumperModule] Event: " 
		<< std::dec << event_ 
		<< " fed_id: " << id 
		<< " size_fed: " << data.size() << "\n"<< std::endl;
      
      if ( ( data.size() %16 ) !=0)
	{
	  std::cout << "***********************************************" << std::endl;
	  std::cout<< "Fed size in bits not multiple of 64, strange." << std::endl;
	  std::cout << "***********************************************" << std::endl;
	}
      
      
      int length = data.size();
      const unsigned long               * pData     = ( reinterpret_cast<unsigned long*>(const_cast<unsigned char*> ( data.data())));
      std::cout << std::setfill('0');
      for (int words=0; words < length/4; (words+=2)  )
	{
	  std::cout << std::setw(8)   << std::hex << pData[words+1] << " ";
	  std::cout << std::setw(8)   << std::hex << pData[words] << std::endl;
	}

      std::cout << "\n";


      if (beg_fed_id_ <= id && id <= end_fed_id_ && writeDcc_)
	{
	  dumpFile.write( reinterpret_cast <const char *> (pData), length);
	}
    }
    
  }
  dumpFile.close();    
  if (! writeDcc_) remove(filename_.c_str()); 
}

