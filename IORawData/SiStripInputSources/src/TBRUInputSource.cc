#include "TFile.h"
#include "TTree.h"

#include "IORawData/SiStripInputSources/interface/TBRUInputSource.h"
#include "TBRU.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "PluginManager/PluginCapabilities.h"

#include "interface/evb/include/i2oEVBMsgs.h"
#include "interface/shared/include/i2oXFunctionCodes.h"

#include <iostream>

ClassImp(TBRU);


using namespace edm;
using namespace std;

TBRUInputSource::TBRUInputSource(const edm::ParameterSet & pset, edm::InputSourceDescription const& desc) : 
  edm::ExternalInputSource(pset,desc),
  m_quiet( pset.getUntrackedParameter<bool>("quiet",true))
{
  m_tree=0;
  m_fileCounter=-1;
  m_file=0;
  m_i=0;
  n_fed9ubufs=0;
  for (int i=0;i<MAX_FED9U_BUFFER;i++)
    m_fed9ubufs[i]=0;

  produces<FEDRawDataCollection>();
}



void TBRUInputSource::openFile(const std::string& filename) {
  if (m_file!=0) {
    m_file->Close();
    m_file=0;
    m_tree=0;
  }
  
  //  try {
  m_file=TFile::Open(filename.c_str());
  if (m_file==0) {
    cout << "Unable to open " << filename << endl;
    m_tree=0;
    return;
  } 
  
  // Get the run number

  int ipass = filename.find("RU");
  int ipath = filename.find("_");
  string run;
  run.clear();
  run=filename.substr(ipass+2,ipath-ipass-2);
  cout << run << " et " << run.c_str() <<endl;
  sscanf(run.c_str(),"%d",&n_run);
  printf("%d\n",n_run);
  //

  m_tree=(TTree*)m_file->Get("TRU");
  
  if (m_tree==0) {
    m_file->Close();
    m_file=0;
    cout << "Unable to find TBRU tree" << endl;
    return;
  }
  
  if (!m_quiet) {
    cout << "Opening '" << filename << "' with " << m_tree->GetEntries() << " events.\n";
  }

  TObjArray* lb=m_tree->GetListOfBranches();
  n_fed9ubufs=0;
  for (int i=0; i<lb->GetSize(); i++) {
    TBranch* b=(TBranch*)lb->At(i);
    if (b==0) continue;

    char sizename[256];
    char arrayname[256];
    sprintf(sizename,"size_RU_%x",i/2);

      
    sprintf(arrayname,"RU_%x",i/2);
      

    cout <<b->GetName()<< endl;
    cout <<sizename<<" or " <<arrayname <<endl;
    if (!strcmp(b->GetName(),sizename)) {
      if (m_fed9ubufs[n_fed9ubufs] ==  0)
      {
	m_fed9ubufs[n_fed9ubufs]= new TBRU(i/2);
	cout << "Creating TBRU " << n_fed9ubufs<< " Instance" << i/2 << endl;
	cout << hex<<m_fed9ubufs[n_fed9ubufs]<< dec <<endl;
	}
       b->SetAddress(&(m_fed9ubufs[n_fed9ubufs]->fSize));

    } else {
      if (strcmp(b->GetName(),arrayname)) continue;
      b->SetAddress(&(m_fed9ubufs[n_fed9ubufs]->fBuffer));
      n_fed9ubufs++;
    }
  }
  m_i=0;
  cout << "File " << filename << " is opened" <<endl;
}

void TBRUInputSource::setRunAndEventInfo() {
  bool is_new=false;

  while (m_tree==0 || m_i==m_tree->GetEntries()) {
    m_fileCounter++;
    if (m_file!=0) {
       m_file->Close();
       m_file=0;
       m_tree=0;
    }
    if (m_fileCounter>=int(fileNames().size())) return; // nothing good
    openFile(fileNames()[m_fileCounter]);
    is_new=true;
  }

  if (m_tree==0 || m_i==m_tree->GetEntries()) return; //nothing good

  m_tree->GetEntry(m_i);
  if (!m_quiet)
  	cout <<"Event " <<m_i<< "is read " << endl;
  m_i++;
  TBRU* r= m_fed9ubufs[0];
  cout << "TBRU 0 is "<<hex <<r<<dec <<endl;
  int* rud =r->fBuffer;

  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block = ( I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) rud;

cout << "Event number " << block->eventNumber << endl;
    setRunNumber(n_run);
    setEventNumber( block->eventNumber);

  // time is a hack
  edm::TimeValue_t present_time = presentTime();
  unsigned long time_between_events = timeBetweenEvents();

  setTime(present_time + time_between_events);
  if (!m_quiet)
  	cout <<"Event & run end" << endl;
}

bool TBRUInputSource::produce(edm::Event& e) {

  if (m_tree==0) return false;
 
  std::auto_ptr<FEDRawDataCollection> bare_product(new  FEDRawDataCollection());
  for (int i=0; i<n_fed9ubufs; i++) 
    {
	if (!m_quiet)
	std::cout << "Reading bytes for FED " << i << std::endl;
      I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block = ( I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) m_fed9ubufs[i]->fBuffer;
      size_t msgHeaderSize       = sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
      
      // Fill in payload check markers for debugging
      unsigned char* cbuf        = (unsigned char*) block + msgHeaderSize;
      int* ibuf = (int*) cbuf;
      int len = (ibuf[0]-8);
      int id=0;
      if (i == 0)
	id =1023;
      else
	{
	  unsigned char* _buffer= (unsigned char*) &ibuf[1];
	  id = ((_buffer[5]<<8) | _buffer[6]) ;
	  // 	int* rudat = &ibuf[1];
	  // 	int buf_len= ibuf[0]/sizeof(int)-2;
	  // 	try
	  // 	  {
	  // 	    fedEvent.Init((u32*) rudat,0,(u32)buf_len);
	  // 	    fedEvent.checkEvent();

	  // 	  } catch (ICUtils::ICException &e)
	  // 	    {
	  // 	      std::cout<<"Fed ID Might be "<<fedEvent.getSourceId()<<std::endl;
	  // 	      continue;

	  // 	    }

	}

      const unsigned char* data=(const unsigned char*) &ibuf[1];
      FEDRawData& fed=bare_product->FEDData(id);
      int fulllen =(len%8)?len+8-len%8:len;
      fed.resize(fulllen);
      memcpy(fed.data(),data,len);

      if (!m_quiet)
	std::cout << "Reading " << len << " bytes for FED " << id << std::endl;
    }



  e.put(bare_product);

  return true;
}


