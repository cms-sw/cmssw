#include "TFile.h"
#include "TTree.h"

#include "IORawData/SiStripInputSources/interface/TBRUInputSource.h"
#include "TBRU.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
//#include "PluginManager/PluginCapabilities.h"

#include "interface/evb/include/i2oEVBMsgs.h"
#include "interface/shared/include/i2oXFunctionCodes.h"
#include "Fed9UUtils.hh"
#include <iostream>
#include <sstream>
using namespace  Fed9U;

#include "ICExDecl.hh"
using namespace  ICUtils;
ClassImp(TBRU);


using namespace edm;
using namespace std;

TBRUInputSource::TBRUInputSource(const edm::ParameterSet & pset, edm::InputSourceDescription const& desc) : 
  edm::ExternalInputSource(pset,desc),
  m_quiet( pset.getUntrackedParameter<bool>("quiet",true)),
  m_branches(-1),
  nfeds(-1) // R.B.
{
  m_tree=0;
  m_fileCounter=-1;
  m_file=0;
  m_i=0;
  n_fed9ubufs=0;
  for (int i=0;i<MAX_FED9U_BUFFER;i++)
    m_fed9ubufs[i]=0;
  nfeds = pset.getUntrackedParameter<int>("nFeds",-1); // R.B.
  
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
    edm::LogError("TBRU") << "Unable to open " << filename;
    m_tree=0;
    return;
  } 
  
  // Get the run number

  int ipass = filename.find("RU");
  int ipath = filename.find("_");
  string run;
  run.clear();
  run=filename.substr(ipass+2,ipath-ipass-2);
  LogDebug("TBRU") << run << " et " << run.c_str();
  sscanf(run.c_str(),"%d",&n_run);
  printf("%d\n",n_run);
  //

  m_tree=(TTree*)m_file->Get("TRU");
  
  if (m_tree==0) {
    m_file->Close();
    m_file=0;
    edm::LogError("TBRU") << "Unable to find TBRU tree";
    return;
  }
  
  if (!m_quiet) {
    LogDebug("TBRU") << "Opening '" << filename << "' with " << m_tree->GetEntries() << " events.";
  }

  TObjArray* lb=m_tree->GetListOfBranches();
  n_fed9ubufs=0;
  m_branches = (nfeds<0) ? lb->GetSize() : (nfeds+1)*2; // R.B.
  for (int i=0; i<lb->GetSize(); i++) {
    TBranch* b=(TBranch*)lb->At(i);
    if (b==0) continue;

    char sizename[256];
    char arrayname[256];
    sprintf(sizename,"size_RU_%x",i/2);

      
    sprintf(arrayname,"RU_%x",i/2);
      

    LogDebug("TBRU") <<"Branch "<< b->GetName()<<" is found ";
    if (!strcmp(b->GetName(),sizename)) {
      if (m_fed9ubufs[n_fed9ubufs] ==  0)
      {
	m_fed9ubufs[n_fed9ubufs]= new TBRU(i/2);
	LogDebug("TBRU") << "Creating TBRU " << n_fed9ubufs<< " for  Instance " << i/2;
	}
       b->SetAddress(&(m_fed9ubufs[n_fed9ubufs]->fSize));

    } else {
      if (strcmp(b->GetName(),arrayname)) continue;
      b->SetAddress(&(m_fed9ubufs[n_fed9ubufs]->fBuffer));
      n_fed9ubufs++;
    }
  }
  m_i=0;
  LogDebug("TBRU") << "File " << filename << " is opened";
  if (nfeds>0) n_fed9ubufs = m_branches/2; //R.B.
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
  m_i++;
  TBRU* r= m_fed9ubufs[0];
  int* rud =r->fBuffer;

  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block = ( I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) rud;
   if ( (!m_quiet) || block->eventNumber%100 == 0) 
    LogDebug("TBRU") << "Event number " << n_run <<":"<< block->eventNumber <<" is read ";
    setRunNumber(n_run);
    setEventNumber( block->eventNumber);

  // time is a hack
  edm::TimeValue_t present_time = presentTime();
  unsigned long time_between_events = timeBetweenEvents();

  setTime(present_time + time_between_events);
  if (!m_quiet)
  	LogDebug("TBRU") <<"Event & run end";
}

bool TBRUInputSource::produce(edm::Event& e) {

  if (m_tree==0) return false;
  Fed9U::Fed9UEvent fedEvent;
  std::auto_ptr<FEDRawDataCollection> bare_product(new  FEDRawDataCollection());
  for (int i=0; i<n_fed9ubufs; i++) 
    {
      if (!m_quiet) {
	stringstream ss;
	ss << "Reading bytes for FED " << i << " At address " <<hex<< m_fed9ubufs[i] << dec;
	LogDebug("TBRU") << ss.str();
      }
      I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block = ( I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) m_fed9ubufs[i]->fBuffer;
      size_t msgHeaderSize       = sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
      
      // Find FED id
      unsigned char* cbuf        = (unsigned char*) block + msgHeaderSize;
      int* ibuf = (int*) cbuf;
      int len = (ibuf[0]-8);
      int id=0;
      if (i == 0) // TA RU
	id =1023;
      else
	{ // FED9U super fragment
	  // unsigned char* _buffer= (unsigned char*) &ibuf[1];
	  //  id = ((_buffer[5 ^ 3]<<8) | _buffer[6 ^ 3]) ;
	 // id=50+i;
	  int* rudat = &ibuf[1];
	  int buf_len= ibuf[0]/sizeof(int)-2;
	  try
	    {
	      fedEvent.Init((u32*) rudat,0,(u32)buf_len);
	      fedEvent.checkEvent();
              id = fedEvent.getSourceId();
	      if (!m_quiet)
	      	LogDebug("TBRU")<<"Fed ID is "<<fedEvent.getSourceId();

	    } catch (ICUtils::ICException &ex)
	      {
		stringstream ss;
		ss <<"======================================================>  ERROR in " << "\n"
		   << e.id().run() <<"::"<<e.id().event()<<"- Super fragment -" << i<< "\n"
		   << ex.what() << "\n"
		   <<"Cannot construct FED Event: Fed ID Might be "<<fedEvent.getSourceId() << "\n" 
		   <<"======================================================>" << "\n";
		edm::LogError("TBRU") << ss.str();
		continue;
	      }
	  
	}
      
      const unsigned char* data=(const unsigned char*) &ibuf[1];

      FEDRawData& fed=bare_product->FEDData(id);
      int fulllen =(len%8)?len+8-len%8:len;
      fed.resize(fulllen);
      memcpy(fed.data(),data,len);

      if (!m_quiet)
	LogDebug("TBRU") << "Reading " << len << " bytes for FED " << id;
    }



  e.put(bare_product);

  return true;
}


