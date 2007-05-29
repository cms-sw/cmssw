#include "IORawData/SiStripInputSources/interface/TBRUInputSource.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "IORawData/SiStripInputSources/src/TBRU.h"
#include "interface/evb/include/i2oEVBMsgs.h"
#include "interface/shared/include/i2oXFunctionCodes.h"
#include "interface/shared/include/frl_header.h"
#include "interface/shared/include/fed_header.h"
#include "interface/shared/include/fed_trailer.h"
#include "Fed9UUtils.hh"
#include "ICExDecl.hh"
#include "TFile.h"
#include "TTree.h"
#include <iostream>
#include <sstream>

using namespace Fed9U;
using namespace ICUtils;
using namespace edm;
using namespace std;
using namespace sistrip;

ClassImp(TBRU)

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
  triggerFedId = pset.getUntrackedParameter<int>("TriggerFedId",1023);
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
    edm::LogWarning(mlInputSource_)
      << "[TBRUInputSource::" << __func__ << "]" 
      << "Unable to open " << filename;
    m_tree=0;
    return;
  } 
  
  // Get the run number
  if ( filename.find("RU") == string::npos ||
       filename.find("_") == string::npos ) {
    n_run = 1;
    edm::LogWarning(mlInputSource_)
      	  << "[TBRUInputSource::" << __func__ << "]" 
	  << " No run number found in 'fileNames' configurable!"
	  << " Expected format is 'RU00xxxxx_yyy.root'"
	  << " Setting run number to '1'";
  } else { 
    unsigned short ipass = filename.find("RU");
    unsigned short ipath = filename.find("_");
    string run;
    run.clear();
    run = filename.substr( ipass+2, ipath-ipass-2 );
    sscanf( run.c_str(), "%d", &n_run );
    LogTrace(mlInputSource_)
      << "[TBRUInputSource::" << __func__ << "]"
      << " Run number: " << run 
      << ", " << run.c_str() 
      << ", " << n_run;
    // printf("%d\n",n_run);
  }
  
  m_tree=(TTree*)m_file->Get("TRU");
  
  if (m_tree==0) {
    m_file->Close();
    m_file=0;
    edm::LogWarning(mlInputSource_)
      << "[TBRUInputSource::" << __func__ << "]"
      << " Unable to find TBRU tree";
    return;
  }
  
  if (!m_quiet) {
    LogTrace(mlInputSource_)
      << "[TBRUInputSource::" << __func__ << "]"
      << " Opening '" << filename << "' with " 
      << m_tree->GetEntries() << " events.";
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
      
    LogTrace(mlInputSource_)
      << "[TBRUInputSource::" << __func__ << "]"
      <<" Branch "<< b->GetName()<<" is found ";

    if (!strcmp(b->GetName(),sizename)) {
      if (m_fed9ubufs[n_fed9ubufs] ==  0)
      {
	m_fed9ubufs[n_fed9ubufs]= new TBRU(i/2);
	LogTrace(mlInputSource_)
	  << "[TBRUInputSource::" << __func__ << "]"
	  << " Creating TBRU " << n_fed9ubufs
	  << " for  Instance " << i/2;
	}
       b->SetAddress(&(m_fed9ubufs[n_fed9ubufs]->fSize));

    } else {
      if (strcmp(b->GetName(),arrayname)) continue;
      b->SetAddress(&(m_fed9ubufs[n_fed9ubufs]->fBuffer));
      n_fed9ubufs++;
    }
  }
  m_i=0;
  LogTrace(mlInputSource_)
    << "[TBRUInputSource::" << __func__ << "]"
    << " File " << filename << " is opened";

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
    LogTrace(mlInputSource_)
      << "[TBRUInputSource::" << __func__ << "]"
      << " Event number " << n_run << ":"
      << block->eventNumber <<" is read ";
  
  setRunNumber(n_run);
  setEventNumber( block->eventNumber);
  
  // time is a hack
  edm::TimeValue_t present_time = presentTime();
  unsigned long time_between_events = timeBetweenEvents();

  setTime(present_time + time_between_events);
  if (!m_quiet)
    LogTrace(mlInputSource_)
      << "[TBRUInputSource::" << __func__ << "]"
      <<" Event & run end";
}
#define FEDID_MASK 0x0003FF00
int TBRUInputSource::getFedId(bool swap, unsigned int* dest) 
{
 if (swap)
 {
 	fedh_t* fedHeader= (fedh_t*) dest;
        return ((fedHeader->sourceid & FEDID_MASK) >>8);
 }
 else
 {
 unsigned int order[1024];
 for (unsigned int i=0;i <sizeof(fedh_t)/sizeof(int);)
 {
   order[i]=dest[i+1];
   order[i+1] = dest[i];
   i+=2;
 }
 fedh_t* fedHeader= (fedh_t*) order;
 return ((fedHeader->sourceid & FEDID_MASK)>>8);
 }
}

bool TBRUInputSource::checkFedStructure(int i, unsigned int* dest,unsigned int &length) 
{
      size_t msgHeaderSize       = sizeof(I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME);
  I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME *block = ( I2O_EVENT_DATA_BLOCK_MESSAGE_FRAME*) m_fed9ubufs[i]->fBuffer;
  unsigned char* cbuf        = (unsigned char*) block + msgHeaderSize;
  int* ibuf = (int*) cbuf;
  // Check old data structure
  bool old = ( (unsigned int) ibuf[0] ==  (m_fed9ubufs[i]->fSize*sizeof(int) - msgHeaderSize));
  // cout << ibuf[0] <<":"<<(m_fed9ubufs[i]->fSize*sizeof(int) - msgHeaderSize) <<endl;
  bool slinkswap = false;
  if (old)
    {
      // Trigger Fed
      if (i == 0)
	{
	  // add fed header
	  fedh_t* fedHeader= (fedh_t*) dest;
          fedHeader->sourceid = triggerFedId<<8;
          fedHeader->eventid  = 0x50000000 | block->eventNumber;
	  // copy data
	  memcpy(&dest[sizeof(fedh_t)/sizeof(int)],ibuf,ibuf[0]);
	  // pay load
	  int tlen = (sizeof(fedh_t) + sizeof(fedt_t)+ibuf[0]);
	  int offset = (tlen%8 ==0)?0:1;
	  // add fed trailer
	  fedt_t* fedTrailer= (fedt_t*) &dest[(sizeof(fedh_t)+ibuf[0])/sizeof(int)+offset];
          fedTrailer->conscheck = 0xDEADFACE;
           fedTrailer->eventsize = 0xA0000000 |
                            ((sizeof(fedh_t) + sizeof(fedt_t)+ibuf[0]+offset*sizeof(int) )>> 3);
	  slinkswap = true;
	  length=(sizeof(fedh_t) + sizeof(fedt_t)+ibuf[0])/sizeof(int)+offset;
	}
      else
	{
	  //copy data
           
	  memcpy(&dest[0],&ibuf[1],ibuf[0]-2*sizeof(int));
	  int fed_len = (ibuf[0]-2*sizeof(int))/sizeof(int);
	  //cout <<"FED Lenght" << fed_len <<endl;
	  //cout <<hex<< dest[fed_len-1]<<":" << dest[fed_len-2]<<dec <<endl;
	  int blen=fed_len*sizeof(int);
	  if ( dest[fed_len-1] ==   (0xa0000000 | (blen>>3)) )
	    slinkswap = true;
	  else
	    if ( dest[fed_len-2] == (0xa0000000 | (blen>>3) ))
	      slinkswap = false;
	    else
	      cout << " Not a FED structure " << i << endl;
	  length =fed_len;
	}
    }
  else
    {
      //copy data
      if (i == 0)
	{
	  //skip the frl header
	  unsigned char* fbuf = cbuf + sizeof(frlh_t);
	  memcpy(&dest[0],fbuf,(m_fed9ubufs[i]->fSize*sizeof(int) - msgHeaderSize- sizeof(frlh_t)));	  
	  slinkswap = true;
	  length = (m_fed9ubufs[i]->fSize*sizeof(int) - msgHeaderSize- sizeof(frlh_t))/sizeof(int);
	}
      else
	{

	  unsigned char* fbuf = cbuf + sizeof(frlh_t);
	  memcpy(&dest[0],fbuf,(m_fed9ubufs[i]->fSize*sizeof(int) - msgHeaderSize)-sizeof(frlh_t));

	  //int frl_len = m_fed9ubufs[i]->fSize - msgHeaderSize/sizeof(int);
	  unsigned int* fedb = &dest[0];
	  int fed_len = m_fed9ubufs[i]->fSize - msgHeaderSize/sizeof(int)- sizeof(frlh_t)/sizeof(int);
	  //cout <<hex<< dest[frl_len-1]<<":" << dest[frl_len-2]<<dec <<endl;
	  int fedlen = (fed_len*sizeof(int))>>3;
	  if ( fedb[fed_len-1] == (0xa0000000 | (fedlen) ))
	    slinkswap = true;
	  else
	    if ( fedb[fed_len-2] == (0xa0000000 | (fedlen) ))
	      slinkswap = false;
	    else
	      cout << " Not a FED structure " << i << endl;
	  
	  length =fed_len;
	}

    }

  return slinkswap;
}
bool TBRUInputSource::produce(edm::Event& e) {


  static unsigned int output[96*1024];
  if (m_tree==0) return false;
  Fed9U::Fed9UEvent fedEvent;
  std::auto_ptr<FEDRawDataCollection> bare_product(new  FEDRawDataCollection());
  for (int i=0; i<n_fed9ubufs; i++) 
    {
    	unsigned int fed_len;
     	bool slinkswap = checkFedStructure(i,output,fed_len);
	int fed_id = getFedId(slinkswap,output);
	//cout <<fed_id << ":"<<fed_len <<endl;
      if (!m_quiet) {
	stringstream ss;
	ss << "[TBRUInputSource::" << __func__ << "]"
	   << " Reading bytes for FED " << i
	   << " At address " <<hex<< m_fed9ubufs[i] << dec;
	LogTrace(mlInputSource_) << ss.str();
      }
#ifdef OLDSTYLE
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
	      	LogTrace(mlInputSource_)
		  << "[TBRUInputSource::" << __func__ << "]"
		  <<" Fed ID is "<<fedEvent.getSourceId();

	    } catch (ICUtils::ICException &ex)
	      {
		stringstream ss;
		ss << "[TBRUInputSource::" << __func__ << "]" << endl
		   << "======================================================>  ERROR in " << "\n"
		   << e.id().run() <<"::"<<e.id().event()<<"- Super fragment -" << i<< "\n"
		   << ex.what() << "\n"
		   << "Cannot construct FED Event: Fed ID Might be "<<fedEvent.getSourceId() << "\n" 
		   << "======================================================>" << "\n";
		edm::LogWarning(mlInputSource_) << ss.str();
		continue;
	      }
	  
	}
      

      if (id!=1023)
	{
	  const unsigned char* data=(const unsigned char*) &ibuf[1];
	  FEDRawData& fed=bare_product->FEDData(id);
	  int fulllen =(len%8)?len+8-len%8:len;
	  fed.resize(fulllen);
	  memcpy(fed.data(),data,len);
	}
      else
	{
	}

      if (!m_quiet)
	LogTrace(mlInputSource_)
	  << "[TBRUInputSource::" << __func__ << "]" 
	  << " Reading " << len << " bytes for FED " << id;
    }
#else
	FEDRawData& fed=bare_product->FEDData(fed_id);
	fed.resize(fed_len*sizeof(int));
	memcpy(fed.data(),output,fed_len*sizeof(int));

	if (!m_quiet)
	  LogTrace(mlInputSource_) 
	    << "[TBRUInputSource::" << __func__ << "]" 
	    << " Reading " << fed_len << " bytes for FED " << fed_id;
    }
#endif

  e.put(bare_product);

  return true;
}


