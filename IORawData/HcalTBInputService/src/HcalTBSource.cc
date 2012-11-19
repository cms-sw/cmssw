#include "TFile.h"
#include "TTree.h"
#include "IORawData/HcalTBInputService/interface/HcalTBSource.h"
#include "CDFChunk.h"
#include "CDFEventInfo.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

ClassImp(CDFChunk)
ClassImp(CDFEventInfo)

using namespace edm;
using namespace std;

HcalTBSource::HcalTBSource(const edm::ParameterSet & pset, edm::InputSourceDescription const& desc) : 
  edm::ProducerSourceFromFiles(pset,desc, true),
  m_quiet( pset.getUntrackedParameter<bool>("quiet",true)),
  m_onlyRemapped( pset.getUntrackedParameter<bool>("onlyRemapped",false))
{
  m_tree=0;
  m_fileCounter=-1;
  m_file=0;
  m_i=0;

  unpackSetup(pset.getUntrackedParameter<std::vector<std::string> >("streams",std::vector<std::string>()));
  produces<FEDRawDataCollection>();
}

void HcalTBSource::unpackSetup(const std::vector<std::string>& params) {
  for (std::vector<std::string>::const_iterator i=params.begin(); i!=params.end(); i++) {
    unsigned long pos=i->find(':');
    std::string streamName=i->substr(0,pos);
    int remapTo=-1;
    if (pos!=std::string::npos) 
      remapTo=atoi(i->c_str()+pos+1);
    
    m_sourceIdRemap.insert(std::pair<std::string,int>(streamName,remapTo));
    if (remapTo!=-1) 
      edm::LogInfo("HCAL") << streamName << " --> " << remapTo << endl;
    else
      edm::LogInfo("HCAL") << streamName << " using fedid in file" << endl;
  }
}

HcalTBSource::~HcalTBSource() {
  if (m_file!=0) {
    m_file->Close();
    m_file=0;
    m_tree=0;
  }
}

void HcalTBSource::openFile(const std::string& filename) {
  if (m_file!=0) {
    m_file->Close();
    m_file=0;
    m_tree=0;
  }
  
  //  try {
  m_file=TFile::Open(filename.c_str());
  if (m_file==0) {
    edm::LogError("HCAL") << "Unable to open " << filename << endl;
    m_tree=0;
    return;
  } 
  
  m_tree=(TTree*)m_file->Get("CMSRAW");
  
  if (m_tree==0) {
    m_file->Close();
    m_file=0;
    edm::LogError("HCAL") << "Unable to find CMSRAW tree" << endl;
    return;
  }
  
  if (!m_quiet) {
    edm::LogInfo("HCAL") << "Opening '" << filename << "' with " << m_tree->GetEntries() << " events.\n";
  }
  
  TObjArray* lb=m_tree->GetListOfBranches();
  n_chunks=0;
  for (int i=0; i<lb->GetSize(); i++) {
    TBranch* b=(TBranch*)lb->At(i);
    if (b==0) continue;
    if (!strcmp(b->GetClassName(),"CDFEventInfo")) {
      m_eventInfo=0;
      b->SetAddress(&m_eventInfo);
    } else {
      if (strcmp(b->GetClassName(),"CDFChunk")) continue;
      if (m_sourceIdRemap.find(b->GetName())==m_sourceIdRemap.end()) {
	if (m_onlyRemapped) continue;
	m_sourceIdRemap.insert(std::pair<std::string,int>(b->GetName(),-1));
	if (!m_quiet)
	  edm::LogInfo("HCAL") << "Also reading branch " << b->GetName();
      }
      
      m_chunks[n_chunks]=0; // allow ROOT to allocate 
      b->SetAddress(&(m_chunks[n_chunks]));
      m_chunkIds[n_chunks]=m_sourceIdRemap[b->GetName()];
      n_chunks++;
    }
  }
  m_i=0;
}

bool HcalTBSource::setRunAndEventInfo(EventID& id, TimeValue_t& time) {
  bool is_new=false;

  while (m_tree==0 || m_i==m_tree->GetEntries()) {
    m_fileCounter++;
    if (m_file!=0) {
       m_file->Close();
       m_file=0; 
       m_tree=0;
    }
    if (m_fileCounter>=int(fileNames().size())) return false; // nothing good
    openFile(fileNames()[m_fileCounter]);
    is_new=true;
  }

  if (m_tree==0 || m_i==m_tree->GetEntries()) return false; //nothing good

  m_tree->GetEntry(m_i);
  m_i++;

  if (m_eventInfo!=0) {
    if (is_new) {
      if (m_eventInfo->getEventNumber()==0) m_eventNumberOffset=1;
      else m_eventNumberOffset=0;
    }
    // ZERO is unacceptable for a run number from a technical point of view
    id = EventID((m_eventInfo->getRunNumber()==0 ? 1 : m_eventInfo->getRunNumber()), id.luminosityBlock(), m_eventInfo->getEventNumber()+m_eventNumberOffset); 
  } else {
    id = EventID(m_fileCounter+10, id.luminosityBlock(), m_i+1);
  }  
  // time is a hack
  edm::TimeValue_t present_time = presentTime();
  unsigned long time_between_events = timeBetweenEvents();

  time = present_time + time_between_events;
  return true;
}

void HcalTBSource::produce(edm::Event& e) {

  std::auto_ptr<FEDRawDataCollection> bare_product(new  FEDRawDataCollection());
  for (int i=0; i<n_chunks; i++) {
    const unsigned char* data=(const unsigned char*)m_chunks[i]->getData();
    int len=m_chunks[i]->getDataLength()*8;

    int natId=m_chunks[i]->getSourceId(); 
    int id=(m_chunkIds[i]>0)?(m_chunkIds[i]):(natId);
    
    FEDRawData& fed=bare_product->FEDData(id);
    fed.resize(len);
    memcpy(fed.data(),data,len);

    // patch the SourceId...
    if (natId!=id) {
      unsigned int* header=(unsigned int*)fed.data();
      header[0]=(header[0]&0xFFF000FFu)|(id<<8);
      // TODO: patch CRC after this change!
    }
    if (!m_quiet) 
      edm::LogInfo("HCAL") << "Reading " << len << " bytes for FED " << id << std::endl;
  }

  e.put(bare_product);
}


