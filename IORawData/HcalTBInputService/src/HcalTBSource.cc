#include "TFile.h"
#include "TTree.h"
#include "IORawData/HcalTBInputService/interface/HcalTBSource.h"
#include "CDFChunk.h"
#include "CDFEventInfo.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/EDProduct/interface/EventID.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include "PluginManager/PluginCapabilities.h"
#include <iostream>

ClassImp(CDFChunk);
ClassImp(CDFEventInfo);

using namespace edm;
using namespace std;



HcalTBSource::HcalTBSource(const edm::ParameterSet & pset, edm::InputSourceDescription const& desc) : 
  edm::InputSource(desc),
  files_( pset.getParameter<std::vector<std::string> >("fileNames") ),
  m_imax( pset.getParameter<int>("maxEvents") ),
  m_quiet( pset.getUntrackedParameter<bool>("quiet",true))
{
  m_tree=0;
  m_file=0;
  fileCounter_=-1;
  m_itotal=0;
  m_i=0;

  edm::Wrapper<FEDRawDataCollection> wrappedProd;
  edm::TypeID myType(wrappedProd);
  prodDesc_.fullClassName_=myType.userClassName();
  prodDesc_.friendlyClassName_ = myType.friendlyClassName(); 

  prodDesc_.module.pid = PS_ID("HcalTBSource");
  prodDesc_.module.moduleName_ = "HcalTBSource";
  prodDesc_.module.moduleLabel_ = "HcalTBSource";
  prodDesc_.module.versionNumber_ = 2UL;
  prodDesc_.module.processName_ =desc.processName_;
  prodDesc_.module.pass = desc.pass;  

  preg_->addProduct(prodDesc_);

  unpackSetup(pset.getParameter<std::vector<std::string> >("streams"));
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
      cout << streamName << " --> " << remapTo << endl;
    else
      cout << streamName << " using fedid in file" << endl;
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
    cout << "Unable to open " << filename << endl;
    m_tree=0;
    return;
  } 
  
  m_tree=(TTree*)m_file->Get("CMSRAW");
  
  if (m_tree==0) {
    m_file->Close();
    m_file=0;
    cout << "Unable to find CMSRAW tree" << endl;
    return;
  }
  
  if (!m_quiet) {
    cout << "Opening '" << filename << "' with " << m_tree->GetEntries() << " events.\n";
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
      if (m_sourceIdRemap.find(b->GetName())==m_sourceIdRemap.end()) continue;
      
      m_chunks[n_chunks]=0; // allow ROOT to allocate 
      b->SetAddress(&(m_chunks[n_chunks]));
      m_chunkIds[n_chunks]=m_sourceIdRemap[b->GetName()];
      n_chunks++;
    }
  }
  m_i=0;
}

std::auto_ptr<edm::EventPrincipal> HcalTBSource::read() {
  if (m_imax>0 && m_imax<=m_itotal) return auto_ptr<EventPrincipal>(0);
  while (m_tree==0 || m_i==m_tree->GetEntries()) {
    fileCounter_++;
    if (fileCounter_>=int(files_.size())) return  auto_ptr<EventPrincipal>(0);
    openFile(files_[fileCounter_]);
  }

  if (m_tree==0 || m_i==m_tree->GetEntries()) return  auto_ptr<EventPrincipal>(0);

  m_tree->GetEntry(m_i);
  m_i++;
  m_itotal++;
  
  EventID id=((m_eventInfo==0)?(EventID(fileCounter_,m_i-1)):(EventID(m_eventInfo->getRunNumber(),m_eventInfo->getEventNumber())));
  unsigned long long evtTime=m_itotal*10000; // hack  -- time is only in the trigger blocks which I do not want to unpack here
 
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
      std::cout << "Reading " << len << " bytes for FED " << id << std::endl;

    // chunk duplication ( for HO testing, mostly )
    /*
    if (m_duplicateChunkAs>0) {
      id=m_duplicateChunkAs;
      FEDRawData& fed2=bare_product->FEDData(id);
      fed2.data_.clear();
      fed2.data_.reserve(len);
      for (int j=0; j<len; j++)
	fed2.data_.push_back(data[j]);
      // patch the SourceId...
      unsigned int* header=(unsigned int*)fed2.data();
      header[0]=(header[0]&0xFFF000FFu)|(id<<8);
      // TODO: patch CRC after this change!
      std::cout << "Duplicating chunk to produce FED " << id << std::endl;
    }
    */
  }

  edm::Wrapper<FEDRawDataCollection>* full_prod=new edm::Wrapper<FEDRawDataCollection>(*bare_product);
  auto_ptr<EventPrincipal> result = auto_ptr<EventPrincipal>(new EventPrincipal(id, Timestamp(evtTime),*preg_));
  auto_ptr<EDProduct>  prod(full_prod);
  auto_ptr<Provenance> prov(new Provenance(prodDesc_));
  result->put(prod, prov);
  
  return result;
}


