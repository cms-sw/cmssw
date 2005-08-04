#include "TFile.h"
#include "TTree.h"
#include "IORawData/HcalTBInputService/interface/HcalTBInputService.h"
#include "CDFChunk.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/EDProduct/interface/CollisionID.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include "PluginManager/PluginCapabilities.h"
#include <iostream>

ClassImp(CDFChunk);

using namespace edm;
using namespace std;

namespace cms {
namespace hcal {
  class TBFakeRetriever : public Retriever {
  public:
    virtual ~TBFakeRetriever(){}
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k){
      throw std::runtime_error("TBFakeRetriever::get called");
      return std::auto_ptr<EDProduct>(0);
    }
  };


HcalTBInputService::HcalTBInputService(const edm::ParameterSet & pset, edm::InputServiceDescription const& desc) : 
  edm::InputService(desc),
  file_( pset.getParameter<std::string>("fileName") ),
  retriever_( new TBFakeRetriever() ),
  m_imax( pset.getParameter<int>("maxEvents") ),
  m_hcalFedOffset( pset.getUntrackedParameter<int>("hcalFedOffset",0) ),
  m_duplicateChunkAs( pset.getUntrackedParameter<int>("DuplicateChuckAs",-1) )
{
  m_tree=0;
  initThis();
  edm::Wrapper<raw::FEDRawDataCollection> wrappedProd;
  edm::TypeID myType(wrappedProd);
  prodDesc_.fullClassName_=myType.userClassName();
  prodDesc_.friendlyClassName_ = myType.friendlyClassName(); 

  prodDesc_.module.pid = PS_ID("HcalTBInputService");
  prodDesc_.module.moduleName_ = "HcalTBInputService";
  prodDesc_.module.moduleLabel_ = "HcalTBInputService";
  prodDesc_.module.versionNumber_ = 2UL;
  prodDesc_.module.processName_ = "DaqProd";
  prodDesc_.module.pass = 1UL;  

  preg_->addProduct(prodDesc_);

  if (m_hcalFedOffset!=0) {
    cout << "HcalTBInputService adding " << m_hcalFedOffset << " to DCC SourceIds" << std::endl;
  }
}

void HcalTBInputService::initThis() {
  m_file=new TFile(file_.c_str());
  if (m_file==0) {
    cout << "Unable to open " << file_ << endl;
    m_tree=0;
    return;
  }
  m_tree=(TTree*)m_file->Get("CMSRAW");
  TObjArray* lb=m_tree->GetListOfBranches();
  n_chunks=0;
  for (int i=0; i<lb->GetSize(); i++) {
    TBranch* b=(TBranch*)lb->At(i);
    if (b==0) continue;
    if (strcmp(b->GetClassName(),"CDFChunk")) continue;
    if (strstr(b->GetName(),"HCAL_DCC")==NULL) continue;
    m_chunks[n_chunks]=new CDFChunk();
    b->SetAddress(&(m_chunks[n_chunks]));
    n_chunks++;
  }
  m_i=0;
}

std::auto_ptr<edm::EventPrincipal> HcalTBInputService::read() {
  if (m_tree==0 || m_i==m_tree->GetEntries() || 
      (m_imax>0 && m_imax<=m_i)) return auto_ptr<EventPrincipal>(0);

  m_tree->GetEntry(m_i);
  m_i++;

  CollisionID id=m_i-1;

  auto_ptr<EventPrincipal> result = auto_ptr<EventPrincipal>(new EventPrincipal(id,  *retriever_,*preg_));
 
  raw::FEDRawDataCollection *bare_product = new raw::FEDRawDataCollection();  
  for (int i=0; i<n_chunks; i++) {
    const unsigned char* data=(const unsigned char*)m_chunks[i]->getData();
    int len=m_chunks[i]->getDataLength()*8;

    int id=m_hcalFedOffset+m_chunks[i]->getSourceId(); 

    raw::FEDRawData& fed=bare_product->FEDData(id);
    fed.data_.clear();
    fed.data_.reserve(len);
    for (int j=0; j<len; j++)
      fed.data_.push_back(data[j]);
    // patch the SourceId...
    if (m_hcalFedOffset!=0) {
      unsigned int* header=(unsigned int*)fed.data();
      header[0]=(header[0]&0xFFF000FFu)|(id<<8);
      // TODO: patch CRC after this change!
    }
    std::cout << "Reading " << len << " bytes for FED " << id << std::endl;

    // chunk duplication ( for HO testing, mostly )
    if (m_duplicateChunkAs>0) {
      id=m_duplicateChunkAs;
      raw::FEDRawData& fed2=bare_product->FEDData(id);
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
  }
  edm::Wrapper<raw::FEDRawDataCollection>* full_prod=new edm::Wrapper<raw::FEDRawDataCollection>(*bare_product);
  auto_ptr<EDProduct>  prod(full_prod);
  auto_ptr<Provenance> prov(new Provenance(prodDesc_));
  result->put(prod, prov);
  
  return result;
}
}
}
