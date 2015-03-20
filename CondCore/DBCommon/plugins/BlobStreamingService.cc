#include "CondCore/DBCommon/interface/BlobStreamerPluginFactory.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondCore/ORA/interface/IBlobStreamingService.h"

#include "TBufferBlobStreamingService.h"


//
#include <cstddef>
//
#include "CoralBase/Blob.h"
//
#include <algorithm>
#include <typeinfo>
#include <string>
#include <cstring>
#include <zlib.h>

namespace cond {

  class BlobStreamingService : virtual public ora::IBlobStreamingService {
  public:
    
    
    BlobStreamingService();
    
    virtual ~BlobStreamingService();
    
    boost::shared_ptr<coral::Blob> write( const void* addressOfInputData,  edm::TypeWithDict const & classDictionary, bool useCompression=true ) override;
    
    void read( const coral::Blob& blobData, void* addressOfContainer,  edm::TypeWithDict const & classDictionary ) override;
    
    
  private:
    
    typedef std::pair<unsigned long long, unsigned long long> uuid;
    
    static const size_t m_idsize=sizeof(uuid);
    static const size_t m_offset = m_idsize + sizeof(unsigned long long);
    static const size_t nVariants=3;
    
    enum Variant { OLD, COMPRESSED_TBUFFER, COMPRESSED_CHARS }; 
    static uuid const variantIds[nVariants];
    
    
    static Variant findVariant(const void* address);
    static int isVectorChar(edm::TypeWithDict const & classDictionary);
    
    
    static boost::shared_ptr<coral::Blob>  compress(const void* addr, size_t isize);
    static boost::shared_ptr<coral::Blob>  expand(const coral::Blob& blobIn);
    
    
    boost::shared_ptr<ora::IBlobStreamingService> rootService; 
    
  };
  
  
  BlobStreamingService::BlobStreamingService() : rootService(new cond::TBufferBlobStreamingService()){}
  
  BlobStreamingService::~BlobStreamingService(){}
  
  boost::shared_ptr<coral::Blob> BlobStreamingService::write( const void* addressOfInputData,  edm::TypeWithDict const & classDictionary, bool useCompression ) {
    boost::shared_ptr<coral::Blob> blobOut;
    int const k = isVectorChar(classDictionary);
    switch (k) {
    case 0 : 
      {
	// at the moment we write TBuffer compressed, than we see....
	// we may wish to avoid one buffer copy...
	boost::shared_ptr<coral::Blob> buffer = rootService->write(addressOfInputData, classDictionary);
        if( useCompression ){
          blobOut = compress(buffer->startingAddress(),buffer->size());
       	  *reinterpret_cast<uuid*>(blobOut->startingAddress()) = variantIds[COMPRESSED_TBUFFER];
	} else {
          blobOut = buffer;
	}
      }
      break;
    case 1 : 
      {
        if( useCompression ){
	  std::vector<unsigned char> const & v = *reinterpret_cast< std::vector<unsigned char> const *> (addressOfInputData);
	  blobOut = compress(&v.front(),v.size());
	  *reinterpret_cast<uuid*>(blobOut->startingAddress()) = variantIds[COMPRESSED_CHARS];
	} else {
          blobOut = rootService->write(addressOfInputData,classDictionary);
	}
      }
      break;
    case 2 : 
      {
        if( useCompression ){
          std::vector<char> const & v = *reinterpret_cast<std::vector<char> const *> (addressOfInputData);
	  blobOut = compress(&v.front(),v.size());
	  *reinterpret_cast<uuid*>(blobOut->startingAddress()) = variantIds[COMPRESSED_CHARS];
	} else {
          blobOut = rootService->write(addressOfInputData,classDictionary);
	}
      }
      break;
      
    }
    return blobOut;

  }
  
  
  void BlobStreamingService::read( const coral::Blob& blobData, void* addressOfContainer,  edm::TypeWithDict const & classDictionary ) {
    // protect against small blobs...
    Variant v =  (size_t(blobData.size()) < m_offset) ? OLD : findVariant(blobData.startingAddress());
    switch (v) {
    case OLD :
      {
	rootService->read( blobData, addressOfContainer, classDictionary);
      }
      break;
    case COMPRESSED_TBUFFER :
      {
	boost::shared_ptr<coral::Blob> blobIn = expand(blobData);
	rootService->read( *blobIn, addressOfContainer, classDictionary);
      }
      break;
    case COMPRESSED_CHARS :
      {
	boost::shared_ptr<coral::Blob> blobIn = expand(blobData);
	int const k = isVectorChar(classDictionary);
	switch (k) {
	case 0 : 
	  {
	    // error!!!
	  }
	  break;
	case 1:
	  {
	    std::vector<unsigned char> & v = *reinterpret_cast< std::vector<unsigned char> *> (addressOfContainer);
	    // we should avoid the copy!
	    v.resize(blobIn->size());
	    std::memcpy(&v.front(),blobIn->startingAddress(),v.size());
	  }
	  break;
	case 2:
	  {
	    std::vector<char> & v = *reinterpret_cast< std::vector<char> *> (addressOfContainer);
	    // we should avoid the copy!
	    v.resize(blobIn->size());
	    std::memcpy(&v.front(),blobIn->startingAddress(),v.size());
	  }
	  break;
	}
      }
    }
  }
      
    
    
    
  const BlobStreamingService::uuid BlobStreamingService::variantIds[nVariants] = {
    BlobStreamingService::uuid(0,0)
    ,BlobStreamingService::uuid(0xf4e92f169c974e8eLL, 0x97851f372586010dLL)
    ,BlobStreamingService::uuid(0xc9a95a45e60243cfLL, 0x8dc549534f9a9274LL)
  };
  
  
  int BlobStreamingService::isVectorChar(edm::TypeWithDict const& classDictionary) {
    if (classDictionary == typeid(std::vector<unsigned char>)) return 1;
    if (classDictionary == typeid(std::vector<char>)) return 2;
    return 0;
  }
  
  
  
  BlobStreamingService::Variant BlobStreamingService::findVariant(const void* address) {
    uuid const & id = *reinterpret_cast<uuid const*>(address);
    uuid const *  v = std::find(variantIds,variantIds+nVariants,id);
    return (v==variantIds+nVariants) ? OLD : (Variant)(v-variantIds);
  }
  
  
  boost::shared_ptr<coral::Blob> BlobStreamingService::compress(const void* addr, size_t isize) {
    uLongf destLen = compressBound(isize);
    size_t usize = destLen + m_offset;
    boost::shared_ptr<coral::Blob> theBlob( new coral::Blob(usize));
     void * startingAddress = (unsigned char*)(theBlob->startingAddress())+ m_offset;
    int zerr =  compress2( (unsigned char*)(startingAddress), &destLen,
			   (unsigned char*)(addr), isize,
			  9);
    if (zerr!=0) edm::LogError("BlobStreamingService")<< "Compression error " << zerr;
    destLen+= m_idsize + sizeof(unsigned long long);
    theBlob->resize(destLen);

    startingAddress = (unsigned char*)(theBlob->startingAddress())+ m_idsize;
    // write expanded size;
    *reinterpret_cast<unsigned long long*>(startingAddress)=isize; 
    return theBlob;
  }

  boost::shared_ptr<coral::Blob> BlobStreamingService::expand(const coral::Blob& blobIn) {
    if (size_t(blobIn.size()) < m_offset) return boost::shared_ptr<coral::Blob>(new coral::Blob());
    long long csize =  blobIn.size() - m_offset;
    unsigned char const * startingAddress = (unsigned char const*)(blobIn.startingAddress())+ m_idsize;
    unsigned long long usize = *reinterpret_cast<unsigned long long const*>(startingAddress);
    startingAddress +=  sizeof(unsigned long long);
    boost::shared_ptr<coral::Blob> theBlob( new coral::Blob(usize));
    uLongf destLen = usize;
    int zerr =  uncompress((unsigned char *)(theBlob->startingAddress()),  &destLen,
			   startingAddress, csize);
    if (zerr!=0 || usize!=destLen) 
      edm::LogError("BlobStreamingService")<< "uncompressing error " << zerr
			       << " original size was " << usize
			       << " new size is " << destLen;
    
     return theBlob; 
  }

}

// keep the old good name
DEFINE_EDM_PLUGIN(cond::BlobStreamerPluginFactory,cond::BlobStreamingService,"COND/Services/BlobStreamingService");
