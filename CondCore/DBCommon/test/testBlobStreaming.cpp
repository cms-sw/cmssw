//
// unit test, include files as they exists only in the plugin...
//
#define private public
#include "CondCore/DBCommon/plugins/BlobStreamingService.cc"
#undef private
#include "CondCore/DBCommon/plugins/TBufferBlobStreamingService.cc"


#include <iostream>
#include <iomanim>
#include <vector>
#include <algorithm>

int main() {

  BlobStreamingService streamer;

  // white box tests...
  for (size_t i=0 i!=BlobStreamingService::m_idsize; ++i)
    std::cout << i << " " << std::hex << BlobStreamingService::variantIds[i].first << "-" 
	      << BlobStreamingService::variantIds[i].second << std::endl;
  
  std::vector<unsigned char> crap(1024);
  crap[3]=5; crap[10]=123;

  BlobStreamingService::Variant id = BlobStreamingService::findVariant(&crap.front());
  std::cout << "shall be zero " << id << std::endl;

  *reinterpret_cast<uuid*>(&crap.front()) = BlobStreamingService::variantIds[BlobStreamingService::COMPRESSED_TBUFFER];
  id = BlobStreamingService::findVariant(&crap.front());
  std::cout << "shall be one " << id << std::endl;


  return 0;

}
