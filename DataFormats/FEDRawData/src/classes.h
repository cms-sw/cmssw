#include <vector>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <FWCore/EDProduct/interface/Wrapper.h>

 namespace{ 
   namespace {

     std::vector<unsigned char> v;  
     raw::FEDRawData              a1; 
     std::vector<raw::FEDRawData> a2;
     raw::FEDRawDataCollection    a3;
     edm::Wrapper<raw::FEDRawDataCollection> d;
   }
 }
