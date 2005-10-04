#include <vector>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <FWCore/EDProduct/interface/Wrapper.h>

 namespace{ 
   namespace {

     std::vector<unsigned char> v;  
     FEDRawData              a1; 
     std::vector<FEDRawData> a2;
     FEDRawDataCollection    a3;
     edm::Wrapper<FEDRawDataCollection> d;
   }
 }
