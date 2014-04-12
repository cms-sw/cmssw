#include <vector>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/Common/interface/Wrapper.h>
#include <DataFormats/Common/interface/RefProd.h>

namespace DataFormats_FEDRawData {
   struct dictionary {
     FEDRawData              a1; 
     std::vector<FEDRawData> a2;
     FEDRawDataCollection    a3;
     edm::Wrapper<FEDRawDataCollection> d;
     edm::RefProd<FEDRawDataCollection> r;
   };
 }
