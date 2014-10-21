#include <vector>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h>
#include <DataFormats/FEDRawData/interface/FEDHeader.h>
#include <DataFormats/FEDRawData/interface/FEDTrailer.h>
#include <DataFormats/FEDRawData/src/fed_header.h>
#include <DataFormats/FEDRawData/src/fed_trailer.h>
#include <DataFormats/Common/interface/Wrapper.h>
#include <DataFormats/Common/interface/RefProd.h>

namespace DataFormats_FEDRawData {
   struct dictionary {
     FEDRawData              a1; 
     std::vector<FEDRawData> a2;
     FEDRawDataCollection    a3;
     FEDHeader               b1;
     FEDTrailer              c1;
     fedh_struct             d1;
     fedt_struct             e1;
     edm::Wrapper<FEDRawDataCollection> d;
     edm::RefProd<FEDRawDataCollection> r;
   };
 }
