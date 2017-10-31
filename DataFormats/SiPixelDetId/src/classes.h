#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include <boost/cstdint.hpp> 
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

#include "DataFormats/SiPixelDetId/interface/PixelFEDChannel.h"
#include "DataFormats/Common/interface/EDCollection.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace DataFormats_PixelFEDChannel {
  struct dictionary {
    std::vector<PixelFEDChannel> vFC_;
    edm::EDCollection<PixelFEDChannel> edcFC_;
    edmNew::DetSet<PixelFEDChannel> dsFC_;
    std::vector<edmNew::DetSet<PixelFEDChannel> > vdsFC_;
    edmNew::DetSetVector<PixelFEDChannel> dsvFC_;
    PixelFEDChannelCollection FCc_;

    edm::Wrapper<std::vector<PixelFEDChannel> > vFCw_;
    edm::Wrapper<edm::EDCollection<PixelFEDChannel> > edcFCw_;
    edm::Wrapper<edmNew::DetSet<PixelFEDChannel> > dsFCw_;
    edm::Wrapper<std::vector<edmNew::DetSet<PixelFEDChannel> > > vdsFCw_;
    edm::Wrapper<edmNew::DetSetVector<PixelFEDChannel> > dsvFCw_;
    edm::Wrapper<PixelFEDChannelCollection> FCcw_;
  };
}
