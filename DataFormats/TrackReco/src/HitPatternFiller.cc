#include "DataFormats/TrackReco/interface/HitPatternFiller.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

using namespace reco;

void HitPatternFiller::fill( const TrackingRecHit & hit, int counter, HitPattern & hitPattern ) {
  DetId id = hit.geographicalId();
  uint32_t valid = static_cast<uint32_t>( hit.isValid() );
  uint32_t pattern = 0;
  uint32_t detid=id.det();
  // adding subdetector bit, removing LS bit (wildcard)
  pattern += ( detid & HitPattern::subDetectorMask ) << HitPattern::subDetectorOffset;
  
  // adding substructure bits, removing LS bit (wildcard)
  uint32_t subdet = id.subdetId();
  
  pattern += ( (subdet) & HitPattern::substrMask) << HitPattern::substrOffset;
  
  uint32_t layer = 0;
  
  // to understand the layer/disk/wheel number, we need to instantiate each 
  if ( detid == DetId::Tracker ) {
    if ( subdet == PixelSubdetector::PixelBarrel )      layer = PXBDetId(id).layer();
    else if (subdet == PixelSubdetector::PixelEndcap)	layer = PXFDetId(id).disk();
    else if (subdet == StripSubdetector::TIB) 	layer = TIBDetId(id).layer();
    else if (subdet == StripSubdetector::TID) 	layer = TIDDetId(id).wheel();
    else if (subdet == StripSubdetector::TOB) 	layer = TOBDetId(id).layer();
    else if (subdet == StripSubdetector::TEC) 	layer = TECDetId(id).wheel();
  } else if (detid == DetId::Muon) {
    if      (subdet == static_cast<uint32_t>( MuonSubdetId::DT  ) )	layer = DTLayerId(id.rawId()).layer();
    else if (subdet == static_cast<uint32_t>( MuonSubdetId::CSC ) )	layer = CSCDetId(id.rawId()).layer();
    else if (subdet == static_cast<uint32_t>( MuonSubdetId::RPC ) )	layer = RPCDetId(id.rawId()).layer();
  }
  pattern += ( layer & HitPattern::layerMask) << HitPattern::layerOffset;
  pattern += ( valid & HitPattern::validMask ) << HitPattern::validOffset;
  
  hitPattern.setHitPattern( counter, pattern );
}
