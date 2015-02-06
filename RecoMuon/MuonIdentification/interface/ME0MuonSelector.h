#ifndef RecoMuon_ME0MuonSelectors_h
#define RecoMuon_ME0MuonSelectors_h
//

#include "DataFormats/MuonReco/interface/ME0Muon.h"

#include "Geometry/GEMGeometry/interface/ME0Geometry.h"
#include <Geometry/GEMGeometry/interface/ME0EtaPartition.h>
#include <Geometry/Records/interface/MuonGeometryRecord.h>
#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include "FWCore/Framework/interface/ESHandle.h"

#include "TMath.h"
#include <string>




namespace muon {
   /// Selector type
   enum SelectionType {
      All = 0,                      // dummy options - always true
      VeryLoose = 1,           //
      Loose = 2,           //
      Tight = 3,           //
   };

   /// a lightweight "map" for selection type string label and enum value
   struct SelectionTypeStringToEnum { const char *label; SelectionType value; };
   SelectionType selectionTypeFromString( const std::string &label );
     
   /// main GoodMuon wrapper call
   bool isGoodMuon( edm::ESHandle <ME0Geometry> me0Geom, const reco::ME0Muon& me0muon, SelectionType type );


   /// Specialized isGoodMuon function called from main wrapper

   bool isGoodMuon( edm::ESHandle <ME0Geometry> me0Geom, const reco::ME0Muon& me0muon, double MaxPullX, double MaxDiffX, double MaxPullY, double MaxDiffY, double MaxDiffPhiDir  );


}
#endif
