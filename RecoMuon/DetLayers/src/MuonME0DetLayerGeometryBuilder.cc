#include <RecoMuon/DetLayers/src/MuonME0DetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
//#include <RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h>
#include "RecoMuon/DetLayers/interface/MuRingForwardLayer.h"
#include "RecoMuon/DetLayers/interface/MuRodBarrelLayer.h"
#include "RecoMuon/DetLayers/interface/MuDetRing.h"
#include "RecoMuon/DetLayers/interface/MuDetRod.h"

#include "Utilities/General/interface/precomputed_value_sort.h"
#include "Geometry/CommonDetUnit/interface/DetSorting.h"
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

MuonME0DetLayerGeometryBuilder::~MuonME0DetLayerGeometryBuilder() {
}

// Builds the forward (first) and backward (second) layers - NOTE: Currently just one layer, all 'front'
// Builds chambers (for segments) only, me0RecHits not used in track finding
pair<vector<DetLayer*>, vector<DetLayer*> > 
MuonME0DetLayerGeometryBuilder::buildEndcapLayers(const ME0Geometry& geo) {
  
  vector<DetLayer*> result[2];
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonME0DetLayerGeometryBuilder";
  LogTrace(metname) << "Starting endcaplayers ";
  for (int endcap = -1; endcap<=1; endcap+=2) {
    int iendcap = (endcap==1) ? 0 : 1; // +1: forward, -1: backward
    int layer = 0;// only need to make layer 0 for segments
    vector<int> rolls, chambers;      
    rolls.push_back(0);
    for(int chamber = ME0DetId::minChamberId+1; chamber <= ME0DetId::maxChamberId; chamber++ )
      chambers.push_back(chamber);
    
    LogTrace(metname) << "Encap =  " << endcap
		      << "Chambers =  " << chambers.size()
		      << "Rolls =  " << rolls.size();
    MuRingForwardLayer* ringLayer = buildLayer(endcap, layer, chambers, rolls, geo);          

    if (ringLayer) result[iendcap].push_back(ringLayer);
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]); 

  return res_pair;

}

MuRingForwardLayer* 
MuonME0DetLayerGeometryBuilder::buildLayer(int endcap,
					   int layer,
					   vector<int>& chambers,
					   vector<int>& rolls,
					   const ME0Geometry& geo) {

  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonME0DetLayerGeometryBuilder";
  MuRingForwardLayer * result = 0;
  vector<const ForwardDetRing*> frontRings, backRings;

  LogTrace(metname) << "Starting to Build Layer ";
  
  for (vector<int>::iterator roll = rolls.begin(); roll!=rolls.end(); roll++) {    
    LogTrace(metname) << "On a roll ";
    
    vector<const GeomDet*> frontDets, backDets;
      
    for(std::vector<int>::iterator chamber=chambers.begin(); chamber<chambers.end(); chamber++) {
      ME0DetId me0Id(endcap,layer,(*chamber), (*roll));
      const GeomDet* geomDet = geo.idToDet(me0Id);
	  
      if (geomDet !=0) {
	bool isInFront = isFront(me0Id);
	if(isInFront)
	  {
	    frontDets.push_back(geomDet);
	  }
	else 
	  {
	    backDets.push_back(geomDet);
	  }
      }
    }

    if (frontDets.size()!=0) {
      precomputed_value_sort(frontDets.begin(), frontDets.end(), geomsort::DetPhi());
      frontRings.push_back(new MuDetRing(frontDets));
      LogTrace(metname) << "New front ring with " << frontDets.size()
			<< " chambers at z="<< frontRings.back()->position().z();
    }
    if (backDets.size()!=0) {
      precomputed_value_sort(backDets.begin(), backDets.end(), geomsort::DetPhi());
      backRings.push_back(new MuDetRing(backDets));
      LogTrace(metname) << "New back ring with " << backDets.size()
			<< " chambers at z="<< backRings.back()->position().z();
    }

  }
  
  LogTrace(metname) << "About to make a MuRingForwardLayer";
  if(frontRings.size()!=0) result = new MuRingForwardLayer(frontRings);
  else result = 0;
  if(result != 0){
    LogTrace(metname) << "New MuRingForwardLayer with " << frontRings.size()
	 << " and " << backRings.size()
	 << " rings, at Z " << result->position().z()
	 << " R1: " << result->specificSurface().innerRadius()
	 << " R2: " << result->specificSurface().outerRadius();
  }
  return result;

}


bool MuonME0DetLayerGeometryBuilder::isFront(const ME0DetId & me0Id)
{

  //ME0s do not currently have an arrangement of which are front and which are back, going to always return true

  bool result = true;
  return result;
}

MuDetRing * MuonME0DetLayerGeometryBuilder::makeDetRing(vector<const GeomDet*> & geomDets)
{
    const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonME0DetLayerGeometryBuilder";


    precomputed_value_sort(geomDets.begin(), geomDets.end(), geomsort::DetPhi());
    MuDetRing * result = new MuDetRing(geomDets);
    LogTrace(metname) << "New MuDetRing with " << geomDets.size()
                        << " chambers at z="<< result->position().z()
                        << " R1: " << result->specificSurface().innerRadius()
                        << " R2: " << result->specificSurface().outerRadius();
    return result;
}
