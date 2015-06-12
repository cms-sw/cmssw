#include <RecoMuon/DetLayers/src/MuonME0DetLayerGeometryBuilder.h>

#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
//#include <RecoMuon/DetLayers/interface/MuRingForwardDoubleLayer.h>
#include <RecoMuon/DetLayers/interface/MuRingForwardLayer.h>
#include <RecoMuon/DetLayers/interface/MuRodBarrelLayer.h>
#include <RecoMuon/DetLayers/interface/MuDetRing.h>
#include <RecoMuon/DetLayers/interface/MuDetRod.h>

#include <Utilities/General/interface/precomputed_value_sort.h>
#include <Geometry/CommonDetUnit/interface/DetSorting.h>
#include "Utilities/BinningTools/interface/ClusterizingHistogram.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include <iostream>

using namespace std;

MuonME0DetLayerGeometryBuilder::~MuonME0DetLayerGeometryBuilder() {
}


// Builds the forward (first) and backward (second) layers - NOTE: Currently just one layer, all 'front'
pair<vector<DetLayer*>, vector<DetLayer*> > 
MuonME0DetLayerGeometryBuilder::buildEndcapLayers(const ME0Geometry& geo) {
  
  vector<DetLayer*> result[2];
  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonME0DetLayerGeometryBuilder";
  LogTrace(metname) << "Starting endcaplayers ";

  for (int endcap = -1; endcap<=1; endcap+=2) {
    int iendcap = (endcap==1) ? 0 : 1; // +1: forward, -1: backward

    //No stations defined for ME0
    //for(int station = ME0DetId::minStationId; station < ME0DetId::maxStationId; ++station) {

    //Changing to not be dependent on layers...
    vector<int> rolls;      
    std::vector<int> rings;
    std::vector<int> chambers;
	
      // for(int layer = ME0DetId::minLayerId; layer <= ME0DetId::maxLayerId; ++layer) { 
      // 	vector<int> rolls;      
      // 	std::vector<int> rings;
      // 	std::vector<int> chambers;
	
	//No rings defined for ME0
	// for(int ring = ME0DetId::minRingId; ring <= ME0DetId::maxRingId; ++ring) {
	//   rings.push_back(ring);
	// }

	for(int roll = ME0DetId::minRollId+1; roll <= ME0DetId::maxRollId; ++roll) {
	  rolls.push_back(roll);
	}
	for(int chamber = ME0DetId::minChamberId+1; chamber <= ME0DetId::maxChamberId; chamber++ ){
	  chambers.push_back(chamber);
	}

	LogTrace(metname) << "Encap =  " << endcap
	  //<< "Layer =  " << layer
			  << "Chambers =  " << chambers.size()
			  << "Rolls =  " << rolls.size();


	//MuRingForwardDoubleLayer* ringLayer = buildLayer(endcap, rings, station, layer, chambers, rolls, geo);          
	//MuRingForwardDoubleLayer* ringLayer = buildLayer(endcap, layer, chambers, rolls, geo);          
	//MuRingForwardLayer* ringLayer = buildLayer(endcap, layer, chambers, rolls, geo);          
	MuRingForwardLayer* ringLayer = buildLayer(endcap, chambers, rolls, geo);          

	if (ringLayer) result[iendcap].push_back(ringLayer);

      // }

      //}
  }
  pair<vector<DetLayer*>, vector<DetLayer*> > res_pair(result[0], result[1]); 

  return res_pair;

}




// MuonME0DetLayerGeometryBuilder::buildLayer(int endcap,vector<int>& rings, int station,
// 					   int layer,
// 					   vector<int>& chambers,
// 					   vector<int>& rolls,
// 					   const ME0Geometry& geo) {

//MuRingForwardDoubleLayer* 


// MuRingForwardLayer* 
// MuonME0DetLayerGeometryBuilder::buildLayer(int endcap, int layer,
// 					   vector<int>& chambers,
// 					   vector<int>& rolls,
// 					   const ME0Geometry& geo) {

MuRingForwardLayer* 
MuonME0DetLayerGeometryBuilder::buildLayer(int endcap,
					   vector<int>& chambers,
					   vector<int>& rolls,
					   const ME0Geometry& geo) {

  const std::string metname = "Muon|RecoMuon|RecoMuonDetLayers|MuonME0DetLayerGeometryBuilder";
  //MuRingForwardDoubleLayer * result = 0;
  MuRingForwardLayer * result = 0;
  vector<const ForwardDetRing*> frontRings, backRings;

  LogTrace(metname) << "Starting to Build Layer ";
//  for (std::vector<int>::iterator ring=rings.begin(); ring!=rings.end();ring++){ 

  //No rings defined in ME0
  //for (std::vector<int>::iterator ring=rings.begin(); ring!=rings.end()-2;ring++){ 

    for (vector<int>::iterator roll = rolls.begin(); roll!=rolls.end(); roll++) {    
      LogTrace(metname) << "On a roll ";
      vector<const GeomDet*> frontDets, backDets;

      for(std::vector<int>::iterator chamber=chambers.begin(); chamber<chambers.end(); chamber++) {
	//for(std::vector<int>::iterator chamber=chambers.begin(); chamber<chambers.end(); chamber++) {

	//LogTrace(metname) << "On a chamber ";
	//No rings or stations in ME0
	//ME0DetId me0Id(endcap,(*ring), station,layer,(*chamber), (*roll));
      //ME0DetId me0Id(endcap,layer,(*chamber), (*roll));
	//for (int i =1; i<=6;++i){

	  //ME0DetId me0Id(endcap,1,(*chamber), (*roll));
	//ME0DetId me0Id(endcap,0,(*chamber), (*roll));
	//ME0DetId me0Id(endcap,0,(*chamber), (*roll));


	//ME0DetId me0Id(endcap,1,(*chamber), (*roll));
	ME0DetId me0Id(endcap,1,(*chamber), 0);

      LogTrace(metname) << "Have an me0Id :"
			<<me0Id;


 	  const GeomDet* geomDet = geo.idToDet(me0Id);
	  LogTrace(metname) << "Got Geometry with id: "
			    <<geomDet;
	  
	  if (geomDet !=0) {
	  LogTrace(metname) << "In Geo step";
	    bool isInFront = isFront(me0Id);
	  LogTrace(metname) << "Getting bool";
	    if(isInFront)
            {
              frontDets.push_back(geomDet);
	  LogTrace(metname) << "Is in front";
            }
            else 
            {
              backDets.push_back(geomDet);
            }
	    LogTrace(metname) << "get ME0 Endcap roll "
			      << me0Id
                              << (isInFront ? "front" : "back ")
			      << " at R=" << geomDet->position().perp()
			      << ", phi=" << geomDet->position().phi()
                              << ", Z=" << geomDet->position().z();
	  }
	  //}
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

    //}

  // How should they be sorted?
  //    precomputed_value_sort(muDetRods.begin(), muDetRods.end(), geomsort::ExtractZ<GeometricSearchDet,float>());                                   


    //if(backRings.size()!=0 && frontRings.size()!=0) result = new MuRingForwardDoubleLayer(frontRings, backRings);
    
    //Using a single layer now, did a hack where we'll only fill the frontRings vector
    LogTrace(metname) << "About to make a MuRingForwardLayer";
    result = new MuRingForwardLayer(frontRings);
    
  //result = 0;
  
  LogTrace(metname) << "New MuRingForwardLayer with " << frontRings.size()
                    << " and " << backRings.size()
                    << " rings, at Z " << result->position().z()
                    << " R1: " << result->specificSurface().innerRadius()
                    << " R2: " << result->specificSurface().outerRadius();
  
  return result;

}


bool MuonME0DetLayerGeometryBuilder::isFront(const ME0DetId & me0Id)
{

  //ME0s do not currently have an arrangement of which are front and which are back, going to always return true

  bool result = true;
  return result;
 
  // bool result = false;
  // int chamber = me0Id.chamber();
  // if(chamber%2 == 0) result = !result;
  // return result;

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
