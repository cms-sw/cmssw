#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include <climits>

class TrackerTopologyAnalyzer : public edm::EDAnalyzer {
public:
  explicit TrackerTopologyAnalyzer( const edm::ParameterSet& ) {};
  ~TrackerTopologyAnalyzer() {};
  
  virtual void analyze( const edm::Event&, const edm::EventSetup& );
private:
  //
};

void TrackerTopologyAnalyzer::analyze( const edm::Event &iEvent, const edm::EventSetup& iSetup) {
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<IdealGeometryRecord>().get(tTopo);

  typedef std::vector<DetId>                 DetIdContainer;

  edm::ESHandle<TrackerGeometry> geo;
  iSetup.get<TrackerDigiGeometryRecord>().get(geo);

  DetIdContainer allIds=geo->detIds();
  unsigned int nOk=0;
  unsigned int resultsOld[100];
  unsigned int resultsNew[100];
  unsigned int nComp=14;

  //result 0 is layer number
  for( DetIdContainer::const_iterator id = allIds.begin(), detUnitIdEnd = allIds.end(); id != detUnitIdEnd; ++id ) {
    if ( id->det()==DetId::Tracker ) {
      unsigned int subdet=id->subdetId();

      for ( unsigned int i=0; i<nComp; i++) resultsOld[i]=0;
      for ( unsigned int i=0; i<nComp; i++) resultsNew[i]=0;

      resultsNew[0]=tTopo->layer(*id);
      resultsNew[1]=tTopo->module(*id);
      //[2] is ladder - which is pxb specific
      //or side for all the others
      //[3] is tobRod and tec/tib/tid order

      if (subdet == PixelSubdetector::PixelBarrel) {
	resultsOld[0] = tTopo->pxbLayer(*id);
	resultsOld[1] = tTopo->pxbModule(*id);
	resultsNew[2] = tTopo->pxbLadder(*id);
	resultsNew[3] = 0;
	resultsOld[2] = tTopo->pxbLadder(*id);
      }
      else if (subdet == PixelSubdetector::PixelEndcap) {
	resultsOld[0] = tTopo->pxfDisk(*id);
	resultsOld[1] = tTopo->pxfModule(*id);
	resultsNew[2] = tTopo->pxfSide(*id);
	resultsNew[3] = 0;
	resultsNew[4] = tTopo->pxfDisk(*id);
	resultsNew[5] = tTopo->pxfBlade(*id);
	resultsNew[6] = tTopo->pxfPanel(*id);
	resultsOld[2] = tTopo->pxfSide(*id);
	resultsOld[4] = tTopo->pxfDisk(*id);
	resultsOld[5] = tTopo->pxfBlade(*id);
	resultsOld[6] = tTopo->pxfPanel(*id);
      }
      else if (subdet == StripSubdetector::TIB) {
	resultsOld[0] = tTopo->tibLayer(*id);
	resultsOld[1] = tTopo->tibModule(*id);
	resultsNew[2] = tTopo->tibSide(*id);
	resultsNew[3] = tTopo->tibOrder(*id);
	resultsNew[4] = 0;
	resultsNew[5] = 0;
	resultsNew[6] = tTopo->tibIsDoubleSide(*id);
	resultsNew[7] = tTopo->tibIsRPhi(*id);
	resultsNew[8] = tTopo->tibIsStereo(*id);
	resultsNew[9] = tTopo->tibIsZPlusSide(*id);
	resultsNew[10] = tTopo->tibIsZMinusSide(*id);
	resultsNew[11] = tTopo->tibString(*id);
	resultsNew[12] = tTopo->tibIsInternalString(*id);
	resultsNew[13] = tTopo->tibIsExternalString(*id);
	resultsOld[2] = tTopo->tibSide(*id);
	resultsOld[3] = tTopo->tibOrder(*id);
	resultsOld[6] = tTopo->tibIsDoubleSide(*id);
	resultsOld[7] = tTopo->tibIsRPhi(*id);
	resultsOld[8] = tTopo->tibIsStereo(*id);
	resultsOld[9] = tTopo->tibIsZPlusSide(*id);
	resultsOld[10] = tTopo->tibIsZMinusSide(*id);
	resultsOld[11] = tTopo->tibString(*id);
	resultsOld[12] = tTopo->tibIsInternalString(*id);
	resultsOld[13] = tTopo->tibIsExternalString(*id);

      }
      else if (subdet == StripSubdetector::TID) {
	resultsOld[0] = tTopo->tidWheel(*id);
	resultsOld[1] = tTopo->tidModule(*id);
	resultsNew[2] = tTopo->tidSide(*id);
	resultsNew[3] = tTopo->tidOrder(*id);
	resultsNew[4] = tTopo->tidRing(*id);
	resultsNew[5] = 0;
	resultsNew[6] = tTopo->tidIsDoubleSide(*id);
	resultsNew[7] = tTopo->tidIsRPhi(*id);
	resultsNew[8] = tTopo->tidIsStereo(*id);
	resultsNew[9] = tTopo->tidIsZPlusSide(*id);
	resultsNew[10] = tTopo->tidIsZMinusSide(*id);
	resultsNew[11] = tTopo->tidIsBackRing(*id);
	resultsNew[12] = tTopo->tidIsFrontRing(*id);
	resultsOld[2] = tTopo->tidSide(*id);
	resultsOld[3] = tTopo->tidOrder(*id);
	resultsOld[4] = tTopo->tidRing(*id);
	resultsOld[6] = tTopo->tidIsDoubleSide(*id);
	resultsOld[7] = tTopo->tidIsRPhi(*id);
	resultsOld[8] = tTopo->tidIsStereo(*id);
	resultsOld[9] = tTopo->tidIsZPlusSide(*id);
	resultsOld[10] = tTopo->tidIsZMinusSide(*id);
	resultsOld[11] = tTopo->tidIsBackRing(*id);
	resultsOld[12] = tTopo->tidIsFrontRing(*id);
      }
      else if (subdet == StripSubdetector::TOB) {
	resultsOld[0] = tTopo->tobLayer(*id);
	resultsOld[1] = tTopo->tobModule(*id);
	resultsNew[2] = tTopo->tobSide(*id);
	resultsNew[3] = tTopo->tobRod(*id);
	resultsNew[4] = 0;
	resultsNew[5] = 0;
	resultsNew[6] = tTopo->tobIsDoubleSide(*id);
	resultsNew[7] = tTopo->tobIsRPhi(*id);
	resultsNew[8] = tTopo->tobIsStereo(*id);
	resultsNew[9] = tTopo->tobIsZPlusSide(*id);
	resultsNew[10] = tTopo->tobIsZMinusSide(*id);
	resultsOld[2] = tTopo->tobSide(*id);
	resultsOld[3] = tTopo->tobRod(*id);
	resultsOld[6] = tTopo->tobIsDoubleSide(*id);
	resultsOld[7] = tTopo->tobIsRPhi(*id);
	resultsOld[8] = tTopo->tobIsStereo(*id);
	resultsOld[9] = tTopo->tobIsZPlusSide(*id);
	resultsOld[10] = tTopo->tobIsZMinusSide(*id);
      }
      else if (subdet == StripSubdetector::TEC) {
	resultsOld[0] = tTopo->tecWheel(*id);
	resultsOld[1] = tTopo->tecModule(*id);
	resultsNew[2] = tTopo->tecSide(*id);
	resultsNew[3] = tTopo->tecOrder(*id);
	resultsNew[4] = tTopo->tecRing(*id);
	resultsNew[5] = tTopo->tecPetalNumber(*id);
	resultsNew[6] = tTopo->tecIsDoubleSide(*id);
	resultsNew[7] = tTopo->tecIsRPhi(*id);
	resultsNew[8] = tTopo->tecIsStereo(*id);
	resultsNew[9] = tTopo->tecIsBackPetal(*id);
	resultsNew[10] = tTopo->tecIsFrontPetal(*id);
	resultsOld[2] = tTopo->tecSide(*id);
	resultsOld[3] = tTopo->tecOrder(*id);
	resultsOld[4] = tTopo->tecRing(*id);
	resultsOld[5] = tTopo->tecPetalNumber(*id);
	resultsOld[6] = tTopo->tecIsDoubleSide(*id);
	resultsOld[7] = tTopo->tecIsRPhi(*id);
	resultsOld[8] = tTopo->tecIsStereo(*id);
	resultsOld[9] = tTopo->tecIsBackPetal(*id);
	resultsOld[10] = tTopo->tecIsFrontPetal(*id);
      }

      bool isGood=true;
      for ( unsigned int i=0; i<nComp; i++)
	if ( resultsOld[i]!=resultsNew[i])
	  isGood=false;

      if (isGood) 
	nOk+=1;
      else {
	std::cout << "Bad " << id->rawId() << " " << id->subdetId() << " ";
	for ( unsigned int i=0; i<nComp; i++)
	  std::cout << "["<<resultsOld[i]<<","<<resultsNew[i]<< "] " <<std::endl;
      }
    }
  }
  std::cout << "Good: " << nOk << std::endl;

}

DEFINE_FWK_MODULE(TrackerTopologyAnalyzer);

