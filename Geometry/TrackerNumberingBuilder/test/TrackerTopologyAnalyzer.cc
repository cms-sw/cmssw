#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"

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
  
  typedef std::vector<DetId>                 DetIdContainer;

  edm::ESHandle<TrackerTopology> tTopo;
  iSetup.get<IdealGeometryRecord>().get(tTopo);

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
	resultsOld[0] = PXBDetId(*id).layer();
	resultsOld[1] = PXBDetId(*id).module();
	resultsNew[2] = tTopo->pxbLadder(*id);
	resultsNew[3] = 0;
	resultsOld[2] = PXBDetId(*id).ladder();
      }
      else if (subdet == PixelSubdetector::PixelEndcap) {
	resultsOld[0] = PXFDetId(*id).disk();
	resultsOld[1] = PXFDetId(*id).module();
	resultsNew[2] = tTopo->pxfSide(*id);
	resultsNew[3] = 0;
	resultsNew[4] = tTopo->pxfDisk(*id);
	resultsNew[5] = tTopo->pxfBlade(*id);
	resultsNew[6] = tTopo->pxfPanel(*id);
	resultsOld[2] = PXFDetId(*id).side();
	resultsOld[4] = PXFDetId(*id).disk();
	resultsOld[5] = PXFDetId(*id).blade();
	resultsOld[6] = PXFDetId(*id).panel();
      }
      else if (subdet == StripSubdetector::TIB) {
	resultsOld[0] = TIBDetId(*id).layer();
	resultsOld[1] = TIBDetId(*id).module();
	resultsNew[2] = tTopo->tibSide(*id);
	resultsNew[3] = tTopo->tibOrder(*id);
	resultsNew[4] = 0;
	resultsNew[5] = 0;
	resultsNew[6] = tTopo->tibIsDoubleSide(*id);
	resultsNew[7] = tTopo->tibIsRPhi(*id);
	resultsNew[8] = tTopo->tibIsStereo(*id);
	resultsNew[9] = tTopo->tibIsZPlusSide(*id);
	resultsNew[10] = tTopo->tibIsZMinusSide(*id);
	resultsNew[11] = tTopo->tibStringNumber(*id);
	resultsNew[12] = tTopo->tibIsInternalString(*id);
	resultsNew[13] = tTopo->tibIsExternalString(*id);
	resultsOld[2] = TIBDetId(*id).side();
	resultsOld[3] = TIBDetId(*id).order();
	resultsOld[6] = TIBDetId(*id).isDoubleSide();
	resultsOld[7] = TIBDetId(*id).isRPhi();
	resultsOld[8] = TIBDetId(*id).isStereo();
	resultsOld[9] = TIBDetId(*id).isZPlusSide();
	resultsOld[10] = TIBDetId(*id).isZMinusSide();
	resultsOld[11] = TIBDetId(*id).stringNumber();
	resultsOld[12] = TIBDetId(*id).isInternalString();
	resultsOld[13] = TIBDetId(*id).isExternalString();

      }
      else if (subdet == StripSubdetector::TID) {
	resultsOld[0] = TIDDetId(*id).wheel();
	resultsOld[1] = TIDDetId(*id).moduleNumber();
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
	resultsOld[2] = TIDDetId(*id).side();
	resultsOld[3] = TIDDetId(*id).order();
	resultsOld[4] = TIDDetId(*id).ring();
	resultsOld[6] = TIDDetId(*id).isDoubleSide();
	resultsOld[7] = TIDDetId(*id).isRPhi();
	resultsOld[8] = TIDDetId(*id).isStereo();
	resultsOld[9] = TIDDetId(*id).isZPlusSide();
	resultsOld[10] = TIDDetId(*id).isZMinusSide();
	resultsOld[11] = TIDDetId(*id).isBackRing();
	resultsOld[12] = TIDDetId(*id).isFrontRing();
      }
      else if (subdet == StripSubdetector::TOB) {
	resultsOld[0] = TOBDetId(*id).layer();
	resultsOld[1] = TOBDetId(*id).module();
	resultsNew[2] = tTopo->tobSide(*id);
	resultsNew[3] = tTopo->tobRod(*id);
	resultsNew[4] = 0;
	resultsNew[5] = 0;
	resultsNew[6] = tTopo->tobIsDoubleSide(*id);
	resultsNew[7] = tTopo->tobIsRPhi(*id);
	resultsNew[8] = tTopo->tobIsStereo(*id);
	resultsNew[9] = tTopo->tobIsZPlusSide(*id);
	resultsNew[10] = tTopo->tobIsZMinusSide(*id);
	resultsOld[2] = TOBDetId(*id).side();
	resultsOld[3] = TOBDetId(*id).rodNumber();
	resultsOld[6] = TOBDetId(*id).isDoubleSide();
	resultsOld[7] = TOBDetId(*id).isRPhi();
	resultsOld[8] = TOBDetId(*id).isStereo();
	resultsOld[9] = TOBDetId(*id).isZPlusSide();
	resultsOld[10] = TOBDetId(*id).isZMinusSide();
      }
      else if (subdet == StripSubdetector::TEC) {
	resultsOld[0] = TECDetId(*id).wheel();
	resultsOld[1] = TECDetId(*id).module();
	resultsNew[2] = tTopo->tecSide(*id);
	resultsNew[3] = tTopo->tecOrder(*id);
	resultsNew[4] = tTopo->tecRing(*id);
	resultsNew[5] = tTopo->tecPetalNumber(*id);
	resultsNew[6] = tTopo->tecIsDoubleSide(*id);
	resultsNew[7] = tTopo->tecIsRPhi(*id);
	resultsNew[8] = tTopo->tecIsStereo(*id);
	resultsNew[9] = tTopo->tecIsBackPetal(*id);
	resultsNew[10] = tTopo->tecIsFrontPetal(*id);
	resultsOld[2] = TECDetId(*id).side();
	resultsOld[3] = TECDetId(*id).order();
	resultsOld[4] = TECDetId(*id).ring();
	resultsOld[5] = TECDetId(*id).petalNumber();
	resultsOld[6] = TECDetId(*id).isDoubleSide();
	resultsOld[7] = TECDetId(*id).isRPhi();
	resultsOld[8] = TECDetId(*id).isStereo();
	resultsOld[9] = TECDetId(*id).isBackPetal();
	resultsOld[10] = TECDetId(*id).isFrontPetal();
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

