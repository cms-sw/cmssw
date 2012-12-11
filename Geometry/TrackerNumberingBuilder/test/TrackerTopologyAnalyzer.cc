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
  unsigned int nComp=2;

  //result 0 is layer number
  for( DetIdContainer::const_iterator id = allIds.begin(), detUnitIdEnd = allIds.end(); id != detUnitIdEnd; ++id ) {
    if ( id->det()==DetId::Tracker ) {
      unsigned int subdet=id->subdetId();

      for ( unsigned int i=0; i<nComp; i++) resultsOld[i]=0;
      for ( unsigned int i=0; i<nComp; i++) resultsNew[i]=0;

      resultsNew[0]=tTopo->layer(*id);
      resultsNew[1]=tTopo->module(*id);

      if (subdet == PixelSubdetector::PixelBarrel) {
	resultsOld[0] = PXBDetId(*id).layer();
	resultsOld[1] = PXBDetId(*id).module();
      }
      else if (subdet == PixelSubdetector::PixelEndcap) {
	resultsOld[0] = PXFDetId(*id).disk();
	resultsOld[1] = PXFDetId(*id).module();
      }
      else if (subdet == StripSubdetector::TIB) {
	resultsOld[0] = TIBDetId(*id).layer();
	resultsOld[1] = TIBDetId(*id).module();
      }
      else if (subdet == StripSubdetector::TID) {
	resultsOld[0] = TIDDetId(*id).wheel();
	resultsOld[1] = TIDDetId(*id).moduleNumber();
      }
      else if (subdet == StripSubdetector::TOB) {
	resultsOld[0] = TOBDetId(*id).layer();
	resultsOld[1] = TOBDetId(*id).module();
      }
      else if (subdet == StripSubdetector::TEC) {
	resultsOld[0] = TECDetId(*id).wheel();
	resultsOld[1] = TECDetId(*id).module();
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

