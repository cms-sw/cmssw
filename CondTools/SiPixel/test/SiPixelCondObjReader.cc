#include <memory>

#include "CondTools/SiPixel/test/SiPixelCondObjReader.h"
#include "CondFormats/DataRecord/interface/SiPixelGainCalibrationRcd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
namespace cms{
SiPixelCondObjReader::SiPixelCondObjReader(const edm::ParameterSet& iConfig)
{
}

void
SiPixelCondObjReader::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   unsigned int nmodules = 0;

  // Get the Geometry
  edm::ESHandle<TrackerGeometry> pDD;
  iSetup.get<TrackerDigiGeometryRecord>().get( pDD );     
  edm::LogInfo("SiPixelCondObjReader") <<" There are "<<pDD->dets().size() <<" detectors"<<std::endl;
  
  // Get the calibration data
  edm::ESHandle<SiPixelGainCalibration> SiPixelGainCalibration_;
  iSetup.get<SiPixelGainCalibrationRcd>().get(SiPixelGainCalibration_);
  edm::LogInfo("SiPixelCondObjReader") << "[SiPixelCondObjReader::analyze] End Reading CondObjects" << std::endl;

  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
     if( dynamic_cast<PixelGeomDetUnit*>((*it))!=0){
       uint32_t detid=((*it)->geographicalId()).rawId();
       nmodules++;
       if(nmodules>3) break;

       const PixelGeomDetUnit * pixDet  = dynamic_cast<const PixelGeomDetUnit*>((*it));
       const PixelTopology & topol = pixDet->specificTopology();       
       // Get the module sizes.
       int nrows = topol.nrows();      // rows in x
       int ncols = topol.ncolumns();   // cols in y
       std::cout << " ---> PIXEL DETID " << detid << " Cols " << ncols << " Rows " << nrows << std::endl;

       SiPixelGainCalibration::Range theRange = SiPixelGainCalibration_->getRange(detid);
       for(int col_iter=0; col_iter<ncols; col_iter++) {
	 for(int row_iter=0; row_iter<nrows; row_iter++) {

	   float ped  = SiPixelGainCalibration_->getPed (col_iter, row_iter, theRange, ncols);
	   float gain = SiPixelGainCalibration_->getGain(col_iter, row_iter, theRange, ncols);
	   std::cout << "       Col "<<col_iter<<" Row "<<row_iter<<" Ped "<<ped<<" Gain "<<gain<<std::endl;

	 }
       }
     }
  }

}

// ------------ method called once each job just before starting event loop  ------------
void 
SiPixelCondObjReader::beginJob(const edm::EventSetup&)
{
  std::cout<<" ---> Begin job "<<std::endl;
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiPixelCondObjReader::endJob() {
  std::cout<<" ---> End job "<<std::endl;
}
}
