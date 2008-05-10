#include "DataFormats/TrackReco/interface/TrajectorySateOnDetInfo.h"

using namespace reco;
using namespace std;

TrajectorySateOnDetInfo::TrajectorySateOnDetInfo(const LocalTrajectoryParameters theLocalParameters, std::vector<float> theLocalErrors, const ClusterRef theCluster)
{
   _theLocalParameters = theLocalParameters;
   _theLocalErrors     = theLocalErrors;
   _theCluster         = theCluster;
}

unsigned int TrajectorySateOnDetInfo::charge(){
   const vector<uint16_t>& Ampls       = _theCluster->amplitudes();
// const vector<uint8_t> &  Ampls      = _theCluster->amplitudes();
  
   unsigned int charge=0;
   for(unsigned int a=0;a<Ampls.size();a++){charge+=Ampls[a];}
   return charge;

}

double TrajectorySateOnDetInfo::thickness(edm::ESHandle<TrackerGeometry> tkGeom){
   const GeomDetUnit* it = tkGeom->idToDetUnit(DetId(_theCluster->geographicalId()));
   if (dynamic_cast<const StripGeomDetUnit*>(it)==0 && dynamic_cast<const PixelGeomDetUnit*>(it)==0) {
     std::cout << "this detID doesn't seem to belong to the Tracker" << std::endl;
     return -1;
   }

   return it->surface().bounds().thickness();
}

double TrajectorySateOnDetInfo::chargeOverPath(edm::ESHandle<TrackerGeometry> tkGeom){
   double Charge    = (double) charge();
   double Thickness = thickness(tkGeom); 
   double Cosine    = fabs(_theLocalParameters.momentum().z() /  _theLocalParameters.momentum().mag());
   return (Charge*Cosine)/(10.0*Thickness);
}

double TrajectorySateOnDetInfo::pathLength(edm::ESHandle<TrackerGeometry> tkGeom){
   double Thickness = thickness(tkGeom);
   double Cosine    = fabs(_theLocalParameters.momentum().z() /  _theLocalParameters.momentum().mag());
   return Cosine/(10.0*Thickness);
}



LocalVector TrajectorySateOnDetInfo::momentum(){
   return _theLocalParameters.momentum();
}

LocalPoint TrajectorySateOnDetInfo::point(){
   return _theLocalParameters.position();
}


