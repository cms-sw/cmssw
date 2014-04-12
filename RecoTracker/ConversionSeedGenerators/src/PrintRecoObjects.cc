#include "RecoTracker/ConversionSeedGenerators/interface/PrintRecoObjects.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

void PrintRecoObjects::
print(std::stringstream& ss, const SiStripCluster& clus){
  ss << "\n\t"
     << " firstStrip " <<  clus.firstStrip()
     << " size " <<  clus.amplitudes().size();
  float charge=0;
  size_t idx=0;
  BOOST_FOREACH(const uint8_t& adc,clus.amplitudes()){
    charge+=adc;
    ss << "\n\t\t strip " << ++idx << " adc " << (size_t) adc;
  }
  ss << "\n\t\t charge " << charge
     << " barycenter " << clus.barycenter()
     << std::endl;
}

void 
PrintRecoObjects::print(std::stringstream& ss, const TrajectorySeed& tjS){

  ss << "\n\t nHits "<< tjS.nHits() 
     << "\n\t PTrajectoryStateOnDet: detid " << tjS.startingState().detId()
     << " tsos momentum" << tjS.startingState().parameters().momentum()
     << " pt " << sqrt(tjS.startingState().parameters().momentum().mag2()-tjS.startingState().parameters().momentum().y()*tjS.startingState().parameters().momentum().y())
     << " charge " << tjS.startingState().parameters().charge()
     << "\n\t error ";
  for(size_t ie=0;ie<15;++ie)
    ss << "\t " << tjS.startingState().error(ie);
  for(TrajectorySeed::const_iterator iter=tjS.recHits().first;iter!=tjS.recHits().second;++iter)
    ss << "\n\t TrackingRecHit on detid " << iter->geographicalId().rawId() << " \t localPos " << iter->localPosition();
  
}

void PrintRecoObjects::
print(std::stringstream& ss, const uint32_t& detid) const{
  ss<< getString(detid);
}

std::string PrintRecoObjects::
getString(uint32_t detid) const{
  std::string append=" ";
  char cindex[128];
  SiStripDetId a(detid);
  if ( a.subdetId() == 3 ){
    append+="_TIB_L";
    sprintf(cindex,"%d",TIBDetId(detid).layer()); 
  } else if ( a.subdetId() == 4 ) {
    if(TIDDetId(detid).side()==1){
      append+="_M_D";
    }else{
      append+="_M_P";
    }
    sprintf(cindex,"%d",TIDDetId(detid).wheel());
  } else if ( a.subdetId() == 5 ) {
    append+="_TOB_L";
    sprintf(cindex,"%d",TOBDetId(detid).layer());
  } else if ( a.subdetId() == 6 ) {
    if(TECDetId(detid).side()==1){
      append+="_TEC_M";
    }else{
      append+="_TEC_P";
    }
    sprintf(cindex,"%d",TECDetId(detid).wheel());
  } 

  append+=std::string(cindex);
  return append;  
}

void PrintRecoObjects::
print(std::stringstream& ss, const reco::Track* track, const math::XYZPoint& vx){
  
  ss
    << "[PrintObject] " 
    << "\n\tcharge \t"          << track->charge()
    << "\talgo \t"              << track->algo() 
    << "\n\treferencePoint \t"  << track->referencePoint() << " \t r: " << track->referencePoint().rho()
    << "\n\tinnerpos \t"          << track->innerPosition() << " \t r: " << track->innerPosition().rho();

 
  ss
    << "\n\tmomentum \t"         << track->momentum()       << " pt " << track->momentum().rho()
    << "\n\tinnerMom \t"        << track->innerMomentum()  << " pt " << track->innerMomentum().rho();
  

  ss
    << "\n\tinnerok \t"         << track->innerOk()
    << "\n\tinnerdetid \t"      << track->innerDetId()
    << "\n\tdxy    \t"             << track->dxy() 
    << "\n\tdxy(vx) \t"         << track->dxy(vx)
    << "\t where vx \t"         << vx
    << std::endl;

}
