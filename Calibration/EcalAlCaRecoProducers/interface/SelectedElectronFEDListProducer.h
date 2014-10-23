#ifndef SelectedElectronFEDListProducer_h
#define SelectedFEDListProducer_h

#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <fstream>
#include "TLorentzVector.h"
#include "TVector3.h"
#include <ostream>
#include <memory>
#include <stdint.h>

#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDHeader.h"
#include "DataFormats/FEDRawData/interface/FEDTrailer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "DataFormats/EgammaReco/interface/PreshowerClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingMap.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCabling.h"
#include "CondFormats/SiPixelObjects/interface/CablingPathToDetUnit.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"
#include "CondFormats/SiPixelObjects/interface/GlobalPixel.h"
#include "CondFormats/SiPixelObjects/interface/LocalPixel.h"
#include "CondFormats/SiPixelObjects/interface/ElectronicIndex.h"
#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/DataRecord/interface/SiPixelFedCablingMapRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFedCablingTree.h"


using namespace std;


// Pixel region class
class PixelRegion {  
     public:     
      PixelRegion(math::XYZVector & momentum, float dphi = 0.5, float deta = 0.5, float maxz = 24.0){
       vector = momentum;
       dPhi = dphi ;
       dEta = deta ;
       maxZ = maxz ; 
       cosphi = vector.x()/vector.rho();
       sinphi = vector.y()/vector.rho(); 
       atantheta = vector.z()/vector.rho();
      }

     math::XYZVector vector;
     float dPhi,dEta,maxZ;
     float cosphi, sinphi, atantheta;
};

// Pixel module class
class PixelModule{
    public:

      PixelModule() {}
      PixelModule(float phi, float eta) : Phi(phi), Eta(eta), x(0.), y(0.), z(0.), DetId(0), Fed(0) {}
      bool operator < (const PixelModule& m) const {
        if(Phi < m.Phi) return true;
        if(Phi == m.Phi && Eta < m.Eta) return true;
        if(Phi == m.Phi && Eta == m.Eta && DetId < m.DetId) return true;
        return false;
      }

      float Phi,Eta;
      float x, y, z;
      unsigned int DetId;
      unsigned int Fed;

};


// HCAL Fed ID
class HCALFedId {
  public: 
   HCALFedId(){}
   virtual ~HCALFedId(){}
   HCALFedId( const int & subdet, const int & iphi, const int & ieta, const int & depth){
       subdet_ = subdet;  
       iphi_   = iphi;
       ieta_   = ieta;
       depth_  = depth;
   }

   bool operator < (const HCALFedId& m) const {
        if(subdet_ <  m.subdet_) return true;
        if(subdet_ == m.subdet_ && iphi_ < m.iphi_) return true;
        if(subdet_ == m.subdet_ && iphi_ == m.iphi_ && ieta_ < m.ieta_) return true;
        if(subdet_ == m.subdet_ && iphi_ == m.iphi_ && ieta_ == m.ieta_ && depth_ < m.depth_) return true;
        return false;
   }

   bool operator < (HCALFedId* m) const {
        if(subdet_ <  m->subdet_) return true;
        if(subdet_ == m->subdet_ && iphi_ < m->iphi_) return true;
        if(subdet_ == m->subdet_ && iphi_ == m->iphi_ && ieta_ < m->ieta_) return true;
        if(subdet_ == m->subdet_ && iphi_ == m->iphi_ && ieta_ == m->ieta_ && depth_ < m->depth_) return true;
        return false;
   }

   bool operator == (const HCALFedId & j1) const {
     if((*this).subdet_ ==  j1.subdet_ && (*this).iphi_ == j1.iphi_ && (*this).ieta_ == j1.ieta_ && (*this).depth_ == j1.depth_ ) return true ;
     return false ;        
   }

   bool operator == (HCALFedId* j1) const {
     if((*this).subdet_ ==  j1->subdet_ && (*this).iphi_ == j1->iphi_ && (*this).ieta_ == j1->ieta_ && (*this).depth_ == j1->depth_ ) return true ;
     return false ;        
   }

   void setDCCId(const int & dcc){ 
     dcc_ = dcc;
     fed_ = dcc+700 ;
   }

   int getFed(){
     return (*this).fed_ ;
   }  

   int iphi_, ieta_, depth_, dcc_, fed_, subdet_ ;  
};


// main class
template<typename TEle, typename TCand>
class SelectedElectronFEDListProducer : public edm::EDProducer {

 public:

   explicit SelectedElectronFEDListProducer( const edm::ParameterSet &);
   virtual ~SelectedElectronFEDListProducer();

 protected:

  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:

  typedef std::vector<TEle>  TEleColl ;
  typedef std::vector<TCand> TCandColl ;

 public:

 void pixelFedDump( std::vector<PixelModule>::const_iterator & itDn,  
                    std::vector<PixelModule>::const_iterator & itUp,
                    const PixelRegion & region);

 private:

  // input parameter of the producer
  std::vector<edm::InputTag> recoEcalCandidateCollections_ ;
  std::vector<edm::InputTag> electronCollections_ ;
  edm::InputTag   beamSpotTag_ ;
  edm::InputTag   rawDataLabel_ ;
  edm::InputTag   HBHERecHitCollection_;

  std::vector<int> isGsfElectronCollection_ ;
  std::vector<int> addThisSelectedFEDs_ ;

  math::XYZVector beamSpotPosition_;

  edm::FileInPath ESLookupTable_ ; 
  edm::FileInPath HCALLookupTable_ ; 

  bool dumpSelectedEcalFed_ ;
  bool dumpSelectedSiStripFed_ ;
  bool dumpSelectedSiPixelFed_ ;
  bool dumpSelectedHCALFed_;
  bool dumpAllEcalFed_ ;
  bool dumpAllTrackerFed_;
  bool dumpAllHCALFed_;

  bool debug_ ;

  double dRStripRegion_  ;
  double dPhiPixelRegion_;
  double dEtaPixelRegion_;
  double maxZPixelRegion_;
  double dRHcalRegion_;

  uint32_t eventCounter_ ;

  std::string outputLabelModule_ ;

  // internal info of geometry of each sub-detector
  int ES_fedId_[2][2][40][40];
  std::vector<HCALFedId> HCAL_fedId_ ;
  const static int HBHERecHitShift_ = 9 ;

  // fed list and output raw data
  std::vector<uint32_t> fedList_ ;
  FEDRawDataCollection* RawDataCollection_ ;

  const EcalElectronicsMapping* TheMapping_ ;
  const CaloGeometry* geometry_ ;
  const CaloSubdetectorGeometry *geometryES_ ;

  std::unique_ptr<SiPixelFedCablingTree> PixelCabling_;
  std::vector<PixelModule> pixelModuleVector_ ;

  const SiStripRegionCabling* StripRegionCabling_;
  SiStripRegionCabling::Cabling cabling_ ;
  std::pair<double,double> regionDimension_ ;
};

#endif
