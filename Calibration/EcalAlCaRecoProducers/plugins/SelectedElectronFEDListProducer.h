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

// common 
#include "DataFormats/Common/interface/Handle.h"
// egamma objects
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
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
// raw data
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
// detector id
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
// Math
#include "DataFormats/Math/interface/normalizedPhi.h"
// Hcal rec hit
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/CaloRecHit/interface/CaloRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
// Geometry
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// strip geometry
#include "CalibFormats/SiStripObjects/interface/SiStripRegionCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripRegionCablingRcd.h"
// FW core
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
// Message logger
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
// Strip and pixel
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

// Hcal objects
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/HcalObjects/interface/HcalElectronicsMap.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"

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


// main class
template<typename TEle, typename TCand>
class SelectedElectronFEDListProducer : public edm::EDProducer {

 public:

   explicit SelectedElectronFEDListProducer( const edm::ParameterSet &);
   virtual ~SelectedElectronFEDListProducer();

   static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);


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
  std::vector<edm::InputTag> recoEcalCandidateTags_ ;
  std::vector<edm::InputTag> electronTags_ ;
  edm::InputTag              beamSpotTag_ ;
  edm::InputTag              rawDataTag_ ;

  std::vector<int> isGsfElectronCollection_ ;
  std::vector<int> addThisSelectedFEDs_ ;

  edm::InputTag              HBHERecHitTag_;

  edm::FileInPath ESLookupTable_ ; 

  bool dumpSelectedEcalFed_ ;
  bool dumpSelectedSiStripFed_ ;
  bool dumpSelectedSiPixelFed_ ;
  bool dumpSelectedHCALFed_;
  bool dumpAllEcalFed_ ;
  bool dumpAllTrackerFed_;
  bool dumpAllHCALFed_;

  double dRStripRegion_  ;
  double dPhiPixelRegion_;
  double dEtaPixelRegion_;
  double maxZPixelRegion_;
  double dRHcalRegion_;

  std::string outputLabelModule_ ;

  // Token for the input collection
  edm::EDGetTokenT<FEDRawDataCollection>     rawDataToken_ ;
  edm::EDGetTokenT<reco::BeamSpot>           beamSpotToken_ ;
  edm::EDGetTokenT<HBHERecHitCollection> hbheRecHitToken_;
  std::vector<edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> > recoEcalCandidateToken_ ;
  std::vector<edm::EDGetTokenT<TEleColl> >   electronToken_; 

  // used inside the producer
  uint32_t eventCounter_ ;
  math::XYZVector beamSpotPosition_;

  // internal info for ES geometry
  int ES_fedId_[2][2][40][40];

  // fed list and output raw data
  std::vector<uint32_t> fedList_ ;

  // get the raw data
  FEDRawDataCollection* RawDataCollection_ ;
  // get calo geomentry and electronic map
  const EcalElectronicsMapping* TheMapping_ ;
  const CaloGeometry* geometry_ ;
  const CaloSubdetectorGeometry *geometryES_ ;

  // get pixel geometry and electronic map
  std::unique_ptr<SiPixelFedCablingTree> PixelCabling_;
  std::vector<PixelModule>               pixelModuleVector_ ;

  // get strip geometry and electronic map
  const SiStripRegionCabling*   StripRegionCabling_;
  SiStripRegionCabling::Cabling cabling_ ;
  std::pair<double,double>      regionDimension_ ;

  // get hcal geometry and electronic map
  const HcalElectronicsMap* hcalReadoutMap_;

};

#endif
