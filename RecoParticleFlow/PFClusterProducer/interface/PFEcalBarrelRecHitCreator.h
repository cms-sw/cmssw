#ifndef RecoParticleFlow_PFClusterProducer_PFEcalBarrelRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFEcalBarrelRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloTopology/interface/EcalPreshowerTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

template <typename Geometry,PFLayer::Layer Layer,int Detector>
  class PFEcalBarrelRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFEcalBarrelRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
      srFlagToken_ = iC.consumes<EBSrFlagCollection>(iConfig.getParameter<edm::InputTag>("srFlags"));
      SREtaSize_ = iConfig.getUntrackedParameter<int> ("SREtaSize",1);
      SRPhiSize_ = iConfig.getUntrackedParameter<int> ("SRPhiSize",1);
      crystalsinTT_.resize(2448);
      TTHighInterest_.resize(2448,0);
      theTTDetIds_.resize(2448);
      neighboringTTs_.resize(2448);
    }
    
    void importRecHits(std::unique_ptr<reco::PFRecHitCollection>&out,std::unique_ptr<reco::PFRecHitCollection>& cleaned ,const edm::Event& iEvent,const edm::EventSetup& iSetup) {

      beginEvent(iEvent,iSetup);
      
      edm::Handle<EcalRecHitCollection> recHitHandle;

      edm::ESHandle<CaloGeometry> geoHandle;
      iSetup.get<CaloGeometryRecord>().get(geoHandle);
  
      iEvent.getByToken(srFlagToken_,srFlagHandle_);

      // get the ecal geometry
      const CaloSubdetectorGeometry *gTmp = 
	geoHandle->getSubdetectorGeometry(DetId::Ecal, Detector);

      const Geometry *ecalGeo =dynamic_cast< const Geometry* > (gTmp);

      iEvent.getByToken(recHitToken_,recHitHandle);
      for(const auto& erh : *recHitHandle ) {      
	const DetId& detid = erh.detid();
	auto energy = erh.energy();
	auto time = erh.time();
        bool hi = isHighInterest(detid);

	const CaloCellGeometry * thisCell= ecalGeo->getGeometry(detid);
  
	// find rechit geometry
	if(!thisCell) {
	  edm::LogError("PFEcalBarrelRecHitCreator")
	    <<"warning detid "<<detid.rawId()
	    <<" not found in geometry"<<std::endl;
	  continue;
	}

	out->emplace_back(thisCell, detid.rawId(),Layer,
			   energy); 

        auto & rh = out->back();
	
	bool rcleaned = false;
	bool keep=true;

	//Apply Q tests
	for( const auto& qtest : qualityTests_ ) {
	  if (!qtest->test(rh,erh,rcleaned,hi)) {
	    keep = false;	    
	  }
	}
	  
	if(keep) {
	  rh.setTime(time);
	  rh.setDepth(1);
	} 
        else {
	  if (rcleaned) 
	    cleaned->push_back(std::move(out->back()));
          out->pop_back();
        }
      }
    }

    void init(const edm::EventSetup &es) {

      edm::ESHandle<CaloGeometry> pG;
      es.get<CaloGeometryRecord>().get(pG);   

      //  edm::ESHandle<CaloTopology> theCaloTopology;
      //  es.get<CaloTopologyRecord>().get(theCaloTopology);     

      edm::ESHandle<EcalTrigTowerConstituentsMap> hetm;
      es.get<IdealGeometryRecord>().get(hetm);
      eTTmap_ = &(*hetm);
  
      const EcalBarrelGeometry * myEcalBarrelGeometry = dynamic_cast<const EcalBarrelGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalBarrel));
      // std::cout << "Got the geometry " << myEcalBarrelGeometry << std::endl;
      const std::vector<DetId>& vec(myEcalBarrelGeometry->getValidDetIds(DetId::Ecal,EcalBarrel));
      unsigned size=vec.size();    
      for(unsigned ic=0; ic<size; ++ic) 
        {
          EBDetId myDetId(vec[ic]);
          int crystalHashedIndex=myDetId.hashedIndex();
          /// barrelRawId_[crystalHashedIndex]=vec[ic].rawId();
          // save the Trigger tower DetIds
          EcalTrigTowerDetId towid= eTTmap_->towerOf(EBDetId(vec[ic]));
          int TThashedindex=towid.hashedIndex();      
          theTTDetIds_[TThashedindex]=towid;                  
          crystalsinTT_[TThashedindex].push_back(crystalHashedIndex);
        }
      unsigned nTTs=theTTDetIds_.size();

      //  EBDetId myDetId(-58,203);
      ////  std::cout << " CellID " << myDetId << std::endl;
      //  EcalTrigTowerDetId towid= eTTmap_->towerOf(myDetId);
      ////  std::cout << " EcalTrigTowerDetId ieta, iphi" << towid.ieta() << " , " << towid.iphi() << std::endl;
      ////  std::cout << " Constituents of this tower " <<towid.hashedIndex() << std::endl;
      //  const std::vector<int> & xtals(crystalsinTT_[towid.hashedIndex()]);
      //  unsigned Size=xtals.size();
      //  for(unsigned i=0;i<Size;++i)
      //    {
      //      std::cout << EBDetId(barrelRawId_[xtals[i]]) << std::endl;
      //    }

      // now loop on each TT and save its neighbors. 

      for(unsigned iTT=0;iTT<nTTs;++iTT)
        {
          int ietaPivot=theTTDetIds_[iTT].ieta();
          int iphiPivot=theTTDetIds_[iTT].iphi();
          int TThashedIndex=theTTDetIds_[iTT].hashedIndex();
          //      std::cout << " TT Pivot " << TThashedIndex << " " << ietaPivot << " " << iphiPivot << " iz " << theTTDetIds_[iTT].zside() << std::endl;
          int ietamin=std::max(ietaPivot-SREtaSize_,-17);
          if(ietamin==0) ietamin=-1;
          int ietamax=std::min(ietaPivot+SREtaSize_,17);
          if(ietamax==0) ietamax=1;
          int iphimin=iphiPivot-SRPhiSize_;
          int iphimax=iphiPivot+SRPhiSize_;
          for(int ieta=ietamin;ieta<=ietamax;)
            {
              int iz=(ieta>0)? 1 : -1; 
              for(int iphi=iphimin;iphi<=iphimax;)
                {
                  int riphi=iphi;
                  if(riphi<1) riphi+=72;
                  else if(riphi>72) riphi-=72;
                  EcalTrigTowerDetId neighborTTDetId(iz,EcalBarrel,abs(ieta),riphi);
                  //      std::cout << " Voisin " << ieta << " " << riphi << " " <<neighborTTDetId.hashedIndex()<< " " << neighborTTDetId.ieta() << " " << neighborTTDetId.iphi() << std::endl;
                  if(ieta!=ietaPivot||riphi!=iphiPivot)
                    {
                      neighboringTTs_[TThashedIndex].push_back(neighborTTDetId.hashedIndex());
                    }
                  ++iphi;

                }
              ++ieta;
              if(ieta==0) ieta=1;
            }
        }

      // std::cout << "EB Made the array " << std::endl;

    }
      

 protected:


    bool isHighInterest(const EBDetId& detid) {

      EcalTrigTowerDetId towid = detid.tower();
      int tthi = towid.hashedIndex();

      if(TTHighInterest_[tthi]!=0) return (TTHighInterest_[tthi]>0);

      TTHighInterest_[tthi] = srFullReadOut(towid) ? 1:-1;

      // if high interest, can leave ; otherwise look at the neighbours
      if( TTHighInterest_[tthi]==1) {
        theTTofHighInterest_.push_back(tthi);
        return true;
      }

      // now look if a neighboring TT is of high interest
      const std::vector<int> & tts(neighboringTTs_[tthi]);
      // a tower is of high interest if it or one of its neighbour is above the SR threshold
      unsigned size=tts.size();
      bool result=false;
      for(unsigned itt=0;itt<size&&!result;++itt) {
        towid = theTTDetIds_[tts[itt]];
        if(srFullReadOut(towid)) result=true;
      }

      TTHighInterest_[tthi]=(result)? 1:-1;
      theTTofHighInterest_.push_back(tthi);
      return result;
    }

    bool srFullReadOut(EcalTrigTowerDetId& towid) {
      EBSrFlagCollection::const_iterator srf = srFlagHandle_->find(towid);
      if(srf == srFlagHandle_->end()) return false;
      return ((srf->value() & ~EcalSrFlag::SRF_FORCED_MASK) == EcalSrFlag::SRF_FULL);  
    }

    edm::EDGetTokenT<EcalRecHitCollection> recHitToken_;
    edm::EDGetTokenT<EBSrFlagCollection> srFlagToken_;

    const EcalTrigTowerConstituentsMap* eTTmap_;  
    // Array of the DetIds
    std::vector<EcalTrigTowerDetId> theTTDetIds_;
    // neighboring TT DetIds
    std::vector<std::vector<int> > neighboringTTs_;
    // the crystals in a given TT 
    std::vector<std::vector<int> > crystalsinTT_;
    // the towers which have been looked at 
    std::vector<int> theTTofHighInterest_;
    // the status of the towers. A tower is of high interest if it or one of its neighbour is above the threshold
    std::vector<int> TTHighInterest_;

    // selective readout flags collection
    edm::Handle<EBSrFlagCollection> srFlagHandle_;

    // selective readout threshold
    int SREtaSize_;
    int SRPhiSize_;

};


typedef PFEcalBarrelRecHitCreator<EcalBarrelGeometry,PFLayer::ECAL_BARREL,EcalBarrel> PFEBRecHitCreator;

#endif
