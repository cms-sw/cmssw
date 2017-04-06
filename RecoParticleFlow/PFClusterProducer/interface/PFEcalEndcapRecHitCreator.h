#ifndef RecoParticleFlow_PFClusterProducer_PFEcalEndcapRecHitCreator_h
#define RecoParticleFlow_PFClusterProducer_PFEcalEndcapRecHitCreator_h

#include "RecoParticleFlow/PFClusterProducer/interface/PFRecHitCreatorBase.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"


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
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

template <typename Geometry,PFLayer::Layer Layer,int Detector>
  class PFEcalEndcapRecHitCreator :  public  PFRecHitCreatorBase {

 public:  
  PFEcalEndcapRecHitCreator(const edm::ParameterSet& iConfig,edm::ConsumesCollector& iC):
    PFRecHitCreatorBase(iConfig,iC)
    {
      recHitToken_ = iC.consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("src"));
      srFlagToken_ = iC.consumes<EBSrFlagCollection>(iConfig.getParameter<edm::InputTag>("srFlags"));

      towerOf_.resize(EEDetId::kSizeForDenseIndexing);
      theTTDetIds_.resize(1440);
      SCofTT_.resize(1440);
      SCHighInterest_.resize(633,0);
      TTofSC_.resize(633);
      CrystalsinSC_.resize(633);
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
	  edm::LogError("PFEcalEndcapRecHitCreator")
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
  
      const EcalEndcapGeometry * myEcalEndcapGeometry = dynamic_cast<const EcalEndcapGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalEndcap));
      // std::cout << " Got the geometry " << myEcalEndcapGeometry << std::endl;
      const std::vector<DetId>& vec(myEcalEndcapGeometry->getValidDetIds(DetId::Ecal,EcalEndcap));
      unsigned size=vec.size();    
      for(unsigned ic=0; ic<size; ++ic) 
        {
          EEDetId myDetId(vec[ic]);
          int cellhashedindex=myDetId.hashedIndex();
          /// endcapRawId_[crystalHashedIndex]=vec[ic].rawId();
          // a bit of trigger tower and SuperCrystals algebra
          // first get the trigger tower 
          EcalTrigTowerDetId towid1= eTTmap_->towerOf(vec[ic]);
          int tthashedindex=TThashedIndexforEE(towid1);
          towerOf_[cellhashedindex]=tthashedindex;

          // get the SC of the cell
          int schi=SChashedIndex(EEDetId(vec[ic]));
          if(schi<0) 
            {
              //  std::cout << " OOps " << schi << std::endl;
              EEDetId myID(vec[ic]);
              //  std::cout << " DetId " << myID << " " << myID.isc() << " " <<  myID.zside() <<  " " << myID.isc()+(myID.zside()+1)*158 << std::endl;
            }
      
          theTTDetIds_[tthashedindex]=towid1;

          // check if this SC is already in the list of the corresponding TT
          std::vector<int>::const_iterator itcheck=find(SCofTT_[tthashedindex].begin(),
                                                        SCofTT_[tthashedindex].end(),
                                                        schi);
          if(itcheck==SCofTT_[tthashedindex].end())
            SCofTT_[tthashedindex].push_back(schi);
      
          // check if this crystal is already in the list of crystals per sc
          itcheck=find(CrystalsinSC_[schi].begin(),CrystalsinSC_[schi].end(),cellhashedindex);
          if(itcheck==CrystalsinSC_[schi].end())
            CrystalsinSC_[schi].push_back(cellhashedindex);

          // check if the TT is already in the list of sc
          //      std::cout << " SCHI " << schi << " " << TTofSC_.size() << std::endl;
          //      std::cout << TTofSC_[schi].size() << std::endl;
          itcheck=find(TTofSC_[schi].begin(),TTofSC_[schi].end(),tthashedindex);
          if(itcheck==TTofSC_[schi].end())
            TTofSC_[schi].push_back(tthashedindex);
        }
      // std::cout << " Made the array " << std::endl;

    }

 protected:


    bool isHighInterest(const EEDetId& detid) {

      int schi=SChashedIndex(detid);
      //  std::cout <<  detid << "  " << schi << " ";
      // check if it has already been treated or not
      // 0 <=> not treated
      // 1 <=> high interest
      // -1 <=> low interest
      //  std::cout << SCHighInterest_[schi] << std::endl;
      if(SCHighInterest_[schi]!=0) return (SCHighInterest_[schi]>0);

      // now look if a TT contributing is of high interest
      const std::vector<int> & tts(TTofSC_[schi]);
      unsigned size=tts.size();
      bool result=false;
      for(unsigned itt=0;itt<size&&!result;++itt)
        {
          EcalTrigTowerDetId towid = theTTDetIds_[tts[itt]];
          if(srFullReadOut(towid)) result=true;
        }
      SCHighInterest_[schi]=(result)? 1:-1;
      theFiredSC_.push_back(schi);
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

    // arraws for the selective readout emulation
    // fast EEDetId -> TT hashedindex conversion
    std::vector<int>  towerOf_;
    // vector of the original DetId if needed
    std::vector<EcalTrigTowerDetId> theTTDetIds_;
    // list of SC "contained" in a TT.
    std::vector<std::vector<int> > SCofTT_;
    // list of TT of a given sc
    std::vector<std::vector<int> > TTofSC_;
    // status of each SC 
    std::vector<int> SCHighInterest_;
    // list of fired SC
    std::vector<int> theFiredSC_;
    // the list of fired TT
    std::vector<int> theFiredTTs_;
    // the cells in each SC
    std::vector<std::vector<int> > CrystalsinSC_;

    // selective readout flags collection
    edm::Handle<EBSrFlagCollection> srFlagHandle_;

 private:
    // there are 2448 TT in the barrel. 
    inline int TThashedIndexforEE(int originalhi) const {return originalhi-2448;}
    inline int TThashedIndexforEE(const EcalTrigTowerDetId &detid) const {return detid.hashedIndex()-2448;}
    // the number of the SuperCrystals goes from 1 to 316 (with some holes) in each EE
    // z should -1 or 1 
    inline int SChashedIndex(int SC,int z) const {return SC+(z+1)*158;}
    inline int SChashedIndex(const EEDetId& detid) const {
      //    std::cout << "In SC hashedIndex " <<  detid.isc() << " " << detid.zside() << " " << detid.isc()+(detid.zside()+1)*158 << std::endl;
      return detid.isc()+(detid.zside()+1)*158;
    }

};


typedef PFEcalEndcapRecHitCreator<EcalEndcapGeometry,PFLayer::ECAL_ENDCAP,EcalEndcap> PFEERecHitCreator;

#endif
