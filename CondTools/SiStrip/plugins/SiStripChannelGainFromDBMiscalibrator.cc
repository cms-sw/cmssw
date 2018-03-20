// -*- C++ -*-
//
// Package:    CondTools/SiStrip
// Class:      SiStripChannelGainFromDBMiscalibrator
// 
/**\class SiStripChannelGainFromDBMiscalibrator SiStripChannelGainFromDBMiscalibrator.cc CondTools/SiStrip/plugins/SiStripChannelGainFromDBMiscalibrator.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Marco Musich
//         Created:  Tue, 03 Oct 2017 12:57:34 GMT
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h" 
#include "Geometry/Records/interface/TrackerTopologyRcd.h" 

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h" 
#include "CondFormats/SiStripObjects/interface/SiStripSummary.h"

#include "TRandom3.h"

//
// class declaration
//

namespace ApvGain {
  struct GainSmearings{
    GainSmearings(){
      m_doScale = false;
      m_doSmear = false;
      m_scaleFactor = 1.;
      m_smearFactor = 0.;
    }
    ~GainSmearings(){}
    
    void setSmearing(bool doScale,bool doSmear,double the_scaleFactor,double the_smearFactor){
      m_doScale = doScale;
      m_doSmear = doSmear;
      m_scaleFactor = the_scaleFactor;
      m_smearFactor = the_smearFactor;
    }
    
    bool m_doScale;
    bool m_doSmear;
    double m_scaleFactor;
    double m_smearFactor;
  };

}

class SiStripChannelGainFromDBMiscalibrator : public edm::one::EDAnalyzer<>  {
   public:
      explicit SiStripChannelGainFromDBMiscalibrator(const edm::ParameterSet&);
      ~SiStripChannelGainFromDBMiscalibrator();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      std::unique_ptr<SiStripApvGain> getNewObject(const std::map<std::pair<uint32_t,int>,float>& theMap);
      sistripsummary::TrackerRegion getRegionFromString(std::string region);
      std::vector<sistripsummary::TrackerRegion> getRegionsFromDetId(const TrackerTopology* tTopo,DetId detid);
      virtual void endJob() override;

      // ----------member data ---------------------------
      const std::string m_Record;  
      const uint32_t m_gainType;
      const std::vector<edm::ParameterSet> m_parameters;
};

//
// constructors and destructor
//
SiStripChannelGainFromDBMiscalibrator::SiStripChannelGainFromDBMiscalibrator(const edm::ParameterSet& iConfig):
  m_Record{iConfig.getUntrackedParameter<std::string> ("record" , "SiStripApvGainRcd")},
  m_gainType{iConfig.getUntrackedParameter<uint32_t>("gainType",1)},
  m_parameters{iConfig.getParameter<std::vector<edm::ParameterSet> >("params")}
{
   //now do what ever initialization is needed
}


SiStripChannelGainFromDBMiscalibrator::~SiStripChannelGainFromDBMiscalibrator()
{
}


//
// member functions
//

// ------------ method called for each event  ------------
void
SiStripChannelGainFromDBMiscalibrator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::ESHandle<TrackerTopology> tTopoHandle;
   iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
   const auto* const tTopo = tTopoHandle.product();

   std::vector<std::string> partitions;

   // fill the list of partitions 
   for(auto& thePSet : m_parameters){
     const std::string partition(thePSet.getParameter<std::string>("partition"));
     // only if it is not yet in the list
     if(std::find(partitions.begin(), partitions.end(), partition) == partitions.end()) {
       partitions.push_back(partition);
     }
   }

   std::map<sistripsummary::TrackerRegion,ApvGain::GainSmearings> mapOfSmearings;

   for(auto& thePSet : m_parameters){
     
     const std::string partition(thePSet.getParameter<std::string>("partition"));
     sistripsummary::TrackerRegion region = this->getRegionFromString(partition);
     
     bool    m_doScale(thePSet.getParameter<bool>("doScale"));
     bool    m_doSmear(thePSet.getParameter<bool>("doSmear"));
     double  m_scaleFactor(thePSet.getParameter<double>("scaleFactor"));
     double  m_smearFactor(thePSet.getParameter<double>("smearFactor"));
     
     ApvGain::GainSmearings params = ApvGain::GainSmearings();
     params.setSmearing(m_doScale,m_doSmear,m_scaleFactor,m_smearFactor);
     mapOfSmearings[region]=params;
   }
   

   edm::ESHandle<SiStripGain> SiStripApvGain_;
   iSetup.get<SiStripGainRcd>().get(SiStripApvGain_);

   std::map<std::pair<uint32_t,int>,float> theMap;
   std::shared_ptr<TRandom3> random(new TRandom3(1));
   
   std::vector<uint32_t> detid;
   SiStripApvGain_->getDetIds(detid);
   for (const auto & d : detid) {
     SiStripApvGain::Range range=SiStripApvGain_->getRange(d,m_gainType);
     float nAPV=0;

     auto regions = getRegionsFromDetId(tTopo,d); 

     // sort by largest to smallest
     std::sort(regions.rbegin(), regions.rend());
     
     ApvGain::GainSmearings params = ApvGain::GainSmearings();
     
     for (unsigned int j=0; j<regions.size();j++){
       bool checkRegion = (mapOfSmearings.count(regions[j]) != 0);

       if(!checkRegion) {
	 // if the subdetector is not in the list and there's no indication for the whole tracker, just use the default 
	 // i.e. no change
	 continue;
       } else {
	 params = mapOfSmearings[regions[j]];
	 break;
       }
     }
     
     for(int it=0;it<range.second-range.first;it++){
       nAPV+=1;
       float Gain=SiStripApvGain_->getApvGain(it,range);
       std::pair<uint32_t,int> index = std::make_pair(d,nAPV);
       
       if(params.m_doScale){
	 Gain*=params.m_scaleFactor;
       }
       
       if(params.m_doSmear){
	 float smearedGain = random->Gaus(Gain,params.m_smearFactor);
	 Gain=smearedGain;
       }

       theMap[index]=Gain;
       
     } // loop over APVs
   } // loop over DetIds

   std::unique_ptr<SiStripApvGain> theAPVGains = this->getNewObject(theMap);

   // write out the APVGains record
   edm::Service<cond::service::PoolDBOutputService> poolDbService;
  
   if( poolDbService.isAvailable() )
     poolDbService->writeOne(theAPVGains.get(),poolDbService->currentTime(),m_Record);
   else
     throw std::runtime_error("PoolDBService required.");
 
}


// ------------ method called once each job just before starting event loop  ------------
void 
SiStripChannelGainFromDBMiscalibrator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SiStripChannelGainFromDBMiscalibrator::endJob() 
{
}

//********************************************************************************//
std::unique_ptr<SiStripApvGain>
SiStripChannelGainFromDBMiscalibrator::getNewObject(const std::map<std::pair<uint32_t,int>,float>& theMap) 
{
  std::unique_ptr<SiStripApvGain> obj = std::unique_ptr<SiStripApvGain>(new SiStripApvGain());
  
  std::vector<float> theSiStripVector;
  uint32_t PreviousDetId = 0; 
  for(const auto &element : theMap){
    uint32_t DetId = element.first.first;
    if(DetId != PreviousDetId){
      if(!theSiStripVector.empty()){
	SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
	if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
      }
      theSiStripVector.clear();
      PreviousDetId = DetId;
    }
    theSiStripVector.push_back(element.second);
    
    edm::LogInfo("SiStripChannelGainFromDBMiscalibrator")<<" DetId: "<<DetId 
						 <<" APV:   "<<element.first.second
						 <<" Gain:  "<<element.second
						 <<std::endl;
  }
  
  if(!theSiStripVector.empty()){
    SiStripApvGain::Range range(theSiStripVector.begin(),theSiStripVector.end());
    if ( !obj->put(PreviousDetId,range) )  printf("Bug to put detId = %i\n",PreviousDetId);
  }
  
  return obj;
}


// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SiStripChannelGainFromDBMiscalibrator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  
  desc.setComment("Creates rescaled / smeared SiStrip Gain payload. Can be used for both G1 and G2."
                  "PoolDBOutputService must be set up for 'SiStripApvGainRcd'.");
  
  edm::ParameterSetDescription descScaler;
  descScaler.setComment("ParameterSet specifying the Strip tracker partition to be scaled / smeared "
                        "by a given factor.");

  descScaler.add<std::string>("partition", "Tracker");
  descScaler.add<bool>("doScale",true);
  descScaler.add<bool>("doSmear",true);
  descScaler.add<double>("scaleFactor", 1.0);
  descScaler.add<double>("smearFactor", 1.0);
  desc.addVPSet("params", descScaler, std::vector<edm::ParameterSet>(1));

  desc.addUntracked<std::string>("record","SiStripApvGainRcd");
  desc.addUntracked<unsigned int>("gainType",1);

  descriptions.add("scaleAndSmearSiStripGains", desc);

}

/*--------------------------------------------------------------------*/
sistripsummary::TrackerRegion SiStripChannelGainFromDBMiscalibrator::getRegionFromString(std::string region)
/*--------------------------------------------------------------------*/
{
  if(region.find("Tracker")!=std::string::npos){
    return sistripsummary::TRACKER ;
  } else if(region.find("TIB")!=std::string::npos){
    if (region=="TIB_1") return sistripsummary::TIB_1;
    else if (region=="TIB_2") return sistripsummary::TIB_2;
    else if (region=="TIB_3") return sistripsummary::TIB_3;
    else if (region=="TIB_4") return sistripsummary::TIB_4;
    else return sistripsummary::TIB ;
  } else if(region.find("TOB")!=std::string::npos){ 
    if (region=="TOB_1") return sistripsummary::TOB_1;
    else if (region=="TOB_2") return sistripsummary::TOB_2;
    else if (region=="TOB_3") return sistripsummary::TOB_3;
    else if (region=="TOB_4") return sistripsummary::TOB_4;
    else if (region=="TOB_5") return sistripsummary::TOB_5;
    else if (region=="TOB_6") return sistripsummary::TOB_6;
    else return sistripsummary::TOB ;
  } else if(region.find("TID")!=std::string::npos){
    if(region.find("TIDM")!=std::string::npos){
      if (region=="TIDM_1") return sistripsummary::TIDM_1;
      else if (region=="TIDM_2") return sistripsummary::TIDM_2;
      else if (region=="TIDM_3") return sistripsummary::TIDM_3;
      else return sistripsummary::TIDM;
    } else if(region.find("TIDP")!=std::string::npos){
      if (region=="TIDP_1") return sistripsummary::TIDP_1;
      else if (region=="TIDP_2") return sistripsummary::TIDP_2;
      else if (region=="TIDP_3") return sistripsummary::TIDP_3;
      else return sistripsummary::TIDP;
    } else return sistripsummary::TID ;
  } else if(region.find("TEC")!=std::string::npos) {
    if(region.find("TECM")!=std::string::npos){
      if (region=="TECM_1") return sistripsummary::TECM_1;
      else if (region=="TECM_2") return sistripsummary::TECM_2;
      else if (region=="TECM_3") return sistripsummary::TECM_3;
      else if (region=="TECM_4") return sistripsummary::TECM_4;
      else if (region=="TECM_5") return sistripsummary::TECM_5;
      else if (region=="TECM_6") return sistripsummary::TECM_6;
      else if (region=="TECM_7") return sistripsummary::TECM_7;
      else if (region=="TECM_8") return sistripsummary::TECM_8;
      else if (region=="TECM_9") return sistripsummary::TECM_9;
      else return sistripsummary::TECM;
    } else if(region.find("TECP")!=std::string::npos){
      if (region=="TECP_1") return sistripsummary::TECP_1;
      else if (region=="TECP_2") return sistripsummary::TECP_2;
      else if (region=="TECP_3") return sistripsummary::TECP_3;
      else if (region=="TECP_4") return sistripsummary::TECP_4;
      else if (region=="TECP_5") return sistripsummary::TECP_5;
      else if (region=="TECP_6") return sistripsummary::TECP_6;
      else if (region=="TECP_7") return sistripsummary::TECP_7;
      else if (region=="TECP_8") return sistripsummary::TECP_8;
      else if (region=="TECP_9") return sistripsummary::TECP_9;
      else return sistripsummary::TECP;
    } else return sistripsummary::TEC ;
  } else {
    edm::LogError("LogicError") << "Unknown partition: " << region;
    throw cms::Exception("Invalid Partition passed"); 
  }  
}

/*--------------------------------------------------------------------*/
std::vector<sistripsummary::TrackerRegion> SiStripChannelGainFromDBMiscalibrator::getRegionsFromDetId(const TrackerTopology* m_trackerTopo,DetId detid)
/*--------------------------------------------------------------------*/      
{
  int layer    = 0;
  int side     = 0;
  int subdet   = 0;
  int detCode  = 0;

  std::vector<sistripsummary::TrackerRegion> ret;

  switch (detid.subdetId()) {
  case StripSubdetector::TIB:
    layer = m_trackerTopo->tibLayer(detid);
    subdet = 1;
    break;
  case StripSubdetector::TOB:
    layer = m_trackerTopo->tobLayer(detid);
    subdet = 2;
    break;
  case StripSubdetector::TID:
    // is this module in TID+ or TID-?
    layer = m_trackerTopo->tidWheel(detid);
    side  = m_trackerTopo->tidSide(detid);
    subdet = 3*10+side;
    break;
  case StripSubdetector::TEC:
    // is this module in TEC+ or TEC-?
    layer = m_trackerTopo->tecWheel(detid);
    side  = m_trackerTopo->tecSide(detid);
    subdet = 4*10+side;
    break;
  }
  
  detCode = (subdet*10)+layer;
  
  ret.push_back(static_cast<sistripsummary::TrackerRegion>(detCode));

  if(subdet/10 > 0) {
    ret.push_back(static_cast<sistripsummary::TrackerRegion>(subdet/10));
  }

  ret.push_back(static_cast<sistripsummary::TrackerRegion>(subdet));
  ret.push_back(sistripsummary::TRACKER);

  return ret;
}


//define this as a plug-in
DEFINE_FWK_MODULE(SiStripChannelGainFromDBMiscalibrator);


