#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterInfo.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsService.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsSubtractor.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripMedianCommonModeNoiseSubtraction.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripTT6CommonModeNoiseSubtraction.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfoProducer.h"


namespace cms
{
  SiStripClusterInfoProducer::SiStripClusterInfoProducer(edm::ParameterSet const& conf) : 
    conf_(conf),
    SiStripNoiseService_(conf),
    RawModeRun_(conf.getParameter<bool>("RawModeRun")),
    _NEIGH_STRIP_(conf.getParameter<uint32_t>("NeighbourStripNumber")),
    CMNSubtractionMode_(conf.getParameter<std::string>("CommonModeNoiseSubtractionMode")){


    edm::LogInfo("SiStripClusterInfoProducer") << "[SiStripClusterInfoProducer::SiStripClusterInfoProducer] Constructing object...";
      
 
    //------------------------
    if ( CMNSubtractionMode_ == "Median") { 
      SiStripCommonModeNoiseSubtractor_ = new SiStripMedianCommonModeNoiseSubtraction();
      validCMNSubtraction_ = true;
    }
    else if ( CMNSubtractionMode_ == "TT6") { 
      SiStripCommonModeNoiseSubtractor_ = new SiStripTT6CommonModeNoiseSubtraction(conf.getParameter<double>("CutToAvoidSignal"));
      validCMNSubtraction_ = true;
    }
    else {
      edm::LogError("SiStripClusterInfoProducer") << "[SiStripClusterInfoProducer::SiStripClusterInfoProducer] No valid CommonModeNoiseSubtraction Mode selected, possible CMNSubtractionMode: Median or TT6" << std::endl;
      validCMNSubtraction_ = false;
    } 

  //------------------------

    SiStripPedestalsSubtractor_ = new SiStripPedestalsSubtractor();

    produces< edm::DetSetVector<SiStripClusterInfo> > ();
  }

  // Virtual destructor needed.
  SiStripClusterInfoProducer::~SiStripClusterInfoProducer() { 
    edm::LogInfo("SiStripClusterInfoProducer") << "[SiStripClusterInfoProducer::~SiStripClusterInfoProducer] Destructing object...";
  }  

  // Functions that gets called by framework every event
  void SiStripClusterInfoProducer::produce(edm::Event& e, const edm::EventSetup& es)
  {
    // Step A: Get ESObject 
    SiStripNoiseService_.setESObjects(es);

    // Step B: Get Inputs 
    edm::Handle< edm::DetSetVector<SiStripCluster> >  input_cluster;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> >  input_rawdigi;

    // Step C: produce output product
    std::vector< edm::DetSet<SiStripClusterInfo> > vSiStripClusterInfo;
    vSiStripClusterInfo.reserve(10000);
    std::string ClusterProducer = conf_.getParameter<std::string>("ClusterProducer");
    std::string ClusterLabel = conf_.getParameter<std::string>("ClusterLabel");
    e.getByLabel(ClusterProducer,ClusterLabel,input_cluster);  
    if (input_cluster->size()){
      cluster_algorithm(*input_cluster,vSiStripClusterInfo);
    
      if (RawModeRun_){
	typedef std::vector<edm::ParameterSet> Parameters;
	Parameters RawDigiProducersList = conf_.getParameter<Parameters>("RawDigiProducersList");
	Parameters::iterator itRawDigiProducersList = RawDigiProducersList.begin();
	for(; itRawDigiProducersList != RawDigiProducersList.end(); ++itRawDigiProducersList ) {
	  std::string rawdigiProducer = itRawDigiProducersList->getParameter<std::string>("RawDigiProducer");
	  std::string rawdigiLabel = itRawDigiProducersList->getParameter<std::string>("RawDigiLabel");
	  e.getByLabel(rawdigiProducer,rawdigiLabel,input_rawdigi);
	  if (input_rawdigi->size())
	    digi_algorithm(*input_rawdigi,vSiStripClusterInfo,rawdigiLabel);
	}
      }
    }
    // Step D: create and fill output collection
    std::auto_ptr< edm::DetSetVector<SiStripClusterInfo> > output(new edm::DetSetVector<SiStripClusterInfo>(vSiStripClusterInfo) );

    // Step D: write output to file
    e.put(output);
  }


  void SiStripClusterInfoProducer::cluster_algorithm(const edm::DetSetVector<SiStripCluster>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output){
    
    edm::DetSetVector<SiStripCluster>::const_iterator detset_iter=input.begin();
    for (; detset_iter!=input.end();detset_iter++){
      edm::DetSet<SiStripClusterInfo> DetSet_(detset_iter->id);
      edm::DetSet<SiStripCluster>::const_iterator cluster_iter=detset_iter->data.begin();
      for (; cluster_iter!=detset_iter->data.end(); cluster_iter++){
	
	SiStripClusterInfo SiStripClusterInfo_(*cluster_iter);
	
	int count=0;	
	uint16_t maxPosition=0;
	float charge=0;
	float noise=0;
	float noise2=0;
	float chargeL=0;
	float chargeR=0;
	float maxCharge=0;
	std::vector<float>   stripNoises;
	
	const std::vector<uint16_t>& amplitudes_ =  cluster_iter->amplitudes();
	for(size_t i=0; i<amplitudes_.size();i++)
	  if (amplitudes_[i]>0){
	    charge+=amplitudes_[i];
	    float noise=SiStripNoiseService_.getNoise(detset_iter->id,cluster_iter->firstStrip()+i);
	    stripNoises.push_back(noise);
	    noise2+=noise*noise;
	    
	    count++;
	    
	    //Find strip with max charge
	    if (maxCharge<amplitudes_[i]){
	      maxCharge=amplitudes_[i];
	      maxPosition=i;
	    }
	  }
	
	//Evaluate Ql and Qr
	for(size_t i=0; i<amplitudes_.size();i++){
	  if (i<maxPosition)
	    chargeL+=amplitudes_[i];	  
	  if (i>maxPosition)
	    chargeR+=amplitudes_[i];
	}
	
	noise=sqrt(noise2/count);
	maxPosition+=cluster_iter->firstStrip();

	SiStripClusterInfo_.setCharge(charge);
	SiStripClusterInfo_.setNoise(noise);
	SiStripClusterInfo_.setMaxCharge(maxCharge);
	SiStripClusterInfo_.setMaxPos(maxPosition);
	SiStripClusterInfo_.setChargeL(chargeL);
	SiStripClusterInfo_.setChargeR(chargeR);
	SiStripClusterInfo_.setStripNoises(stripNoises);
      
	DetSet_.data.push_back(SiStripClusterInfo_);
    }
      if (DetSet_.data.size())
	output.push_back(DetSet_);  // insert the DetSet<SiStripClusterInfo> in the  DetSetVec<SiStripClusterInfo> only if there is at least a digi
    }
  }

  
  void SiStripClusterInfoProducer::digi_algorithm(const edm::DetSetVector<SiStripRawDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output,std::string rawdigiLabel){
  
    std::vector< edm::DetSet<SiStripClusterInfo> >::iterator output_iter=output.begin();
    for (; output_iter!=output.end();output_iter++){
      edm::DetSetVector<SiStripRawDigi>::const_iterator detset_iter=input.find(output_iter->id);
      if (detset_iter!=input.end()){
	std::vector<int16_t> vssRd(detset_iter->data.size());

	if ( rawdigiLabel == "VirginRaw" ) {
	  //Create a temporary edm::DetSet<SiStripRawDigi> 

	  SiStripPedestalsSubtractor_->subtract(*detset_iter,vssRd);
	  if (validCMNSubtraction_)
	    SiStripCommonModeNoiseSubtractor_->subtract(detset_iter->id,vssRd);
	  else
	    edm::LogError("SiStripClusterInfoProducer") << "[SiStripClusterInfoProducer::digi_algorithm] No valid CommonModeNoiseSubtraction Mode selected, possible CMNSubtractionMode: Median or TT6" << std::endl;
	} 

	edm::DetSet<SiStripClusterInfo>::iterator cluster_iter=output_iter->data.begin();      
	for (; cluster_iter!=output_iter->data.end(); cluster_iter++){	
	  uint16_t max_strip=cluster_iter->maxPos();
	  std::vector<uint16_t>   RawDigiAmplitudesL;
	  std::vector<uint16_t>   RawDigiAmplitudesR;
	  for (uint16_t istrip=max_strip-1; istrip>max_strip-(_NEIGH_STRIP_+1);istrip--){
	    if ( rawdigiLabel == "VirginRaw" ){
	      RawDigiAmplitudesL.push_back(vssRd[istrip]);
	    }else{
	      RawDigiAmplitudesL.push_back(detset_iter->data[istrip].adc());
	    }
	  }
	  for (uint16_t istrip=max_strip+1; istrip>max_strip+(_NEIGH_STRIP_+1);istrip++){
	    if ( rawdigiLabel == "VirginRaw" ){
	      RawDigiAmplitudesR.push_back(vssRd[istrip]);
	    }else{
	      RawDigiAmplitudesR.push_back(detset_iter->data[istrip].adc());
	    }
	  }
	  cluster_iter->setRawDigiAmplitudesL(RawDigiAmplitudesL);
	  cluster_iter->setRawDigiAmplitudesL(RawDigiAmplitudesR);
	}
      }
    }
  }
}
  
