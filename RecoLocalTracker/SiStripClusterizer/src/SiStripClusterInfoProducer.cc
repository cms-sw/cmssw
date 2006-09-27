#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Data Formats
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterInfo.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripNoiseService.h"

#include "RecoLocalTracker/SiStripClusterizer/interface/SiStripClusterInfoProducer.h"

namespace cms
{
  SiStripClusterInfoProducer::SiStripClusterInfoProducer(edm::ParameterSet const& conf) : 
    conf_(conf),
    SiStripNoiseService_(conf){

    edm::LogInfo("SiStripClusterInfoProducer") << "[SiStripClusterInfoProducer::SiStripClusterInfoProducer] Constructing object...";

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
    edm::Handle< edm::DetSetVector<SiStripCluster> >  input;

    // Step C: produce output product
    std::vector< edm::DetSet<SiStripClusterInfo> > vSiStripClusterInfo;
    vSiStripClusterInfo.reserve(10000);
    std::string ClusterProducer = conf_.getParameter<std::string>("ClusterProducer");
    std::string ClusterLabel = conf_.getParameter<std::string>("ClusterLabel");
    e.getByLabel(ClusterProducer,ClusterLabel,input);  
    if (input->size())
      algorithm(*input,vSiStripClusterInfo);
    
    
    // Step D: create and fill output collection
    std::auto_ptr< edm::DetSetVector<SiStripClusterInfo> > output(new edm::DetSetVector<SiStripClusterInfo>(vSiStripClusterInfo) );

    // Step D: write output to file
    e.put(output);
  }


  void SiStripClusterInfoProducer::algorithm(const edm::DetSetVector<SiStripCluster>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output){
    
    edm::DetSetVector<SiStripCluster>::const_iterator detset_iter=input.begin();
    for (; detset_iter!=input.end();detset_iter++){
      edm::DetSet<SiStripClusterInfo> DetSet_(detset_iter->id);
      edm::DetSet<SiStripCluster>::const_iterator cluster_iter=detset_iter->data.begin();
      for (; cluster_iter!=detset_iter->data.end(); cluster_iter++){
	
	SiStripClusterInfo SiStripClusterInfo_(*cluster_iter);

	int count=0;	
	float charge=0;
	float noise=0;
	float noise2=0;
	float chargeL=0;
	float chargeR=0;
	float maxCharge=0;
	float maxPosition=0;
	std::vector<float>   stripNoises;

	const std::vector<short>& amplitudes_ =	cluster_iter->amplitudes();
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
}


