#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//Data Formats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripDigi/interface/SiStripRawDigi.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "CommonTools/SiStripZeroSuppression/interface/SiStripPedestalsService.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripMedianCommonModeNoiseSubtraction.h"
#include "CommonTools/SiStripZeroSuppression/interface/SiStripTT6CommonModeNoiseSubtraction.h"

#include "AnalysisAlgos/SiStripClusterInfoProducer/interface/SiStripClusterInfoProducer.h"

#include <sstream>

namespace cms
{
  SiStripClusterInfoProducer::SiStripClusterInfoProducer(edm::ParameterSet const& conf) : 
    conf_(conf),
    SiStripNoiseService_(conf),
    SiStripPedestalsService_(conf),
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
    SiStripPedestalsSubtractor_->setSiStripPedestalsService(&SiStripPedestalsService_);

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
    SiStripPedestalsService_.setESObjects(es);

    // Step B: Get Inputs 
    edm::Handle< edm::DetSetVector<SiStripCluster> >  input_cluster;
    edm::Handle< edm::DetSetVector<SiStripRawDigi> >  input_rawdigi;
    edm::Handle< edm::DetSetVector<SiStripDigi> >     input_digi;

    // Step C: produce output product
    std::vector< edm::DetSet<SiStripClusterInfo> > vSiStripClusterInfo;
    vSiStripClusterInfo.reserve(10000);
    std::string ClusterProducer = conf_.getParameter<std::string>("ClusterProducer");
    std::string ClusterLabel = conf_.getParameter<std::string>("ClusterLabel");
    e.getByLabel(ClusterProducer,ClusterLabel,input_cluster);  
    
    if (input_cluster->size()){
    
      //Dump Cluster Info from Cluster to ClusterInfo
      cluster_algorithm(*input_cluster,vSiStripClusterInfo);
    
      //Dump Digi Info from (Raw)Digi to ClusterInfo
      bool RawMode=false;      
      typedef std::vector<edm::ParameterSet> Parameters;
      Parameters RawDigiProducersList = conf_.getParameter<Parameters>("RawDigiProducersList");      
      Parameters::iterator itRawDigiProducersList = RawDigiProducersList.begin();
      for(; itRawDigiProducersList != RawDigiProducersList.end(); ++itRawDigiProducersList ) {
	std::string rawdigiProducer = itRawDigiProducersList->getParameter<std::string>("RawDigiProducer");
	std::string rawdigiLabel = itRawDigiProducersList->getParameter<std::string>("RawDigiLabel");
	e.getByLabel(rawdigiProducer,rawdigiLabel,input_rawdigi);
	if (input_rawdigi->size()){
	  RawMode=true;
	  rawdigi_algorithm(*input_rawdigi,vSiStripClusterInfo,rawdigiLabel);
	}
      }
      if (!RawMode){
	std::string DigiProducer = conf_.getParameter<std::string>("DigiProducer");
 	std::string DigiLabel = conf_.getParameter<std::string>("DigiLabel");
 	e.getByLabel(DigiProducer,DigiLabel,input_digi);  
 	if (input_digi->size()){
 	  digi_algorithm(*input_digi,vSiStripClusterInfo);	  
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
	for(size_t i=0; i<amplitudes_.size();i++){
	  float noise=SiStripNoiseService_.getNoise(detset_iter->id,cluster_iter->firstStrip()+i);
	  stripNoises.push_back(noise);
	  
	  if (amplitudes_[i]>0){
	    charge+=amplitudes_[i];
	    noise2+=noise*noise;
	    
	    count++;
	    
	    //Find strip with max charge
	    if (maxCharge<amplitudes_[i]){
	      maxCharge=amplitudes_[i];
	      maxPosition=i;
	    }
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
      
	if (edm::isDebugEnabled()){
	  std::stringstream ss;
	  SiStripClusterInfo_.print(ss);
	  LogTrace("SiStripClusterInfoProducer") << "[SiStripClusterInfoProducer::cluster_algorithm] Cluster " << ss.str();
	}
	DetSet_.data.push_back(SiStripClusterInfo_);
    }
      if (DetSet_.data.size())
	output.push_back(DetSet_);  // insert the DetSet<SiStripClusterInfo> in the  DetSetVec<SiStripClusterInfo> only if there is at least a digi
    }
  }

  void SiStripClusterInfoProducer::digi_algorithm(const edm::DetSetVector<SiStripDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output){
    std::vector<int16_t> vstrip;
    std::vector<int16_t> vadc;

    std::vector< edm::DetSet<SiStripClusterInfo> >::iterator output_iter=output.begin();
    for (; output_iter!=output.end();output_iter++){
      edm::DetSetVector<SiStripDigi>::const_iterator detset_iter=input.find(output_iter->id);
      if (detset_iter!=input.end()){
      
	//Get List of digis for the current DetId
	vstrip.clear();
	vadc.clear();
	std::stringstream sss;
	int idig=0;
	edm::DetSet<SiStripDigi>::const_iterator digi_iter=detset_iter->data.begin();
	for(;digi_iter!=detset_iter->data.end();digi_iter++){
	  vstrip.push_back(digi_iter->strip());
	  vadc.push_back(digi_iter->adc());
	  if (edm::isDebugEnabled())
	    sss << "\n " << idig++ << " \t digi strip " << digi_iter->strip() << " digi adc " << digi_iter->adc();
	}
	LogTrace("SiStripClusterInfoProducer") << "\n["<<__PRETTY_FUNCTION__<<"]\n detid " << output_iter->id  << sss.str();

	findNeigh("digi",output_iter,vadc,vstrip);
      }
    }
  }  

  void SiStripClusterInfoProducer::rawdigi_algorithm(const edm::DetSetVector<SiStripRawDigi>& input,std::vector< edm::DetSet<SiStripClusterInfo> >& output,std::string rawdigiLabel){
    LogTrace("SiStripClusterInfoProducer") << "["<<__PRETTY_FUNCTION__<<"]";    
    
    std::vector< edm::DetSet<SiStripClusterInfo> >::iterator output_iter=output.begin();
    for (; output_iter!=output.end();output_iter++){
      edm::DetSetVector<SiStripRawDigi>::const_iterator detset_iter=input.find(output_iter->id);
      if (detset_iter!=input.end()){
	std::vector<int16_t> vssRd(detset_iter->data.size());

	if ( rawdigiLabel == "ProcessedRaw"){
	  edm::DetSet<SiStripRawDigi>::const_iterator digi_iter=detset_iter->data.begin();
	  for(;digi_iter!=detset_iter->data.end();digi_iter++){
	    vssRd.push_back(digi_iter->adc());

	    if (edm::isDebugEnabled()){
	      std::stringstream sss;
	      int idig=0;
	      std::vector<int16_t>::const_iterator digi_iter=vssRd.begin();
	      for(;digi_iter!=vssRd.end();digi_iter++)
		sss << "\n digi strip " << idig++ << " digi adc " << *digi_iter;
	      LogTrace("SiStripClusterInfoProducer") << " detid " << output_iter->id  << " Pedestal subtracted digis \n" << sss.str();	
	    }	    
	  }
	} else if ( rawdigiLabel == "VirginRaw" ) {
	  
	  if (edm::isDebugEnabled()){
	    std::stringstream sss;
	    int idig=0;
	    edm::DetSet<SiStripRawDigi>::const_iterator digi_iter=detset_iter->data.begin();
	    for(;digi_iter!=detset_iter->data.end();digi_iter++)
	      sss << "\n digi strip " << idig++ << " digi adc " << digi_iter->adc();
	    LogTrace("SiStripClusterInfoProducer") << " detid " << output_iter->id  << " RawDigis \n" << sss.str();
	  }

	  //Subtract Pedestals
	  SiStripPedestalsSubtractor_->subtract(*detset_iter,vssRd);
	  
	  if (edm::isDebugEnabled()){
	    std::stringstream sss;
	    int idig=0;
	    std::vector<int16_t>::const_iterator digi_iter=vssRd.begin();
	    for(;digi_iter!=vssRd.end();digi_iter++)
	      sss << "\n digi strip " << idig++ << " digi adc " << *digi_iter;
	    LogTrace("SiStripClusterInfoProducer") << " detid " << output_iter->id  << " Pedestal subtracted digis \n" << sss.str();	
	  }

	  //Subtract CMN
	  if (validCMNSubtraction_){
	    SiStripCommonModeNoiseSubtractor_->subtract(detset_iter->id,vssRd);

	    if (edm::isDebugEnabled()){
	      std::stringstream sss;
	      int idig=0;
	      std::vector<int16_t>::const_iterator digi_iter=vssRd.begin();
	      for(;digi_iter!=vssRd.end();digi_iter++)
		sss << "\n digi strip " << idig++ << " digi adc " << *digi_iter;
	      LogTrace("SiStripClusterInfoProducer") << " detid " << output_iter->id  << " CMN subtracted digis \n" << sss.str();	
	    }
	    
	  }else{
	    throw cms::Exception("") <<"[" << __PRETTY_FUNCTION__<<"] No valid CommonModeNoiseSubtraction Mode selected, possible CMNSubtractionMode: Median or TT6" << std::endl;
	  }
	} else {
	  return;
	}

	
       	findNeigh("raw",output_iter,vssRd,vssRd);	
      }
    }
  }

  void SiStripClusterInfoProducer::findNeigh(char* mode,std::vector< edm::DetSet<SiStripClusterInfo> >::iterator output_iter,std::vector<int16_t>& vadc,std::vector<int16_t>& vstrip){

    //Find Digi adiacent to the clusters of this detid
    int16_t lastStrip_previousCluster=-1;
    int16_t firstStrip_nextCluster=10000;
    edm::DetSet<SiStripClusterInfo>::iterator cluster_iter=output_iter->data.begin();      
    for (; cluster_iter!=output_iter->data.end(); cluster_iter++){	
      
      //Avoid overlapping with neighbour clusters
      if (cluster_iter!=output_iter->data.begin())
	lastStrip_previousCluster=(cluster_iter-1)->firstStrip()+(cluster_iter-1)->stripAmplitudes().size() -1;	  
      if (cluster_iter!=output_iter->data.end()-1)
	firstStrip_nextCluster=(cluster_iter+1)->firstStrip();
      
      LogTrace("SiStripClusterInfoProducer") << "["<< __PRETTY_FUNCTION__ << "]\n lastStrip_previousCluster " << lastStrip_previousCluster << " firstStrip_nextCluster " << firstStrip_nextCluster ;
      
      
      int16_t firstStrip=cluster_iter->firstStrip();
      int16_t lastStrip=firstStrip + cluster_iter->stripAmplitudes().size() -1;
      std::vector<int16_t>   RawDigiAmplitudesL;
      std::vector<int16_t>   RawDigiAmplitudesR;
      std::vector<int16_t>::iterator   ptr;
      if (mode=="digi"){
	ptr=std::find(vstrip.begin(),vstrip.end(),firstStrip); 
	if (ptr==vstrip.end())
	  throw cms::Exception("") << "\n Expected Digi not found in detid " << output_iter->id << " strip " << firstStrip << std::endl;
      }
      else{ 
	ptr=vstrip.begin()+firstStrip; //For raw mode vstrip==vadc==vector of digis for all strips in the det
      }
      LogTrace("SiStripClusterInfoProducer") 
	<< " ptr at strip " << ptr-vstrip.begin() << " adc " << *ptr 
	<< "\nelements above " << ptr-vstrip.begin() << std::endl;
      
      //Looking at digis before firstStrip	  
      for (uint16_t istrip=1;istrip<_NEIGH_STRIP_+1;istrip++){
	if (istrip>ptr-vstrip.begin()) //avoid underflow
	  {break;}
	if (mode=="digi")
	  if (firstStrip-istrip!=*(ptr-istrip)) //avoid not contiguous digis
	    {break;}
	if (firstStrip-istrip==lastStrip_previousCluster) //avoid clusters overlapping 
	  {break;}
	RawDigiAmplitudesL.push_back(*(vadc.begin()+(ptr-vstrip.begin())-istrip));
      }
      if (RawDigiAmplitudesL.size())
	cluster_iter->setRawDigiAmplitudesL(RawDigiAmplitudesL);
      
      ptr+=lastStrip-firstStrip;
      LogTrace("SiStripClusterInfoProducer") 
	<< " ptr at strip " << ptr-vstrip.begin() << " adc " << *ptr 
	<< "\nelements below " << vstrip.end()-ptr-1 << std::endl;
      
      //Looking at digis after LastStrip
      for (uint16_t istrip=1;istrip<_NEIGH_STRIP_+1;istrip++){
	if (istrip>vstrip.end()-ptr-1) //avoid overflow
	  {break;}
	if (mode=="digi")
	  if (lastStrip+istrip!=*(ptr+istrip)) //avoid not contiguous digis
	  {break;}
	if (lastStrip+istrip==firstStrip_nextCluster) //avoid clusters overlapping 
	  {break;}
	
	RawDigiAmplitudesR.push_back(*(vadc.begin()+(ptr-vstrip.begin())+istrip));
      }
      if (RawDigiAmplitudesR.size())
	cluster_iter->setRawDigiAmplitudesR(RawDigiAmplitudesR);
      
      if (edm::isDebugEnabled()){
	std::stringstream ss;
	cluster_iter->print(ss);
	LogTrace("SiStripClusterInfoProducer") << "\n Cluster " << ss.str();
      }
    }
  }
}
  
