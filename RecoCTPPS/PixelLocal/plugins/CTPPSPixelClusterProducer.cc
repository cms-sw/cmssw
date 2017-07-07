
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoCTPPS/PixelLocal/interface/CTPPSPixelClusterProducer.h"

CTPPSPixelClusterProducer::CTPPSPixelClusterProducer(const edm::ParameterSet& conf) :
  param_(conf) ,
  clusterizer_(conf){
  
  src_ = conf.getParameter<std::string>("label");
  verbosity_ = conf.getUntrackedParameter<int> ("RPixVerbosity");
	 
  tokenCTPPSPixelDigi_ = consumes<edm::DetSetVector<CTPPSPixelDigi> >(edm::InputTag(src_));
 
  produces<edm::DetSetVector<CTPPSPixelCluster> > ();
  }

CTPPSPixelClusterProducer::~CTPPSPixelClusterProducer() {

}

void CTPPSPixelClusterProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions){
  edm::ParameterSetDescription desc;
  desc.addUntracked<int>("RPixVerbosity",0);
  desc.add<std::string>("label", "ctppsPixelDigis");
  desc.add<int>("SeedADCThreshold",15);
  desc.add<int>("ADCThreshold",10);
  desc.add<double>("ElectronADCGain",135.0);
  desc.add<int>("VCaltoElectronGain",50);
  desc.add<int>("VCaltoElectronOffset",-411);
  desc.add<bool>("doSingleCalibration",false); 
 descriptions.add("ctppsPixelClusters", desc);
}

void CTPPSPixelClusterProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  
/// get inputs
  edm::Handle<edm::DetSetVector<CTPPSPixelDigi> > rpd;
  iEvent.getByToken(tokenCTPPSPixelDigi_, rpd);
 
// get analysis mask to mask channels
  edm::ESHandle<CTPPSPixelAnalysisMask> aMask;

  if(rpd->size())
    iSetup.get<CTPPSPixelAnalysisMaskRcd>().get(aMask);
  
  edm::DetSetVector<CTPPSPixelCluster>  output;

// run clusterisation
  if (rpd->size()){
// get calibration DB
    theGainCalibrationDB.getDB(iEvent,iSetup);
    run(*rpd, output, aMask.product());
  }
// write output
  iEvent.put(std::make_unique<edm::DetSetVector<CTPPSPixelCluster> >(output));

}

void CTPPSPixelClusterProducer::run(const edm::DetSetVector<CTPPSPixelDigi> &input, 
				    edm::DetSetVector<CTPPSPixelCluster> &output, const CTPPSPixelAnalysisMask * mask){

  for (const auto &ds_digi : input)
    {
      edm::DetSet<CTPPSPixelCluster> &ds_cluster = output.find_or_insert(ds_digi.id);
      clusterizer_.buildClusters(ds_digi.id, ds_digi.data, ds_cluster.data, theGainCalibrationDB.getCalibs(), mask);

      if(verbosity_){
	unsigned int cluN=0;
	for(std::vector<CTPPSPixelCluster>::iterator iit = ds_cluster.data.begin(); iit != ds_cluster.data.end(); iit++){
	  edm::LogInfo("CTPPSPixelClusterProducer") << "Cluster " << ++cluN <<" avg row " 
					       << (*iit).avg_row()<< " avg col " << (*iit).avg_col()<<" ADC.size " << (*iit).size();
	}
      }
    }
}

DEFINE_FWK_MODULE( CTPPSPixelClusterProducer);
