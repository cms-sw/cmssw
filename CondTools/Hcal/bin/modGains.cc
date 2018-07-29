#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>
#include <memory>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

class modGains : public edm::one::EDAnalyzer<edm::one::WatchRuns> {

public:
  explicit modGains(const edm::ParameterSet&);
  ~modGains() override;

private:
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  std::string s_operation;
  std::string fileIn, fileOut, fileCorr;
  double      val;
  bool        vectorop;
};

modGains::modGains(const edm::ParameterSet& iConfig) : vectorop(false) {
  s_operation = iConfig.getUntrackedParameter<std::string>("Operation");
  fileIn      = iConfig.getUntrackedParameter<std::string>("FileIn");
  fileOut     = iConfig.getUntrackedParameter<std::string>("FileOut");
  fileCorr    = iConfig.getUntrackedParameter<std::string>("FileCorr");
  val         = iConfig.getUntrackedParameter<double>("ScalarFactor");

  if ( (std::strcmp(s_operation.c_str(),"add")==0) || 
       (std::strcmp(s_operation.c_str(),"sub")==0) || 
       (std::strcmp(s_operation.c_str(),"mult")==0) || 
       (std::strcmp(s_operation.c_str(),"div")==0) ) { // vector operation
    vectorop = true;
  }  else if ((std::strcmp(s_operation.c_str(),"sadd")==0) || 
	      (std::strcmp(s_operation.c_str(),"ssub")==0) || 
	      (std::strcmp(s_operation.c_str(),"smult")==0) || 
	      (std::strcmp(s_operation.c_str(),"sdiv")==0)) {// scalar operation
    std::cerr << "Scalar operation: using val=" << val << std::endl;
  } else {
    throw cms::Exception("Unknown", "modGains") << "Unknown operator. "
						<< s_operation <<" Stopping.\n";
  }
}

modGains::~modGains() { }

void modGains::analyze(edm::Event const&, edm::EventSetup const& iSetup) {

  edm::ESHandle<HcalTopology> htopo;
  iSetup.get<HcalRecNumberingRecord>().get(htopo);
  HcalTopology topo = (*htopo);

  // get base conditions
  std::cerr << fileIn << std::endl;
  std::ifstream inStream  (fileIn.c_str());
  HcalGains gainsIn(&topo);;
  HcalDbASCIIIO::getObject (inStream, &gainsIn);
  inStream.close();

  HcalRespCorrs corrsIn(&topo);;
  if (vectorop) {
    std::ifstream inCorr     (fileCorr.c_str());
    HcalDbASCIIIO::getObject (inCorr, &corrsIn);
    inCorr.close();
  }

  HcalGains gainsOut(&topo);;
  std::vector<DetId> channels = gainsIn.getAllChannels ();
  std::cerr << "size = " << channels.size() << std::endl;
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];

    if (vectorop) { // vector operation
      if ((std::strcmp(s_operation.c_str(),"mult")==0)||(std::strcmp(s_operation.c_str(),"div")==0)) val = 1.0; // mult,div
      if ((std::strcmp(s_operation.c_str(),"add")==0)||(std::strcmp(s_operation.c_str(),"sub")==0)) val = 0.0; // add,sub
      if (corrsIn.exists(id))  {
	val = corrsIn.getValues(id)->getValue();
      }
      if (i%100 == 0)
	std::cerr << "Vector operation, " << i << "th channel: using val=" << val << std::endl;
    }
    
    std::unique_ptr<HcalGain> p_item;
    if ((std::strcmp(s_operation.c_str(),"add")==0) || (std::strcmp(s_operation.c_str(),"sadd")==0))
      p_item = std::make_unique<HcalGain>(id, gainsIn.getValues(id)->getValue(0) + val, gainsIn.getValues(id)->getValue(1) + val, 
			    gainsIn.getValues(id)->getValue(2) + val, gainsIn.getValues(id)->getValue(3) + val);

    if ((std::strcmp(s_operation.c_str(),"sub")==0) || (std::strcmp(s_operation.c_str(),"ssub")==0))
      p_item = std::make_unique<HcalGain>(id, gainsIn.getValues(id)->getValue(0) - val, gainsIn.getValues(id)->getValue(1) - val, 
			    gainsIn.getValues(id)->getValue(2) - val, gainsIn.getValues(id)->getValue(3) - val);
    
    if ((std::strcmp(s_operation.c_str(),"mult")==0) || (std::strcmp(s_operation.c_str(),"smult")==0))
      p_item = std::make_unique<HcalGain>(id, gainsIn.getValues(id)->getValue(0) * val, gainsIn.getValues(id)->getValue(1) * val, 
			    gainsIn.getValues(id)->getValue(2) * val, gainsIn.getValues(id)->getValue(3) * val);

    if ((std::strcmp(s_operation.c_str(),"div")==0) || (std::strcmp(s_operation.c_str(),"sdiv")==0))
      p_item = std::make_unique<HcalGain>(id, gainsIn.getValues(id)->getValue(0) / val, gainsIn.getValues(id)->getValue(1) / val, 
			    gainsIn.getValues(id)->getValue(2) / val, gainsIn.getValues(id)->getValue(3) / val);


    // for all
    if (p_item)  gainsOut.addValues(*p_item);
    //    std::cerr << i << std::endl;
  }
  // write out
  std::ofstream outStream (fileOut.c_str());
  HcalDbASCIIIO::dumpObject (outStream, gainsOut);
  outStream.close();
}

//define this as a plug-in
DEFINE_FWK_MODULE(modGains);
