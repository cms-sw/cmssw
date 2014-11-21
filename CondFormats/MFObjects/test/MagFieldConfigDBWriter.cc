/** \class MagFieldConfigDBWriter
 *
 *  Write the MF configuration object created from pset to a DB
 *
 *  \author N. Amapane - Torino
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/MFObjects/interface/MagFieldConfig.h"


class MagFieldConfigDBWriter : public edm::EDAnalyzer {
 public:
  /// Constructor
  MagFieldConfigDBWriter(const edm::ParameterSet& pset);

  /// Destructor
  ~MagFieldConfigDBWriter();
  
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& setup){};

  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup){}
 
  virtual void endJob();
  
 private:
  MagFieldConfig* conf;

};

MagFieldConfigDBWriter::MagFieldConfigDBWriter(const edm::ParameterSet& pset){
  conf = new MagFieldConfig(pset, false);  
}

MagFieldConfigDBWriter::~MagFieldConfigDBWriter(){
  delete conf;
}


void MagFieldConfigDBWriter::endJob() {
  std::string record = "MagFieldConfigRcd";  
    // Write the ttrig object to DB
  edm::Service<cond::service::PoolDBOutputService> dbOutputSvc;
  if(dbOutputSvc.isAvailable()){
    try{
      if(dbOutputSvc->isNewTagRequest(record)){
	//create mode
	dbOutputSvc->writeOne<MagFieldConfig>(conf, dbOutputSvc->beginOfTime(),record);
      }else{
	//append mode. Note: correct PoolDBESSource must be loaded
	dbOutputSvc->writeOne<MagFieldConfig>(conf, dbOutputSvc->currentTime(),record);
      }
    }catch(const cond::Exception& er){
      std::cout << er.what() << std::endl;
    }catch(const std::exception& er){
      std::cout << "[MagFieldConfigDBWriter] caught std::exception " << er.what() << std::endl;
    }catch(...){
      std::cout << "[MagFieldConfigDBWriter] Unexpected exception" << std::endl;
    }
  }else{
    std::cout << "Service PoolDBOutputService is unavailable" << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MagFieldConfigDBWriter);                                                                                       
