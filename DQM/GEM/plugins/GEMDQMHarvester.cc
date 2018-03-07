#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"

#include <iostream>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

using namespace std;
using namespace edm;

class GEMDQMHarvester: public DQMEDHarvester
{  
public:

  GEMDQMHarvester(const edm::ParameterSet&);
  ~GEMDQMHarvester() override;
    
protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override {}
  
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, const edm::LuminosityBlock &, const edm::EventSetup &) override;

  
private:

	
     

    
    

};



GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet& ps)
{
  //   fName = ps.getUntrackedParameter<std::string>("Name");

  //dbe_path_ = std::string("GEMDQM/");
  //outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "myfile.root");
}

GEMDQMHarvester::~GEMDQMHarvester()
{

}


void GEMDQMHarvester::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, const edm::LuminosityBlock &, const edm::EventSetup &)
{
}

//void GEMDQMHarvestor::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter &ig )
//{
//ig.setCurrentFolder(dbe_path_.c_str());

//}
DEFINE_FWK_MODULE(GEMDQMHarvester);
