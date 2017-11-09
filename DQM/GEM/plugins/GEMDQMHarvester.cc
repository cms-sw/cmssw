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
#include <stdlib.h>
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
  virtual ~GEMDQMHarvester();
  
 
//   virtual void beginJob(){return;};
//   
//   virtual void endJob(){return;};  
//  
//   virtual void analyze(const edm::Event&, const edm::EventSetup&){return;};
//   
//   virtual void endRun(const edm::Run&, const edm::EventSetup&){return;};
  
protected:
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &) override {}
  
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, const edm::LuminosityBlock &, const edm::EventSetup &) override;

  
private:

	void prova(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const & iSetup);
	std::string fName;
	int verbosity;
	DQMStore *dbe;
	
    const GEMGeometry* initGeometry(edm::EventSetup const & iSetup);
    int findVFAT(float min_, float max_, float x_, int roll_);
     
    const GEMGeometry* GEMGeometry_; 

    std::vector<GEMChamber> gemChambers;
    
    int nCh;
    
    std::unordered_map<UInt_t,  MonitorElement*> Eff_Strips_vs_eta;


};

int GEMDQMHarvester::findVFAT(float min_, float max_, float x_, int roll_) {
  float step = abs(max_-min_)/3.0;
  if ( x_ < (min(min_,max_)+step) ) { return 8 - roll_;}
  else if ( x_ < (min(min_,max_)+2.0*step) ) { return 16 - roll_;}
  else { return 24 - roll_;}
}

const GEMGeometry* GEMDQMHarvester::initGeometry(edm::EventSetup const & iSetup) {
  const GEMGeometry* GEMGeometry_ = nullptr;
  try {
    edm::ESHandle<GEMGeometry> hGeom;
    iSetup.get<MuonGeometryRecord>().get(hGeom);
    GEMGeometry_ = &*hGeom;
  }
  catch( edm::eventsetup::NoProxyException<GEMGeometry>& e) {
    edm::LogError("MuonGEMBaseValidation") << "+++ Error : GEM geometry is unavailable on event loop. +++\n";
    return nullptr;
  }

  return GEMGeometry_;
}


GEMDQMHarvester::GEMDQMHarvester(const edm::ParameterSet& ps)
{
//   fName = ps.getUntrackedParameter<std::string>("Name");

  //dbe_path_ = std::string("GEMDQM/");
  //outputFile_ = ps.getUntrackedParameter<std::string>("outputFile", "myfile.root");
}

GEMDQMHarvester::~GEMDQMHarvester()
{

}

void GEMDQMHarvester::prova(DQMStore::IBooker &ibooker, edm::Run const &, edm::EventSetup const & iSetup)
{

  GEMGeometry_ = initGeometry(iSetup);
  if ( GEMGeometry_ == nullptr) return ;  

  const std::vector<const GEMSuperChamber*>& superChambers_ = GEMGeometry_->superChambers();   
  for (auto sch : superChambers_){
    int n_lay = sch->nChambers();
    for (int l=0;l<n_lay;l++){
      gemChambers.push_back(*sch->chamber(l+1));
    }
  }
  nCh = gemChambers.size();
  ibooker.cd();
  ibooker.setCurrentFolder("GEM/eff");
  for (auto ch : gemChambers){
    GEMDetId gid = ch.id();
    string hName_eff = "Eff_Strip_Gemini_"+to_string(gid.superChamberId())+"_la_"+to_string(gid.layer());
    string hTitle_eff = "Eff Strips Gemini ID : "+to_string(gid.superChamberId())+", layer : "+to_string(gid.layer());
    Eff_Strips_vs_eta[ ch.id() ] = ibooker.book2D(hName_eff, hTitle_eff, 384, 0.5, 384.5, 8, 0.5,8.5);
    TH2F *hist_2 = Eff_Strips_vs_eta[ ch.id() ]->getTH2F();
    hist_2->SetMarkerStyle(20);
    hist_2->SetMarkerSize(0.5);
  }

// 	MonitorElement* eta_1 = igetter.get("/GEM/testEta"); 
// 	MonitorElement* eta_2 = igetter.get("/GEM/testEta_2"); 
// 	MonitorElement* eff = igetter.get("/GEM/prova/eff");
// 	
// 	for(int i = 0; i < eta_1->getNbinsX(); i++){
// 		if(eta_2->getBinContent(i) == 0)
// 			eff->setBinContent(i, 0);
// 		else{
// 			double r = eta_1->getBinContent(i) / eta_2->getBinContent(i);
// 			eff->setBinContent(i, r);
// 		}
// 	}
}

void GEMDQMHarvester::dqmEndLuminosityBlock(DQMStore::IBooker &ibooker, DQMStore::IGetter &igetter, const edm::LuminosityBlock &, const edm::EventSetup &)
{
}

//void GEMDQMHarvestor::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter &ig )
//{
  //ig.setCurrentFolder(dbe_path_.c_str());

//}
DEFINE_FWK_MODULE(GEMDQMHarvester);