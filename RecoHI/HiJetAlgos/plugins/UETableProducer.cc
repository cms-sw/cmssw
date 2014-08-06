
// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoCaloTowerCandidate.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Ref.h"

#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "CondFormats/HIObjects/interface/UETable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"


using namespace std;
//
// class decleration
//

class UETableProducer : public edm::EDAnalyzer {
   public:
      explicit UETableProducer(const edm::ParameterSet&);
      ~UETableProducer();

   private:
      virtual void beginRun(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------

  bool debug_;

  string calibrationFile_;
   unsigned int runnum_;

  unsigned int index,
    np[5],
    ni0[2],
    ni1[2],
    ni2[2];

  //    ue_interpolation_pf0[15][344],
  //    ue_interpolation_pf1[15][344],
  //    ue_interpolation_pf2[15][82];

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
UETableProducer::UETableProducer(const edm::ParameterSet& iConfig):
   runnum_(0)
{
   //now do what ever initialization is needed
  //  calibrationFile_ = iConfig.getParameter<std::string>("txtFile");
  calibrationFile_ = "RecoHI/HiJetAlgos/data/ue_calibrations_pf_data.txt";

   debug_ = iConfig.getUntrackedParameter<bool>("debug",false);

   np[0] = 3;// Number of reduced PF ID (track, ECAL, HCAL)
   np[1] = 15;// Number of pseudorapidity block
   np[2] = 5;// Fourier series order
   np[3] = 2;// Re or Im
   np[4] = 82;// Number of feature parameter


}

UETableProducer::~UETableProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void
UETableProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // nothing

}

// ------------ method called once each job just before starting event loop  ------------
void 
UETableProducer::beginRun(const edm::EventSetup& iSetup)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
UETableProducer::endJob() {

  edm::FileInPath ueData(calibrationFile_.data());
  std::string qpDataName = ueData.fullPath();
  std::ifstream textTable_(qpDataName.c_str());

  UETable* ue_predictor_pf = new UETable(np[0]);

  std::vector<float> v(np[4]);
  std::vector<std::vector<float> > vv(np[3]);
  std::vector<std::vector<std::vector<float> > > vvv(np[2]);
  std::vector<std::vector<std::vector<std::vector<float> > > > vvvv(np[1]);

  //  unsigned int Nnp_full = np[0] * np[1] * np[2] * np[3] * np[4];
  unsigned int Nnp = np[0] * np[1] * (1 + (np[2] - 1) * np[3]) * np[4];
  unsigned int Nni0 = ni0[0]*ni0[1];
  unsigned int Nni1 = ni1[0]*ni1[1];
  unsigned int Nni2 = ni2[0]*ni2[1];

  for(unsigned int i = 0; i < np[4]; ++i){
    v.push_back(0);
  }

  for(unsigned int i = 0; i < np[3]; ++i){
    vv.push_back(v);
  }

  for(unsigned int i = 0; i < np[2]; ++i){
    vvv.push_back(vv);
  }

  for(unsigned int i = 0; i < np[1]; ++i){
    vvvv.push_back(vvv);
  }

  for(unsigned int i = 0; i < np[0]; ++i){
    ue_predictor_pf->push_back(vvvv);
  }

  std::string line;

  while( std::getline( textTable_, line)){
    if(!line.size() || line[0]=='#') {
      std::cout<<" continue "<<std::endl;
      continue;
    }
    std::istringstream linestream(line);
    float val;
    int bin0, bin1, bin2, bin3, bin4;
    if(index < Nnp){
      linestream>>bin0>>bin1>>bin2>>bin3>>bin4>>val;
      (*ue_predictor_pf)[bin0][bin1][bin2][bin3][bin4] = val;
    }else if(index < Nnp + Nni0){
      linestream>>bin0>>bin1>>val;
      //      ue_interpolation_pf0[bin0][bin1] = val;
    }else if(index < Nnp + Nni0 + Nni1){
      linestream>>bin0>>bin1>>val;
      //      ue_interpolation_pf1[bin0][bin1] = val;
    }else if(index < Nnp + Nni0 + Nni1 + Nni2){
      linestream>>bin0>>bin1>>val;
      //      ue_interpolation_pf2[bin0][bin1] = val;
    }
    ++index;
  }

      edm::Service<cond::service::PoolDBOutputService> pool;
      if( pool.isAvailable() ){
	 if( pool->isNewTagRequest( "HeavyIonRcd" ) ){
	   pool->createNewIOV<UETable>( ue_predictor_pf, pool->beginOfTime(), pool->endOfTime(), "HeavyIonRcd" );
	 }else{
	    pool->appendSinceTime<UETable>( ue_predictor_pf, pool->currentTime(), "HeavyIonRcd" );
	 }

      }
}


//define this as a plug-in
DEFINE_FWK_MODULE(UETableProducer);
