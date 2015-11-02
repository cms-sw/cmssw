// system include files
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondFormats/DataRecord/interface/HeavyIonUERcd.h"
#include "CondFormats/HIObjects/interface/UETable.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

// For DB entry using JetCorrector to store the vector of float
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h" 

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
  bool jetCorrectorFormat_;

  string calibrationFile_;
  unsigned int runnum_;

  unsigned int index = 0,
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
  calibrationFile_ = iConfig.getParameter<std::string>("txtFile");
  //calibrationFile_ = "RecoHI/HiJetAlgos/data/ue_calibrations_pf_data.txt";

  debug_ = iConfig.getUntrackedParameter<bool>("debug",false);
  jetCorrectorFormat_ = iConfig.getUntrackedParameter<bool>("jetCorrectorFormat",false);

  np[0] = 3;// Number of reduced PF ID (track, ECAL, HCAL)
  np[1] = 15;// Number of pseudorapidity block
  np[2] = 5;// Fourier series order
  np[3] = 2;// Re or Im
  np[4] = 82;// Number of feature parameter

  ni0[0] = np[1];
  ni0[1] = 344;// Number of track binning (kept = ECAL)

  ni1[0] = np[1];
  ni1[1] = 344;// Number of ECAL binning

  ni2[0] = np[1];
  ni2[1] = 82;// Number of HCAL binning

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
  std::string qpDataName = calibrationFile_.c_str();
  std::ifstream textTable_(qpDataName.c_str());

  std::vector<float> ue_vec;
  UETable *ue_predictor_pf = NULL;

  if (!jetCorrectorFormat_) {
	ue_predictor_pf = new UETable();
  }
  //  unsigned int Nnp_full = np[0] * np[1] * np[2] * np[3] * np[4];
  unsigned int Nnp = np[0] * np[1] * (1 + (np[2] - 1) * np[3]) * np[4];
  unsigned int Nni0 = ni0[0]*ni0[1];
  unsigned int Nni1 = ni1[0]*ni1[1];
  unsigned int Nni2 = ni2[0]*ni2[1];

  if (!jetCorrectorFormat_) {
  ue_predictor_pf->np.resize(5);
  ue_predictor_pf->ni0.resize(2);
  ue_predictor_pf->ni1.resize(2);
  ue_predictor_pf->ni2.resize(2);

  std::copy(np, np + 5, ue_predictor_pf->np.begin());
  std::copy(ni0, ni0 + 2, ue_predictor_pf->ni0.begin());
  std::copy(ni1, ni1 + 2, ue_predictor_pf->ni1.begin());
  std::copy(ni2, ni2 + 2, ue_predictor_pf->ni2.begin());

  static const float edge_pseudorapidity[16] = {
	-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191
  };

  ue_predictor_pf->edgeEta.resize(16);

  std::copy(edge_pseudorapidity, edge_pseudorapidity + 16, ue_predictor_pf->edgeEta.begin());

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
	  ue_vec.push_back(val);
    }else if(index < Nnp + Nni0){
      linestream>>bin0>>bin1>>val;
	  ue_vec.push_back(val);
    }else if(index < Nnp + Nni0 + Nni1){
      linestream>>bin0>>bin1>>val;
	  ue_vec.push_back(val);
    }else if(index < Nnp + Nni0 + Nni1 + Nni2){
      linestream>>bin0>>bin1>>val;
	  ue_vec.push_back(val);
    }
    ++index;

  }

  edm::Service<cond::service::PoolDBOutputService> pool;

  if( pool.isAvailable() ){

  if (jetCorrectorFormat_) {
    // A minimal dummy line that satisfies the JME # token >= 6 requirement, and has the correct key type
    JetCorrectorParameters::Definitions definition("1 0 0 0 Correction L1Offset");
    std::vector<JetCorrectorParameters::Record> record(1, JetCorrectorParameters::Record(ue_vec.size(), std::vector<float>(ue_vec.size(), 0), std::vector<float>(ue_vec.size(), 0), ue_vec));
    JetCorrectorParameters parameter(definition, record);

    JetCorrectorParametersCollection *jme_payload = new JetCorrectorParametersCollection();

    jme_payload->push_back(JetCorrectorParametersCollection::L1Offset, parameter);

    if( pool->isNewTagRequest( "JetCorrectionsRecord" ) ){
      pool->createNewIOV<JetCorrectorParametersCollection>( jme_payload, pool->beginOfTime(), pool->endOfTime(), "JetCorrectionsRecord" );
    }else{
      pool->appendSinceTime<JetCorrectorParametersCollection>( jme_payload, pool->currentTime(), "JetCorrectionsRecord" );
    }
  }
  else {
    ue_predictor_pf->values = ue_vec;

    if( pool->isNewTagRequest( "HeavyIonUERcd" ) ){
      pool->createNewIOV<UETable>( ue_predictor_pf, pool->beginOfTime(), pool->endOfTime(), "HeavyIonUERcd" );
    }else{
      pool->appendSinceTime<UETable>( ue_predictor_pf, pool->currentTime(), "HeavyIonUERcd" );
    }
  }
  }
}


//define this as a plug-in
DEFINE_FWK_MODULE(UETableProducer);
