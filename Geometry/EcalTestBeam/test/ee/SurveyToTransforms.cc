#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include <fstream>
#include <iomanip>
#include <iterator>
#include "CLHEP/Units/GlobalSystemOfUnits.h"  
#include "TH1.h"
#include "TH1D.h"
#include "TProfile.h"

using namespace CLHEP;

class SurveyToTransforms : public edm::one::EDAnalyzer<> {
public:
  SurveyToTransforms( const edm::ParameterSet& );
  ~SurveyToTransforms();

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  int pass_;

  edm::Service<TFileService> h_fs;

  TProfile* h_eta;
  TProfile* h_phi;

  TH1D* h_diffs[10][12];
};

SurveyToTransforms::SurveyToTransforms( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed
  pass_=0;
  //  fullEcalDump_=iConfig.getUntrackedParameter<bool>("fullEcalDump",false);
  h_eta = h_fs->make<TProfile>("iEta", "Eta vs iEta", 86*2*4, -86, 86, " " ) ;
  h_phi = h_fs->make<TProfile>("iPhi", "Phi vs iPhi", 360*4, 1, 361, " " ) ;

  const std::string hname[10] = { "EB", "EE", "ES", "HB", "HO", "HE", "HF", "CT", "ZD", "CA" } ;
  const std::string cname[12] = { "XCtr", "YCtr", "ZCtr",
				  "XCor0", "YCor0", "ZCor0",
				  "XCor3", "YCor3", "ZCor3",
				  "XCor6", "YCor6", "ZCor6" } ;

  for( unsigned int i ( 0 ) ; i != 10 ; ++i )
  {
     for( unsigned int j ( 0 ) ; j != 12 ; ++j )
     {
	h_diffs[i][j] = h_fs->make<TH1D>( std::string( hname[i] + cname[j] +
						       std::string("Diff (microns)") ).c_str(), 
					  std::string( hname[i] +
						       std::string(": New-Nom(")
						       + cname[j] + std::string(")") ).c_str(), 
					  200, -200., 200. ) ;
     }
  }
}


SurveyToTransforms::~SurveyToTransforms()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
SurveyToTransforms::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;
   //

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     

   pG->getSubdetectorGeometry( DetId::Ecal, EcalEndcap ) ;
}

//define this as a plug-in

DEFINE_FWK_MODULE(SurveyToTransforms);
