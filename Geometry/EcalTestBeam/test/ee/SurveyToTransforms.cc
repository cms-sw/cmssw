#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
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

class SurveyToTransforms : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  SurveyToTransforms( const edm::ParameterSet& );
  ~SurveyToTransforms() override {}

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::Service<TFileService> h_fs;

  TProfile* h_eta;
  TProfile* h_phi;

  TH1D* h_diffs[10][12];

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geometryToken_;
};

SurveyToTransforms::SurveyToTransforms( const edm::ParameterSet& iConfig )
{
  usesResource("TFileService");

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

  geometryToken_ = esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{});
}

void
SurveyToTransforms::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  const auto& pG = iSetup.getData(geometryToken_);

  pG.getSubdetectorGeometry( DetId::Ecal, EcalEndcap ) ;
}

DEFINE_FWK_MODULE(SurveyToTransforms);
