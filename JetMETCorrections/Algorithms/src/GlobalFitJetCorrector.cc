#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "CondFormats/JetMETObjects/interface/GlobalFitCorrector.h"
#include "JetMETCorrections/Algorithms/interface/GlobalFitJetCorrector.h"


GlobalFitJetCorrector::GlobalFitJetCorrector(const edm::ParameterSet& cfg, const edm::EventSetup& setup)
{
 std::string file="CondFormats/JetMETObjects/data/"+cfg.getParameter<std::string>("tagName")+".txt";
 edm::FileInPath f1(file);

 edm::ESHandle<CaloGeometry> geom;   
 setup.get<IdealGeometryRecord>().get(geom);  
 const CaloSubdetectorGeometry* towerGeom = geom->getSubdetectorGeometry(DetId::Calo, CaloTowerDetId::SubdetId); 
 corrector_ = new GlobalFitCorrector(towerGeom, f1.fullPath());
}

GlobalFitJetCorrector::~GlobalFitJetCorrector()
{
  delete corrector_;
} 

double GlobalFitJetCorrector::correction(const reco::Jet& jet) const 
{
  const reco::CaloJet* caloJet = dynamic_cast<const reco::CaloJet*>(&jet);
  if( caloJet!=0 )
    return corrector_->correction(*caloJet);
  else
    return 1.;
}

double GlobalFitJetCorrector::correction(const reco::Jet& jet, 
					 const edm::Event& event, 
					 const edm::EventSetup& setup) const 
{
  return correction( jet );
}
