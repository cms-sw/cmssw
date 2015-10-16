// -*- C++ -*-
//
// Package:    EcalTBHodoscopeGeometryAnalyzer
// Class:      EcalTBHodoscopeGeometryAnalyzer
// 
/**\class EcalTBHodoscopeGeometryAnalyzer EcalTBHodoscopeGeometryAnalyzer.cc test/EcalTBHodoscopeGeometryAnalyzer/src/EcalTBHodoscopeGeometryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//



// system include files
#include <memory>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//
// class decleration
//

class EcalTBHodoscopeGeometryAnalyzer : public edm::one::EDAnalyzer<>
{
public:
  explicit EcalTBHodoscopeGeometryAnalyzer( const edm::ParameterSet& );
  ~EcalTBHodoscopeGeometryAnalyzer();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:

  void build(const CaloGeometry& cg, DetId::Detector det, int subdetn);

  CLHEP::HepRotation * fromCMStoTB( const double & myEta , const double & myPhi ) const;

  int pass_;

  double eta_;
  double phi_;
  CLHEP::HepRotation * fromCMStoTB_;

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
EcalTBHodoscopeGeometryAnalyzer::EcalTBHodoscopeGeometryAnalyzer( const edm::ParameterSet& iConfig )
{
   //now do what ever initialization is needed
  pass_=0;

  eta_ = iConfig.getUntrackedParameter<double>("eta",0.971226);
  phi_ = iConfig.getUntrackedParameter<double>("phi",0.115052);

  fromCMStoTB_ = fromCMStoTB( eta_ , phi_ );
}


EcalTBHodoscopeGeometryAnalyzer::~EcalTBHodoscopeGeometryAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete fromCMStoTB_;
}


void EcalTBHodoscopeGeometryAnalyzer::build(const CaloGeometry& cg, DetId::Detector det, int subdetn) {
  const CaloSubdetectorGeometry* geom=cg.getSubdetectorGeometry(det,subdetn);
  
  int n=0;
  const std::vector<DetId>& ids=geom->getValidDetIds(det,subdetn);
  for (std::vector<DetId>::const_iterator i=ids.begin(); i!=ids.end(); i++) {
    n++;
    const CaloCellGeometry* cell=geom->getGeometry(*i);
    if (det == DetId::Ecal)
      {
        if (subdetn == EcalLaserPnDiode) 
          {

            CLHEP::Hep3Vector thisCellPos( cell->getPosition().x(), cell->getPosition().y(), cell->getPosition().z() );
            CLHEP::Hep3Vector rotCellPos = (*fromCMStoTB_)*thisCellPos;

            edm::LogInfo("EcalTBGeom") << "Fiber DetId = " << HodoscopeDetId(*i) << " position =  " <<rotCellPos;
          }
      }
  }
}
//
// member functions
//

// ------------ method called to produce the data  ------------
void
EcalTBHodoscopeGeometryAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   std::cout << "Here I am " << std::endl;

   edm::ESHandle<CaloGeometry> pG;
   iSetup.get<CaloGeometryRecord>().get(pG);     
   //
   // get the ecal & hcal geometry
   //

   if (pass_==0) {
     build(*pG,DetId::Ecal,EcalLaserPnDiode);
   }

   pass_++;
      
}


CLHEP::HepRotation * EcalTBHodoscopeGeometryAnalyzer::fromCMStoTB( const double & myEta , const double & myPhi ) const
{

  double myTheta = 2.0*atan(exp(-myEta));

  // rotation matrix to move from the CMS reference frame to the test beam one
  
  CLHEP::HepRotation * CMStoTB = new CLHEP::HepRotation();
  
  double angle1 = 90.*deg - myPhi;
  CLHEP::HepRotationZ * r1 = new CLHEP::HepRotationZ(angle1);
  double angle2 = myTheta;
  CLHEP::HepRotationX * r2 = new CLHEP::HepRotationX(angle2);
  double angle3 = 90.*deg;
  CLHEP::HepRotationZ * r3 = new CLHEP::HepRotationZ(angle3);
  (*CMStoTB) *= (*r3);
  (*CMStoTB) *= (*r2);
  (*CMStoTB) *= (*r1);

  return CMStoTB;

}


//define this as a plug-in

DEFINE_FWK_MODULE(EcalTBHodoscopeGeometryAnalyzer);
