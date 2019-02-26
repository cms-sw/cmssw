#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <cmath>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

//#define EDM_ML_DEBUG

class TestMTDPosition : public edm::one::EDAnalyzer<> {
public:
  explicit TestMTDPosition( const edm::ParameterSet& );
  ~TestMTDPosition() override;

  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

  void theBaseNumber( const DDGeoHistory& gh );

  void checkMTD( const DDCompactView& cpv, std::string fname = "GeoHistory", int nVols = 0 , std::string ddtop_ = "mtd:BarrelTimingLayer" );

private:

  std::string label_;
  bool isMagField_;
  std::string fname_;
  int nNodes_;
  std::string ddTopNodeName_;

  static constexpr double rad2deg = 180./M_PI;

};

TestMTDPosition::TestMTDPosition( const edm::ParameterSet& iConfig ) :
  label_(iConfig.getUntrackedParameter<std::string>("label","")),
  isMagField_(iConfig.getUntrackedParameter<bool>("isMagField",false)),
  fname_(iConfig.getUntrackedParameter<std::string>("outFileName", "GeoHistory")),
  nNodes_(iConfig.getUntrackedParameter<uint32_t>("numNodesToDump", 0)),
  ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "btl:BarrelTimingLayer"))
{
  if ( isMagField_ ) {
    label_ = "magfield";
  }
}

TestMTDPosition::~TestMTDPosition()
{
}

void
TestMTDPosition::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   edm::ESTransientHandle<DDCompactView> pDD;
   if (!isMagField_) {
     iSetup.get<IdealGeometryRecord>().get(label_, pDD );
   } else {
     iSetup.get<IdealMagneticFieldRecord>().get(label_, pDD );
   }
   if (pDD.description()) {
     edm::LogInfo("TestMTDPosition") << pDD.description()->type_ << " label: " << pDD.description()->label_;
   } else {
     edm::LogWarning("TestMTDPosition") << "NO label found pDD.description() returned false.";
   }
   if (!pDD.isValid()) {
     edm::LogError("TestMTDPosition") << "ESTransientHandle<DDCompactView> pDD is not valid!";
   }

   if ( ddTopNodeName_ != "btl:BarrelTimingLayer" && ddTopNodeName_ != "etl:EndcapTimingLayer" ) {
     edm::LogWarning("TestMTDPosition") << ddTopNodeName_ << "Not valid top MTD volume";
     return;
   }
     
   checkMTD( *pDD , fname_ , nNodes_ , ddTopNodeName_ );

}

void TestMTDPosition::checkMTD ( const DDCompactView& cpv, std::string fname, int nVols , std::string ddtop_ ) {

  fname = "dump" + fname;

  DDPassAllFilter filter;
  DDFilteredView fv(cpv, filter);

  edm::LogInfo("TestMTDPosition") << "Top Most LogicalPart = " << fv.logicalPart();

  using nav_type =  DDFilteredView::nav_type;
  using id_type = std::map<nav_type,int>;
  id_type idMap;
  int id=0;
  std::ofstream dump(fname.c_str());
  bool notReachedDepth(true);
  
  bool write = false;
  size_t limit = 0;

  do {
    nav_type pos = fv.navPos();
    idMap[pos]=id;
    
    size_t num = fv.geoHistory().size();

    if ( num <= limit ) { write = false; }
    if ( fv.geoHistory()[num-1].logicalPart().name() == "btl:BarrelTimingLayer" || 
         fv.geoHistory()[num-1].logicalPart().name() == "etl:EndcapTimingLayer" ) {
      limit = num;
      write = true;
    }

    // Actions for MTD volumes: searchg for sensitive detectors

    if ( write && fv.geoHistory()[limit-1].logicalPart().name() == ddtop_ ) { 

      dump << " - " << fv.geoHistory();
      dump << "\n";

      bool isSens = false;

      if ( fv.geoHistory()[num-1].logicalPart().specifics().size() > 0 ) { 
        for ( auto elem : *(fv.geoHistory()[num-1].logicalPart().specifics()[0]) ) {
          if ( elem.second.name() == "SensitiveDetector" ) { isSens = true; break; }
        }
      }
      

      // Check of numbering scheme for sensitive detectors

      if ( isSens ) { 
        
        DDBox mySens = fv.geoHistory()[num-1].logicalPart().solid();
        dump << "Solid shape name: " << DDSolidShapesName::name(mySens.shape()) << "\n";
        if ( static_cast<int>(mySens.shape()) != 1 ) { throw cms::Exception("TestMTDPosition") << "MTD sensitive element not a DDBox"; break; }
        dump << "Box dimensions: " << mySens.halfX() << " " << mySens.halfY() << " " << mySens.halfZ() << "\n";
        
        char buf[256];
        DD3Vector x, y, z;
        fv.rotation().GetComponents(x,y,z);
        size_t s = snprintf(buf, 256, ",%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f",
                            fv.translation().x(),  fv.translation().y(),  fv.translation().z(),
                            x.X(), y.X(), z.X(), 
                            x.Y(), y.Y(), z.Y(),
                            x.Z(), y.Z(), z.Z());
        assert(s < 256);
        dump << "Translation vector and Rotation components: " << buf;
        dump << "\n";

        DD3Vector zeroLocal(0.,0.,0.);
        DD3Vector cn1Local(mySens.halfX(),mySens.halfY(),mySens.halfZ());
        double distLocal = cn1Local.R();
        DD3Vector zeroGlobal = (fv.rotation())(zeroLocal) + fv.translation();
        DD3Vector cn1Global = (fv.rotation())(cn1Local) + fv.translation();;
        double distGlobal = std::sqrt(std::pow(zeroGlobal.X()-cn1Global.X(),2)
                                      +std::pow(zeroGlobal.Y()-cn1Global.Y(),2)
                                      +std::pow(zeroGlobal.Z()-cn1Global.Z(),2));

        dump << "Center global   = " << std::setw(14) << std::fixed << zeroGlobal.X() 
             << std::setw(14) << std::fixed << zeroGlobal.Y() 
             << std::setw(14) << std::fixed << zeroGlobal.Z() 
             << " r = " << std::setw(14) << std::fixed << zeroGlobal.Rho() 
             << " phi = " << std::setw(14) << std::fixed << zeroGlobal.Phi()*rad2deg << "\n";

        dump << "Corner 1 local  = " << std::setw(14) << std::fixed << cn1Local.X() 
             << std::setw(14) << std::fixed << cn1Local.Y() 
             << std::setw(14) << std::fixed << cn1Local.Z() 
             << " DeltaR = " << std::setw(14) << std::fixed << distLocal << "\n";

        dump << "Corner 1 global = " << std::setw(14) << std::fixed << cn1Global.X() 
             << std::setw(14) << std::fixed << cn1Global.Y() 
             << std::setw(14) << std::fixed << cn1Global.Z() 
             << " DeltaR = " << std::setw(14) << std::fixed << distGlobal << "\n";

        dump << "\n";
        if ( std::fabs(distGlobal - distLocal) > 1.e-6 ) { dump << "DIFFERENCE IN DISTANCE \n"; }
        
      }

    }
    ++id;
    if ( nVols != 0 && id > nVols ) notReachedDepth = false;
  } while (fv.next() && notReachedDepth);
  dump << std::flush;
  dump.close();
}




DEFINE_FWK_MODULE(TestMTDPosition);
