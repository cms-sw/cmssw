#include <iostream>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

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

#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"


//#define EDM_ML_DEBUG

class TestMTDNumbering : public edm::one::EDAnalyzer<> {
public:
  explicit TestMTDNumbering( const edm::ParameterSet& );
  ~TestMTDNumbering() override;

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

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;

};

TestMTDNumbering::TestMTDNumbering( const edm::ParameterSet& iConfig ) :
  label_(iConfig.getUntrackedParameter<std::string>("label","")),
  isMagField_(iConfig.getUntrackedParameter<bool>("isMagField",false)),
  fname_(iConfig.getUntrackedParameter<std::string>("outFileName", "GeoHistory")),
  nNodes_(iConfig.getUntrackedParameter<uint32_t>("numNodesToDump", 0)),
  ddTopNodeName_(iConfig.getUntrackedParameter<std::string>("ddTopNodeName", "btl:BarrelTimingLayer")),
  thisN_(),btlNS_(),etlNS_()
{
  if ( isMagField_ ) {
    label_ = "magfield";
  }
}

TestMTDNumbering::~TestMTDNumbering()
{
}

void
TestMTDNumbering::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
   using namespace edm;

   edm::ESTransientHandle<DDCompactView> pDD;
   if (!isMagField_) {
     iSetup.get<IdealGeometryRecord>().get(label_, pDD );
   } else {
     iSetup.get<IdealMagneticFieldRecord>().get(label_, pDD );
   }
   if (pDD.description()) {
     edm::LogInfo("TestMTDNumbering") << pDD.description()->type_ << " label: " << pDD.description()->label_;
   } else {
     edm::LogWarning("TestMTDNumbering") << "NO label found pDD.description() returned false.";
   }
   if (!pDD.isValid()) {
     edm::LogError("TestMTDNumbering") << "ESTransientHandle<DDCompactView> pDD is not valid!";
   }

   if ( ddTopNodeName_ != "btl:BarrelTimingLayer" && ddTopNodeName_ != "etl:EndcapTimingLayer" ) {
     edm::LogWarning("TestMTDNumbering") << ddTopNodeName_ << "Not valid top MTD volume";
     return;
   }
     
   checkMTD( *pDD , fname_ , nNodes_ , ddTopNodeName_ );

}

void TestMTDNumbering::checkMTD ( const DDCompactView& cpv, std::string fname, int nVols , std::string ddtop_ ) {

  fname = "dump" + fname;
  DDExpandedView epv(cpv);
  edm::LogInfo("TestMTDNumbering") << "Top Most LogicalPart = " << epv.logicalPart();
  typedef DDExpandedView::nav_type nav_type;
  typedef std::map<nav_type,int> id_type;
  id_type idMap;
  int id=0;
  std::ofstream dump(fname.c_str());
  bool notReachedDepth(true);
  
  bool write = false;
  bool isBarrel = true;
  size_t limit = 0;

  do {
    nav_type pos = epv.navPos();
    idMap[pos]=id;
    
    size_t num = epv.geoHistory().size();

    if ( epv.geoHistory()[num-1].logicalPart().name() == "btl:BarrelTimingLayer" ) {
      isBarrel = true;
      limit = num;
      write = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
    }
    else if ( epv.geoHistory()[num-1].logicalPart().name() == "etl:EndcapTimingLayer" ) {
      isBarrel = false;
      limit = num;
      write = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
    }

    // Actions for MTD volumes: searchg for sensitive detectors

    if ( write && epv.geoHistory()[limit-1].logicalPart().name() == ddtop_ ) { 

      dump << " - " << epv.geoHistory();
      dump << "\n";

      bool isSens = false;

      if ( epv.geoHistory()[num-1].logicalPart().specifics().size() > 0 ) { 
        for ( auto elem : *(epv.geoHistory()[num-1].logicalPart().specifics()[0]) ) {
          if ( elem.second.name() == "SensitiveDetector" ) { isSens = true; break; }
        }
      }
      

      // Check of numbering scheme for sensitive detectors

      if ( isSens ) { 

        theBaseNumber( epv.geoHistory() );

        if ( isBarrel ) { BTLDetId theId(btlNS_.getUnitID(thisN_)); dump << theId; }
        else { ETLDetId theId(etlNS_.getUnitID(thisN_)); dump << theId; }
        dump << "\n";

      }

    }
    ++id;
    if ( nVols != 0 && id > nVols ) notReachedDepth = false;
  } while (epv.next() && notReachedDepth);
  dump << std::flush;
  dump.close();
}

void TestMTDNumbering::theBaseNumber( const DDGeoHistory& gh ) {

  thisN_.reset();
  thisN_.setSize( gh.size() );

  for ( uint i = gh.size(); i-- > 0; ) {
    std::string name(gh[i].logicalPart().name().fullname());
    int copyN(gh[i].copyno());
    thisN_.addLevel( name, copyN );
#ifdef EDM_ML_DEBUG
    edm::LogInfo("TestMTDNumbering") << name << " " << copyN;
#endif
  }

}


DEFINE_FWK_MODULE(TestMTDNumbering);
