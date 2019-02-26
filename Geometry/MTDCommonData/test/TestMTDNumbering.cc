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

#include "DetectorDescription/Core/interface/DDFilteredView.h"

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
  uint32_t theLayout_;

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
  theLayout_(iConfig.getUntrackedParameter<uint32_t>("theLayout", 1)),
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

  DDPassAllFilter filter;
  DDFilteredView fv(cpv, filter);

  edm::LogInfo("TestMTDNumbering") << "Top Most LogicalPart = " << fv.logicalPart();

  using nav_type =  DDFilteredView::nav_type;
  using id_type = std::map<nav_type,int>;
  id_type idMap;
  int id=0;
  std::ofstream dump(fname.c_str());
  bool notReachedDepth(true);
  
  bool write = false;
  bool isBarrel = true;
  size_t limit = 0;

  do {
    nav_type pos = fv.navPos();
    idMap[pos]=id;

    size_t num = fv.geoHistory().size();

    if ( num <= limit ) { write = false; }
    if ( fv.geoHistory()[num-1].logicalPart().name() == "btl:BarrelTimingLayer" ) {
      isBarrel = true;
      limit = num;
      write = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
    }
    else if ( fv.geoHistory()[num-1].logicalPart().name() == "etl:EndcapTimingLayer" ) {
      isBarrel = false;
      limit = num;
      write = true;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("TestMTDNumbering") << "isBarrel = " << isBarrel;
#endif
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

        theBaseNumber( fv.geoHistory() );

        if ( isBarrel ) { 
          BTLDetId::CrysLayout lay = static_cast< BTLDetId::CrysLayout >(theLayout_);
          BTLDetId theId(btlNS_.getUnitID(thisN_)); 
          int hIndex = theId.hashedIndex( lay );
          BTLDetId theNewId( theId.getUnhashedIndex( hIndex ,  lay ) );
          dump << theId; 
          dump << "\n layout type = " << static_cast< int >(lay);
          dump << "\n ieta        = " << theId.ieta( lay );
          dump << "\n iphi        = " << theId.iphi( lay );
          dump << "\n hashedIndex = " << theId.hashedIndex( lay );
          dump << "\n BTLDetId hI = " << theNewId;
          if ( theId.mtdSide() != theNewId.mtdSide() ) { dump << "\n DIFFERENCE IN SIDE"; }
          if ( theId.mtdRR() != theNewId.mtdRR() ) { dump << "\n DIFFERENCE IN ROD"; }
          if ( theId.module() != theNewId.module() ) { dump << "\n DIFFERENCE IN MODULE"; }
          if ( theId.modType() != theNewId.modType() ) { dump << "\n DIFFERENCE IN MODTYPE"; }
          if ( theId.crystal() != theNewId.crystal() ) { dump << "\n DIFFERENCE IN CRYSTAL"; }
          dump << "\n";
        }
        else { ETLDetId theId(etlNS_.getUnitID(thisN_)); dump << theId; }
        dump << "\n";

      }

    }
    ++id;
    if ( nVols != 0 && id > nVols ) notReachedDepth = false;
  } while (fv.next() && notReachedDepth);
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
