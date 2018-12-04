
// system include files
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"

#include "DetectorDescription/Core/interface/DDFilteredView.h"

#include "Geometry/MTDCommonData/interface/MTDBaseNumber.h"
#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"

#include "DataFormats/ForwardDetId/interface/BTLDetId.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//#define EDM_ML_DEBUG

//
// class declaration
//

class MTDTopologyAnalyzer : public edm::one::EDAnalyzer<>
{
public:
  explicit MTDTopologyAnalyzer( const edm::ParameterSet& );
  ~MTDTopologyAnalyzer() override = default;
  
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

  void theBaseNumber( const DDGeoHistory& gh );

  MTDBaseNumber thisN_;
  BTLNumberingScheme btlNS_;
  ETLNumberingScheme etlNS_;

};

MTDTopologyAnalyzer::MTDTopologyAnalyzer( const edm::ParameterSet& iConfig ) :
  thisN_(),btlNS_(),etlNS_()
{}

// ------------ method called to produce the data  ------------
void
MTDTopologyAnalyzer::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup ){

  edm::ESHandle<MTDTopology> mtdTopo;
  iSetup.get<MTDTopologyRcd>().get( mtdTopo );     
  edm::LogInfo("MTDTopologyAnalyzer") << "MTD topology mode = " << mtdTopo->getMTDTopologyMode();
 
  // Build DetIds based on DDD description, then extract information from topology and compare

  std::string label;

  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get(label, pDD );
  if (pDD.description()) {
    edm::LogInfo("MTDTopologyAnalyzer") << pDD.description()->type_ << " label: " << pDD.description()->label_;
  } else {
    edm::LogWarning("MTDTopologyAnalyzer") << "NO label found pDD.description() returned false.";
  }
  if (!pDD.isValid()) {
    edm::LogError("MTDTopologyAnalyzer") << "ESTransientHandle<DDCompactView> pDD is not valid!";
  }
  
  DDPassAllFilter filter;
  DDFilteredView fv(*pDD, filter);

  edm::LogInfo("MTDTopologyAnalyzer") << "Top Most LogicalPart = " << fv.logicalPart();

  using nav_type =  DDFilteredView::nav_type;
  using id_type = std::map<nav_type,int>;
  id_type idMap;
  int id=0;
  std::string ddtop("");
  size_t limit = 0;

  bool isBarrel = false;
  
  do {
    nav_type pos = fv.navPos();
    idMap[pos]=id;

    size_t num = fv.geoHistory().size();

    if ( fv.geoHistory()[num-1].logicalPart().name() == "btl:BarrelTimingLayer" ) { 
      isBarrel = true;
      limit = num;
      ddtop = "btl:BarrelTimingLayer";
    }
    else if ( fv.geoHistory()[num-1].logicalPart().name() == "etl:EndcapTimingLayer" ) {
      isBarrel = false;
      limit = num;
      ddtop = "etl:EndcapTimingLayer";
    }

    if ( num <= limit && fv.geoHistory()[num-1].logicalPart().name().fullname() != ddtop ) { ddtop.clear(); }

    if ( !ddtop.empty() ) { 

      // Actions for MTD volumes: searchg for sensitive detectors
    
      bool isSens = false; 
    
      if ( fv.geoHistory()[num-1].logicalPart().specifics().size() > 0 ) { 
        for ( auto elem : *(fv.geoHistory()[num-1].logicalPart().specifics()[0]) ) {
          if ( elem.second.name() == "SensitiveDetector" ) { isSens = true; break; }
        }
      }
    
      // Check of numbering scheme for sensitive detectors
    
      if ( isSens ) { 
     
        theBaseNumber( fv.geoHistory() );
  
        edm::LogInfo("MTDTopologyAnalyzer") << fv.geoHistory();

        if ( isBarrel ) { 
          BTLDetId theId(btlNS_.getUnitID(thisN_)); 
          DetId localId( theId.rawId() );
          edm::LogInfo("MTDTopologAnalyzer") << mtdTopo->print(localId) << "\n" << theId;;
        }
        else { 
          ETLDetId theId(etlNS_.getUnitID(thisN_)); 
          DetId localId( theId.rawId() );
          edm::LogInfo("MTDTopologAnalyzer") << mtdTopo->print(localId) << "\n" << theId;;
        }
      }
    }
    
    ++id;
  } while (fv.next());
}

void MTDTopologyAnalyzer::theBaseNumber( const DDGeoHistory& gh ) {

  thisN_.reset();
  thisN_.setSize( gh.size() );

  for ( uint i = gh.size(); i-- > 0; ) {
    std::string name(gh[i].logicalPart().name().fullname());
    int copyN(gh[i].copyno());
    thisN_.addLevel( name, copyN );
#ifdef EDM_ML_DEBUG
    edm::LogInfo("MTDTopologyAnalyzer") << name << " " << copyN;
#endif
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDTopologyAnalyzer);
