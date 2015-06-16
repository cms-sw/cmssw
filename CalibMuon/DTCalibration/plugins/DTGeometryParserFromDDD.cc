/** \file
 *
 *  \author S. Bolognesi - INFN To 
 */

#include "DTGeometryParserFromDDD.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

using namespace std;


DTGeometryParserFromDDD::DTGeometryParserFromDDD(const DDCompactView* cview, const MuonDDDConstants& muonConstants, map<DTLayerId,std::pair<unsigned int,unsigned int> > &theLayerIdWiresMap ){

  try {
    std::string attribute = "MuStructure"; 
    std::string value     = "MuonBarrelDT";
    DDValue val(attribute, value, 0.0);

    // Asking only for the Muon DTs
    DDSpecificsFilter filter;
    filter.setCriteria(val,  // name & value of a variable 
		       DDCompOp::matches,
		       DDLogOp::AND, 
		       true, // compare strings otherwise doubles
		       true  // use merged-specifics or simple-specifics
		       );
    DDFilteredView fview(*cview);
    fview.addFilter(filter);

    parseGeometry(fview, muonConstants, theLayerIdWiresMap);
  }
  catch (const cms::Exception & e ) {
    std::cerr << "DTGeometryParserFromDDD::build() : DDD Exception: something went wrong during XML parsing!" << std::endl
	      << "  Message: " << e << std::endl
	      << "  Terminating execution ... " << std::endl;
    throw;
  }
  catch (const exception & e) {
    std::cerr << "DTGeometryParserFromDDD::build() : an unexpected exception occured: " << e.what() << std::endl; 
    throw;
  }
  catch (...) {
    std::cerr << "DTGeometryParserFromDDD::build() : An unexpected exception occured!" << std::endl
	      << "  Terminating execution ... " << std::endl;
    std::unexpected();           
  }
}

DTGeometryParserFromDDD::~DTGeometryParserFromDDD(){
}
 
void DTGeometryParserFromDDD::parseGeometry(DDFilteredView& fv, const MuonDDDConstants& muonConstants, map<DTLayerId,std::pair<unsigned int,unsigned int> > &theLayerIdWiresMap ) {

  bool doChamber = fv.firstChild();

  // Loop on chambers
  int ChamCounter=0;
  while (doChamber){
    ChamCounter++;
  
    // Loop on SLs
    bool doSL = fv.firstChild();
    int SLCounter=0;
    while (doSL) {
      SLCounter++;
    
      bool doL = fv.firstChild();
      int LCounter=0;
      // Loop on SLs
      while (doL) {
        LCounter++;
        //DTLayer* layer = 
	buildLayer(fv, muonConstants, theLayerIdWiresMap);

        fv.parent();
        doL = fv.nextSibling(); // go to next layer
      } // layers

      fv.parent();
      doSL = fv.nextSibling(); // go to next SL
    } // sls

    fv.parent();
    doChamber = fv.nextSibling(); // go to next chamber
  } // chambers

}


void DTGeometryParserFromDDD::buildLayer(DDFilteredView& fv, const MuonDDDConstants& muonConstants, 
					 map<DTLayerId,std::pair<unsigned int,unsigned int> > &theLayerIdWiresMap ) {
  MuonDDDNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTLayerId layId(rawid);

  // Loop on wires
  bool doWire = fv.firstChild();
  int WCounter=0;
  int firstWire=fv.copyno();
  while (doWire) {
    WCounter++;
    doWire = fv.nextSibling(); // next wire
  }
  theLayerIdWiresMap[layId] = (make_pair(firstWire,WCounter));
}

