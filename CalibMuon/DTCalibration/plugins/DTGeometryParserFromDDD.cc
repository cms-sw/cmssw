/** \file
 *
 *  \author S. Bolognesi - INFN To 
 */

#include "DTGeometryParserFromDDD.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

using namespace std;

DTGeometryParserFromDDD::DTGeometryParserFromDDD(
    const DDCompactView* cview,
    const MuonGeometryConstants& muonConstants,
    map<DTLayerId, std::pair<unsigned int, unsigned int> >& theLayerIdWiresMap) {
  try {
    std::string attribute = "MuStructure";
    std::string value = "MuonBarrelDT";

    // Asking only for the Muon DTs
    DDSpecificsMatchesValueFilter filter{DDValue(attribute, value, 0.0)};
    DDFilteredView fview(*cview, filter);

    parseGeometry(fview, muonConstants, theLayerIdWiresMap);
  } catch (const cms::Exception& e) {
    std::cerr << "DTGeometryParserFromDDD::build() : DDD Exception: something went wrong during XML parsing!"
              << std::endl
              << "  Message: " << e << std::endl
              << "  Terminating execution ... " << std::endl;
    throw;
  } catch (const exception& e) {
    std::cerr << "DTGeometryParserFromDDD::build() : an unexpected exception occured: " << e.what() << std::endl;
    throw;
  } catch (...) {
    std::cerr << "DTGeometryParserFromDDD::build() : An unexpected exception occured!" << std::endl
              << "  Terminating execution ... " << std::endl;
    std::unexpected();
  }
}

DTGeometryParserFromDDD::~DTGeometryParserFromDDD() {}

void DTGeometryParserFromDDD::parseGeometry(DDFilteredView& fv,
                                            const MuonGeometryConstants& muonConstants,
                                            map<DTLayerId, std::pair<unsigned int, unsigned int> >& theLayerIdWiresMap) {
  bool doChamber = fv.firstChild();

  // Loop on chambers
  while (doChamber) {
    // Loop on SLs
    bool doSL = fv.firstChild();
    while (doSL) {
      bool doL = fv.firstChild();
      // Loop on SLs
      while (doL) {
        //DTLayer* layer =
        buildLayer(fv, muonConstants, theLayerIdWiresMap);

        fv.parent();
        doL = fv.nextSibling();  // go to next layer
      }                          // layers

      fv.parent();
      doSL = fv.nextSibling();  // go to next SL
    }                           // sls

    fv.parent();
    doChamber = fv.nextSibling();  // go to next chamber
  }                                // chambers
}

void DTGeometryParserFromDDD::buildLayer(DDFilteredView& fv,
                                         const MuonGeometryConstants& muonConstants,
                                         map<DTLayerId, std::pair<unsigned int, unsigned int> >& theLayerIdWiresMap) {
  MuonGeometryNumbering mdddnum(muonConstants);
  DTNumberingScheme dtnum(muonConstants);
  int rawid = dtnum.getDetId(mdddnum.geoHistoryToBaseNumber(fv.geoHistory()));
  DTLayerId layId(rawid);

  // Loop on wires
  bool doWire = fv.firstChild();
  int WCounter = 0;
  int firstWire = fv.copyno();
  while (doWire) {
    WCounter++;
    doWire = fv.nextSibling();  // next wire
  }
  theLayerIdWiresMap[layId] = (make_pair(firstWire, WCounter));
}
