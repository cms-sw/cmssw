
/*----------------------------------------------------------------------

Toy EDAnalyzer for testing purposes only.

----------------------------------------------------------------------*/

#include <stdexcept>
#include <iostream>
#include <map>
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "CondTools/DT/test/stubs/DTGeometryDump.h"

#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

namespace edmtest {

  DTGeometryDump::DTGeometryDump(edm::ParameterSet const& p) {
  }

  DTGeometryDump::DTGeometryDump(int i) {
  }

  DTGeometryDump::~DTGeometryDump() {
  }

  void DTGeometryDump::analyze( const edm::Event& e,
                                const edm::EventSetup& context ) {

    using namespace edm::eventsetup;
    edm::ESHandle<DTGeometry> muonGeom;
    context.get<MuonGeometryRecord>().get(muonGeom);

    const std::vector<DTChamber*>& ch_cont = muonGeom->chambers();
    std::vector<DTChamber*>::const_iterator ch_iter = ch_cont.begin();
    std::vector<DTChamber*>::const_iterator ch_iend = ch_cont.end();
    std::vector<DTChamberId> ch_id_cont;
    while ( ch_iter != ch_iend ) {
      const DTChamber* ch_ptr = *ch_iter++;
      DTChamberId ch_id = ch_ptr->id();
      std::vector<DTChamberId>::const_iterator ch_id_iter = ch_id_cont.begin();
      std::vector<DTChamberId>::const_iterator ch_id_iend = ch_id_cont.end();
      bool ch_found = false;
      while ( ch_id_iter   != ch_id_iend ) {
        if ( *ch_id_iter++ == ch_id ) {
          ch_found = true;
	  std::cout << "chamber already found:: "
                    << ch_id.wheel()   << " "
                    << ch_id.sector()  << " "
                    << ch_id.station() << std::endl;
          break;
        }
      }
      if ( !ch_found ) ch_id_cont.push_back( ch_id );
      else             continue;
      const std::vector<const DTSuperLayer*>& sl_cont = ch_ptr->superLayers();
      std::vector<const DTSuperLayer*>::const_iterator sl_iter =
                                                       sl_cont.begin();
      std::vector<const DTSuperLayer*>::const_iterator sl_iend =
                                                       sl_cont.end();
      std::vector<DTSuperLayerId> sl_id_cont;
      while ( sl_iter != sl_iend ) {
        const DTSuperLayer* sl_ptr = *sl_iter++;
        DTSuperLayerId sl_id = sl_ptr->id();
        std::vector<DTSuperLayerId>::const_iterator sl_id_iter =
                                                    sl_id_cont.begin();
        std::vector<DTSuperLayerId>::const_iterator sl_id_iend =
                                                    sl_id_cont.end();
        bool sl_found = false;
        while ( sl_id_iter   != sl_id_iend ) {
          if ( *sl_id_iter++ == sl_id ) {
            sl_found = true;
            std::cout << "superlayer already found: "
                      << sl_id.wheel()      << " "
                      << sl_id.sector()     << " "
                      << sl_id.station()    << " "
                      << sl_id.superlayer() << std::endl;
            break;
          }
        }
        if ( !sl_found ) sl_id_cont.push_back( sl_id );
        else             continue;

        const std::vector<const DTLayer*>& cl_cont = sl_ptr->layers();
        std::vector<const DTLayer*>::const_iterator cl_iter =
                                                    cl_cont.begin();
        std::vector<const DTLayer*>::const_iterator cl_iend =
                                                    cl_cont.end();
        std::vector<DTLayerId> cl_id_cont;
        while ( cl_iter != cl_iend ) {
          const DTLayer* cl_ptr = *cl_iter++;
          DTLayerId cl_id = cl_ptr->id();
          std::vector<DTLayerId>::const_iterator cl_id_iter =
                                                 cl_id_cont.begin();
          std::vector<DTLayerId>::const_iterator cl_id_iend =
                                                 cl_id_cont.end();
          bool cl_found = false;
          while ( cl_id_iter   != cl_id_iend ) {
            if ( *cl_id_iter++ == cl_id ) {
              cl_found = true;
              std::cout << "layer already found: "
                        << cl_id.wheel()      << " "
                        << cl_id.sector()     << " "
                        << cl_id.station()    << " "
                        << cl_id.superlayer() << " "
                        << cl_id.layer()      << std::endl;
              break;
            }
          }
          if ( !cl_found ) cl_id_cont.push_back( cl_id );
          else             continue;

          const DTTopology& topo = cl_ptr->specificTopology();
          int fcell = topo.firstChannel();
          int lcell = topo. lastChannel();
          std::cout << "layer: "
                    << cl_id.wheel()      << " "
                    << cl_id.sector()     << " "
                    << cl_id.station()    << " "
                    << cl_id.superlayer() << " "
                    << cl_id.layer()      << " has cells from "
                    << fcell              << " to "
                    << lcell              << std::endl;
        }

      }

    }

//    cond::Time_t itime=(cond::Time_t)evt.time().value();
//    std::cout << "TIME: " << itime << std::endl;
    std::cout << "TIME: " << e.time().value() << std::endl;

  }
  DEFINE_FWK_MODULE(DTGeometryDump);
}
