#include <DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h>

#include <DetectorDescription/Core/interface/DDValue.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/Core/interface/DDPartSelection.h>
#include "DetectorDescription/Core/interface/DDName.h"

#include <fstream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>
#include <set>
#include <cassert>

GeometryInfoDump::GeometryInfoDump () { }

GeometryInfoDump::~GeometryInfoDump () { }
  

void GeometryInfoDump::dumpInfo ( bool dumpHistory, bool dumpSpecs, bool dumpPosInfo
				  , const DDCompactView& cpv, std::string fname, int nVols ) {
  fname = "dump" + fname;
  DDExpandedView epv(cpv);
  std::cout << "Top Most LogicalPart =" << epv.logicalPart() << std::endl;
  if ( dumpHistory || dumpPosInfo) {
    if ( dumpPosInfo ) {
      std::cout << "After the GeoHistory in the output file dumpGeoHistoryOnRead you will see x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33" << std::endl;
    }
    typedef DDExpandedView::nav_type nav_type;
    typedef std::map<nav_type,int> id_type;
    id_type idMap;
    int id=0;
    std::ofstream dump(fname.c_str());
    bool notReachedDepth(true);
    char buf[256];

    do {
      nav_type pos = epv.navPos();
      idMap[pos]=id;
      //      dump << id 
      dump << " - " << epv.geoHistory();
      DD3Vector x, y, z;
      epv.rotation().GetComponents(x,y,z);
      if ( dumpPosInfo ) {
        size_t s = snprintf(buf, 256, ",%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f",
                            epv.translation().x(),  epv.translation().y(),  epv.translation().z(),
                            x.X(), y.X(), z.X(), 
                            x.Y(), y.Y(), z.Y(),
                            x.Z(), y.Z(), z.Z());
        assert(s < 256);
        dump << buf;
      }
      dump << "\n";;
      ++id;
      if ( nVols != 0 && id > nVols ) notReachedDepth = false;
    } while (epv.next() && notReachedDepth);
    dump << std::flush;
    dump.close();
  }
  if ( dumpSpecs ) {
    // dump specifics at every compact-view nodes to have the most detailed "true" 
    // final destination of the DDSpecifics
    std::string dsname = "dumpSpecs" + fname;
    std::ofstream dump(dsname.c_str());
    DDCompactView::DDCompactView::graph_type gra = cpv.graph();
    std::set<DDLogicalPart> lpStore;
    typedef  DDCompactView::graph_type::const_adj_iterator adjl_iterator;
    adjl_iterator git = gra.begin();
    adjl_iterator gend = gra.end();        
    DDCompactView::graph_type::index_type i=0;
    for (; git != gend; ++git) 
    {
      const DDLogicalPart & ddLP = gra.nodeData(git);
      if ( lpStore.find(ddLP) != lpStore.end() && ddLP.attachedSpecifics().size() != 0 ) {
	dump << ddLP.toString() << ": ";
	dumpSpec( ddLP.attachedSpecifics(), dump );
      }
      lpStore.insert(ddLP);

      ++i;
      if (git->size()) 
	{
	  // ask for children of ddLP  
	  for( const auto& cit : *git ) 
	    {
	      const DDLogicalPart & ddcurLP = gra.nodeData(cit.first);
	      if (lpStore.find(ddcurLP) != lpStore.end() && ddcurLP.attachedSpecifics().size() != 0 ) {
		dump << ddcurLP.toString() << ": ";
		dumpSpec( ddcurLP.attachedSpecifics(), dump );
	      }
	      lpStore.insert(ddcurLP);
	    } // iterate over children
	} // if (children)
    } // iterate over graph nodes  
    dump.close();
   }
 }

void GeometryInfoDump::dumpSpec( const std::vector<std::pair< const DDPartSelection*, const DDsvalues_type*> >& attspec, std::ostream& dump) {
  std::vector<std::pair< const DDPartSelection*, const DDsvalues_type*> >::const_iterator bit(attspec.begin()), eit(attspec.end());
  for( ; bit != eit; ++bit ) {
    //  DDPartSelection is a std::vector<DDPartSelectionLevel>
    std::vector<DDPartSelectionLevel>::const_iterator psit(bit->first->begin()), pseit(bit->first->end());
    for ( ; psit != pseit; ++psit ) {
      switch ( psit->selectionType_ ) {
      case ddunknown:
	throw cms::Exception("DetectorDescriptionSpecPar") << "Can not have an unknown selection type!";
	break;
      case ddanynode:
	dump << "//*";
	break;
      case ddanychild:
	dump << "/*";
	break;
      case ddanylogp:
	dump << "//" << psit->lp_.toString();
	break;
      case ddanyposp:
	dump << "//" << psit->lp_.toString() << "[" << psit->copyno_ << "]" ;
	break;
      case ddchildlogp:
	dump << "/" << psit->lp_.toString();
	break;
      case ddchildposp:
	dump << "/" << psit->lp_.toString() << "[" << psit->copyno_ << "]" ;
	break;
      default:
	throw cms::Exception("DetectorDescriptionSpecPar") << "Can not end up here! default of switch on selectionTyp_";
      }
    }
    dump << " ";
    // DDsvalues_type is typedef std::vector< std::pair<unsigned int, DDValue> > DDsvalues_type;
    DDsvalues_type::const_iterator bsit(bit->second->begin()), bseit(bit->second->end());
    for( ; bsit != bseit; ++bsit ) { 
      dump << bsit->second.name() << " ";
      dump << ( bsit->second.isEvaluated() ?  "eval " : "NOT eval " );
      size_t sdind(0);
      for ( ; sdind != bsit->second.strings().size() ; ++sdind ) {
	if (bsit->second.isEvaluated()) {
	  dump << bsit->second.doubles()[sdind] ;
	} else {
	  dump << bsit->second.strings()[sdind] ;
	}
	if (sdind != bsit->second.strings().size() - 1) dump << ", ";
      }
      if ( bsit->second.strings().size() > 0 && bsit + 1 != bseit ) dump << " | ";
    }
    if ( bit->second->size() > 0 && bit + 1 != eit) dump << " | ";
  }
  dump << std::endl;
}
