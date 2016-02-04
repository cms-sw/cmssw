#include <DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h>

#include <DetectorDescription/Core/interface/DDValue.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/Core/interface/DDPartSelection.h>
#include "DetectorDescription/Core/interface/DDName.h"


#include <iostream>
#include <fstream>

// Need thes??? maybe 
#include <cmath>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>
#include <set>

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
    do {
      nav_type pos = epv.navPos();
      idMap[pos]=id;
      //      dump << id 
      dump << " - " << epv.geoHistory();
      DD3Vector x, y, z;
      epv.rotation().GetComponents(x,y,z);
      if ( dumpPosInfo ) {
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.translation().x();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.translation().y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.translation().z();
	//             dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().thetaX()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().phiX()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().thetaY()/deg;
	//             dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().phiY()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().thetaZ()/deg;
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().phiZ()/deg;
	
	//          dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().xx();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().xy();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().xz();
	//          dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().yx();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().yy();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().yz();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().zx();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().zy();
	// 	    dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << epv.rotation().zz();
	
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.X();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.X();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.X();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Y();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << x.Z();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << y.Z();
	dump << "," << std::setw(12) << std::fixed << std::setprecision(4) << z.Z();
      }
      dump << std::endl;;
      ++id;
      if ( nVols != 0 && id > nVols ) notReachedDepth = false;
    } while (epv.next() && notReachedDepth);
    dump.close();
  }
  if ( dumpSpecs ) {
    // dump specifics at every compact-view nodes to have the most detailed "true" 
    // final destination of the DDSpecifics
    std::string dsname = "dumpSpecs" + fname;
    std::ofstream dump(dsname.c_str());
// <<<<<<< GeometryInfoDump.cc
    DDCompactView::DDCompactView::graph_type gra = cpv.graph();
    std::set<DDLogicalPart> lpStore;
    typedef  DDCompactView::graph_type::const_adj_iterator adjl_iterator;
    adjl_iterator git = gra.begin();
    adjl_iterator gend = gra.end();        
    DDCompactView::graph_type::index_type i=0;
    git = gra.begin();
    for (; git != gend; ++git) 
    {
      const DDLogicalPart & ddLP = gra.nodeData(git);
      if ( lpStore.find(ddLP) != lpStore.end() && ddLP.attachedSpecifics().size() != 0 ) {
	dump << ddLP.toString() << ": ";
	dumpSpec( ddLP.attachedSpecifics(), dump );
	//	dumpSpec( ddLP.valueToPartSelectors(), dump );
      }
      lpStore.insert(ddLP);

      ++i;
      if (git->size()) 
	{
	  // ask for children of ddLP  
	  DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
	  DDCompactView::graph_type::edge_list::const_iterator cend = git->end();
	  for (; cit != cend; ++cit) 
	    {
	      const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
	      if (lpStore.find(ddcurLP) != lpStore.end() && ddcurLP.attachedSpecifics().size() != 0 ) {
		dump << ddcurLP.toString() << ": ";
		dumpSpec( ddcurLP.attachedSpecifics(), dump );
		//		dumpSpec( ddcurLP.valueToPartSelectors(), dump );
	      }
	      lpStore.insert(ddcurLP);
	    } // iterate over children
	} // if (children)
    } // iterate over graph nodes  
    dump.close();
// =======
//     DDCompactView::DDCompactView::graph_type gra = cpv.graph();
//     std::vector<std::pair< DDPartSelection*, DDsvalues_type* > > specStore;
//     std::set<DDLogicalPart> lpStore;
//     typedef  DDCompactView::graph_type::const_adj_iterator adjl_iterator;
//     adjl_iterator git = gra.begin();
//     adjl_iterator gend = gra.end();        
//     DDCompactView::graph_type::index_type i=0;
//     git = gra.begin();
//     for (; git != gend; ++git) 
//     {
//       const DDLogicalPart & ddLP = gra.nodeData(git);
//       if ( lpStore.find(ddLP) != lpStore.end() && ddLP.attachedSpecifics().size() != 0 ) {
// 	dump << ddLP.toString() << " : " << std::endl;
// 	specStore.reserve(specStore.size()+ddLP.attachedSpecifics().size());
// 	std::copy(ddLP.attachedSpecifics().begin(), ddLP.attachedSpecifics().end(), std::back_inserter(specStore));//, specStore.end()); //
// 	std::vector<std::pair< DDPartSelection*, DDsvalues_type*> >::const_iterator bit(ddLP.attachedSpecifics().begin()), eit(ddLP.attachedSpecifics().end());
// 	for ( ; bit != eit; ++bit ) {
// 	  // DDsvalues_type is typedef std::vector< std::pair<unsigned int, DDValue> > DDsvalues_type;  
// 	  DDsvalues_type::iterator bsit(bit->second->begin()), bseit(bit->second->end());
// 	  for ( ; bsit != bseit; ++bsit ) {
// 	    dump << bsit->second.name() << " ";
// 	    dump << ( bsit->second.isEvaluated() ?  "evaluated" : "NOT eval." );
// 	    const std::vector<std::string>& strs = bsit->second.strings();
// 	    std::vector<double> ldbls;
// 	    ldbls.resize(strs.size(), 0.0);
// 	    if ( bsit->second.isEvaluated() ) {
// 	      ldbls = bsit->second.doubles();
// 	    }
// 	    if ( strs.size() != ldbls.size() ) std::cout << "CRAP! " << bsit->second.name() << " does not have equal number of doubles and strings." << std::endl;
// 	    size_t sdind(0);
// 	    for ( ; sdind != strs.size() ; ++sdind ) {
// 	      dump << " [" << strs[sdind] << "," << ldbls[sdind] << "]";
// 	    }
// 	  }
// 	  dump << std::endl;
// 	}
//       }
//       lpStore.insert(ddLP);

//       ++i;
//       if (git->size()) 
// 	{
// 	  // ask for children of ddLP  
// 	  DDCompactView::graph_type::edge_list::const_iterator cit  = git->begin();
// 	  DDCompactView::graph_type::edge_list::const_iterator cend = git->end();
// 	  for (; cit != cend; ++cit) 
// 	    {
// 	      const DDLogicalPart & ddcurLP = gra.nodeData(cit->first);
// 	      if (lpStore.find(ddcurLP) != lpStore.end() && ddcurLP.attachedSpecifics().size() != 0 ) {
// 		specStore.reserve(specStore.size()+ddcurLP.attachedSpecifics().size());
// 		std::copy(ddcurLP.attachedSpecifics().begin(), ddcurLP.attachedSpecifics().end(), std::back_inserter(specStore));
// 		std::vector<std::pair< DDPartSelection*, DDsvalues_type*> >::const_iterator bit(ddcurLP.attachedSpecifics().begin()), eit(ddcurLP.attachedSpecifics().end());
// 		dump << ddcurLP.toString() << " : " << std::endl;
// 		for ( ; bit != eit; ++bit ) {
// 		  DDsvalues_type::iterator bsit(bit->second->begin()), bseit(bit->second->end());
// 		  for ( ; bsit != bseit; ++bsit ) {
// 		    dump << bsit->second.name() << " ";
// 		    dump << ( bsit->second.isEvaluated() ? "evaluated" : "NOT eval." );
// 		    const std::vector<std::string>& strs = bsit->second.strings();
// 		    std::vector<double> ldbls;
// 		    ldbls.resize(strs.size(), 0.0);
// 		    if ( bsit->second.isEvaluated() ) {
// 		      ldbls = bsit->second.doubles();
// 		    }
// 		    if ( strs.size() != ldbls.size() ) std::cout << "CRAP! " << bsit->second.name() << " does not have equal number of doubles and strings." << std::endl;
// 		    size_t sdind(0);
// 		    for ( ; sdind != strs.size() ; ++sdind ) {
// 		      dump << " [" << strs[sdind] << "," << ldbls[sdind] << "]";
// 		    }
// 		    dump << std::endl;
// 		  }
// 		  dump << std::endl;
// 		}
// 	      }
// 	      lpStore.insert(ddcurLP);
// 	    } // iterate over children
// 	} // if (children)
//     } // iterate over graph nodes  
//     std::vector<std::pair<DDPartSelection*, DDsvalues_type*> >::iterator spit(specStore.begin()), spend (specStore.end());
//     for (; spit != spend; ++spit) {
//       if ( !spit->isDefined().second ) continue;  
//       const DDSpecifics & sp = *spit;
//       dump << sp << std::endl;
//     }
//     dump.close();
// >>>>>>> 1.12
   }
// <<<<<<< GeometryInfoDump.cc
  
// =======
//   if ( dumpSpecs ) {
//     DDSpecifics::iterator<DDSpecifics> spit(DDSpecifics::begin()), spend(DDSpecifics::end());
//     // ======= For each DDSpecific...
//     std::string dsname = "dumpSpecs" + fname;
//     std::ofstream dump(dsname.c_str());
//     for (; spit != spend; ++spit) {
//       if ( !spit->isDefined().second ) continue;  
//       const DDSpecifics & sp = *spit;
//       dump << sp << std::endl;
//     }
//     dump.close();
//   }
// >>>>>>> 1.12
 }

void GeometryInfoDump::dumpSpec( const std::vector<std::pair< DDPartSelection*, DDsvalues_type*> >& attspec, std::ostream& dump) {
  std::vector<std::pair< DDPartSelection*, DDsvalues_type*> >::const_iterator bit(attspec.begin()), eit(attspec.end());
  for ( ; bit != eit; ++bit ) {
    //  DDPartSelection is a std::vector<DDPartSelectionLevel>
    std::vector<DDPartSelectionLevel>::iterator psit(bit->first->begin()), pseit(bit->first->end());
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
    DDsvalues_type::iterator bsit(bit->second->begin()), bseit(bit->second->end());
    for ( ; bsit != bseit; ++bsit ) { 
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
