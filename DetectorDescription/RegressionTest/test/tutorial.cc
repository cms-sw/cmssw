// Two modules of CLHEP are partly used in DDD
// . unit definitions (such as m, cm, GeV, ...) of module CLHEP/Units
// . rotation matrices and translation std::vectors of module CLHEP/Vector
//   (they are typedef'd to DDRotationMatrix and DDTranslation in
//   DDD/DDCore/interface/DDTransform.h
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDNodes.h"
#include "DetectorDescription/Core/interface/DDSpecifics.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDScope.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDQuery.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDPath.h"

#include "DetectorDescription/ExprAlgo/interface/ExprEvalSingleton.h"

#include <fstream>
#include <iostream>
#include <string>
#include <map>

DDTranslation calc(const DDGeoHistory & aHist)
{
  const DDGeoHistory & h = aHist;
  unsigned int sz = h.size();
  std::vector<DDRotationMatrix> vr;
  std::vector<DDTranslation> vt;
  DDRotationMatrix r;
  vr.push_back(r);
  
  if (h.size()>1) {
    vt.push_back(h[1].posdata()->translation());
    unsigned int i = 1;
    for (; i <= sz-2; ++i) {
      vr.push_back( vr.back() * *(h[i].posdata()->rot_.rotation()) );
      vt.push_back(h[i+1].posdata()->translation());
    }
  }
  
  DDTranslation t;
  for(unsigned int i=0; i<vt.size(); ++i) {
    t += vr[i]*vt[i];
  }
  return t;  
}

void debugHistory(const DDGeoHistory & h)
{
  static constexpr char const c = 'a';
  DDGeoHistory::const_iterator it = h.begin();
  std::string fname ("hdebug_");
  std::ofstream file((fname + c).c_str());
  std::vector<DDRotationMatrix> rmv;
  std::vector<DDTranslation> tv;
  for(; it != h.end(); ++it) {
  }
}

void goPersistent(const DDCompactView & cv, std::string file) {
  std::ofstream f(file.c_str());
  typedef DDCompactView::graph_type graph_t;
  const graph_t & g = cv.graph();
  unsigned int node = 0;
  graph_t::const_adj_iterator it = g.begin();
  for (; it != g.end(); ++it) {
    graph_t::const_edge_iterator eit = it->begin();
    for (; eit != it->end(); ++eit) {
      unsigned int eindex = eit->first;
      int copyno = g.edgeData(eit->second)->copyno_;
      double x,y,z;
      
      x = g.edgeData(eit->second)->trans_.x()/mm;
      y = g.edgeData(eit->second)->trans_.y()/mm;
      z = g.edgeData(eit->second)->trans_.z()/mm;
      f << node << " " << eindex << " " << copyno 
        << " " << x << " " << y << " " << z 
	<< " " << g.edgeData(eit->second)->rot_.ddname().ns()
	<< " " << g.edgeData(eit->second)->rot_.ddname().name()
        << std::endl;
    }
    ++node;  
  
  }
  f.close();  
}

bool NEXT(DDFilteredView & fv, int & count)
{
  bool result = false;
  
  if (fv.firstChild()) {
    result = true;
  }
  else if(fv.nextSibling()) {
    result = true;
  }
  else {
    while(fv.parent()) {
      if(fv.nextSibling()) {
        result = true;
	break;
      }
    }
  }    	    
  
  if (result) ++count;
  return result;
}

void dumpHistory(const DDGeoHistory & h, bool short_dump=false)
{
  DDGeoHistory::size_type i=0;
  for (; i<h.size(); ++i) {
    std::cout << h[i].logicalPart().name() << "[" << h[i].copyno() << "]-";
    if (!short_dump) { 
      DDAxisAngle ra(h[i].absRotation());
      std::cout  << h[i].absTranslation() 
		 << ra.Axis() << ra.Angle()/deg;
    }	  
  }
}

// function object to compare to ExpandedNodes 
struct NodeComp
{
  bool operator()(const DDExpandedNode & n1, const DDExpandedNode & n2)
  {
    const DDTranslation & t1 = n1.absTranslation();
    const DDTranslation & t2 = n2.absTranslation();
     
    bool result = false;
     
    // 'alphabetical ordering' according to absolute position
     
    if (t1.z() < t2.z()) 
      {
	result=true;
      } 
    else if ( (t1.z()==t2.z()) && (t1.y() < t2.y()))
      {
	result=true;
      }  
    else if ( (t1.z()==t2.z()) && (t1.y()==t2.y()) && (t1.x()<t2.x()))
      {
	result=true;
      }
     
    return result;
  }
};

#include<algorithm> // sort
void dump_nodes(DDNodes & nodes, int max=100)
{
  DDNodes::iterator it = nodes.begin();
  DDNodes::iterator ed = nodes.end();
   
  sort(it,ed,NodeComp());
  int nodeCount=1;
  for (; it != nodes.end(); ++it) {
    std::cout << nodeCount << " " << it->logicalPart() << " trans="
	      << it->absTranslation() << " [mm]" << std::endl;	 
    std::cout << "  specifics: " << it->logicalPart().specifics().size() << std::endl;	  
    if (nodeCount==max)
      break;	  
    ++nodeCount;  
  }
  if (nodeCount==max)
    std::cout << " ... truncated ... " << std::endl;
}

void tutorial()
{
  std::cout << std::endl << std::endl 
	    << "   >>>      Entering DDD user-code in tutorial.cc       <<< " << std::endl 
	    << "            -------------------------------------" << std::endl << std::endl;
 
  // Initialize a CompactView. 
  // (During XML parsing internal DDD objects are created in memory.
  //  These objects are then put into a handy structure called DDCompactView
  //  to allow hierarchical access to them)
  // The default constructor creates a CompactView using the content of
  // DDRoot::root() singleton being the root of the hierarchy. 
  // That's why we call DDRootDef::instance().set(..) before!
  // The CompactView is an acyclic directed multigraph with nodes of type
  // DDLogicalPart and edges of type DDPosData* .
  // ( refere to header documentation of these interface-classes located
  //   in DDD/DDCore/interface/..)
  // DDLogicalPart provides access to material and solid definitons
  // DDPosData provides access to relative spacial positionings of the
  // current node relative to its parent node.
 
  // Parser handles it correctly now!
  // DDRootDef::instance().set(DDName("CMS","cms.xml"));
  DDCompactView cpv;
 
  // As we want to demonstrate some DDD features in the ECal, we set the
  // default namespace to 'ecal-endcap.xml' (which is where the endcap
  // geometry is defined).
  // (Setting a default namespace can help to shorten instantiations of
  // references to objects defined in the file the namespace belongs to)
  //typedef DDCurrentNamespace CNS;
  //CNS::ns()="ecal-endcap.xml";

  // Navigating the CompactView (Usecase for building up G4-geometries):
  // -------------------------------------------------------------------
  // In the Protoype SW the CompactView can be navigated only be taking
  // its internal data structure (Graph<DDLogcialPart,DDPosData*> available
  // as type 'graph_type'). 
  // The corresponding interfaces for navigation of graphs are described in 
  // DDD/DDCore/interface/graph.h
 
  // First one takes the graph representation of CompactView
  const DDCompactView::graph_type & cpvGraph = cpv.graph();
 
  // Using the Graph.h interface, some basic information is available:
  std::cout << "CompactView, basic information: " << std::endl
	    << "   LogicalParts = " << cpvGraph.size() << std::endl
	    << "   PosParts( converted to DDPosData *) = [not yet available]" //<< cpvGraph.edgeCount() 
	    << std::endl << std::endl;
      
  // Now we navigate a bit:
  // The root of the graph can be obtained from DDCompactView:
  std::cout << "Root volume of the Detector Description is named [" 
	    << cpv.root() << "]" << std::endl << std::endl;
 
  // The same, but creating a reference to it:
  DDLogicalPart root = cpv.root(); 
  DDLogicalPart world = root; //(DDName("CMS","cms"));
  std::cout << "The world volume is described by following solid:" << std::endl;
  std::cout << world.solid() << std::endl << std::endl;
  DDMaterial worldMaterial = root.material();
  std::cout << "The world volume is filled with following material:" << std::endl;
  std::cout << worldMaterial << std::endl << std::endl;
 
  // Of course, if the names are well known (from the XML) one can always
  // obtain direct access: 
  // (note the implicit usage of the default namespace! Whenever reference
  //  variable are created without using DDName(..) but a simple std::string
  //  the default namespace DDCurrentNamespace::ns_ is automatically
  //  taken into account)
  DDLogicalPart endcapXtal("ECalEndcapCrystal");
  DDLogicalPart endcapBasket("ECalEndcapE7");
 
  // Let's switch from the CompactView to the ExpandedView
  // interfaced through DDExpandedView
  // The expanded view is a tree of DDExpandedNode s. The usual tree
  // navigation is supported. Please refere to the documentation
  // in DDD/DDCore/interface/DDExpandedView 
  DDExpandedView exv(cpv); 
 
  // Let's count all endcap-xtals in the detector
  // (the input geometry from ecal-endcap.xml is the XMLized version
  //  of OSCARs 1.3.0 endcap. When counting these xtals we get about
  //  1000 more than described in the references availabe from the
  //  Encap-Web ...)
  int sensVolCount(0);
  int overall(0);
  std::cout << "Start sensitive volumes counting ... " << std::endl;
  clock_t StartT, EndT;
  StartT = clock();
  while(exv.next()) { // loop over the !whole! ExpandedView ...
    if (exv.logicalPart().category()==DDEnums::sensitive) {
      ++sensVolCount;
    }  
    ++overall;
  }    
  EndT = clock(); 
  std::cout << "Time: " << ((double) (EndT-StartT)) / double(CLOCKS_PER_SEC) << " sec" << std::endl;
  std::cout << "There were " << sensVolCount << " sensitive volumes counted"
	    << std::endl 
	    << "out of " << overall << " expanded nodes! " << std::endl
	    << std::endl;

  // Test the SCOPE
  DDCompactView ccv;
  DDExpandedView ev(ccv);
  ev.firstChild(); ev.firstChild(); ev.firstChild();
  ev.nextSibling(); ev.firstChild(); ev.firstChild();
  std::cout << "now: " << ev.logicalPart() << std::endl;
  DDGeoHistory h = ev.geoHistory();
  std::cout << "now-hist: " << h << std::endl;

  DDExpandedView sev(ccv);
  std::cout << "scope-set=" << sev.setScope(h,1) << std::endl;
  std::cout << "scope-root=" << sev.logicalPart() << std::endl;
  std::cout << "scope-scope=" << sev.scope() << std::endl;
  int si=0;
  while (sev.next()) {
    std::cout << "scope-next=" << sev.geoHistory() << std::endl;
    ++si;
  }
  std::cout << "counted " << si << " nodes in the scope " << std::endl << std::endl;

  // test the filter

  std::map<std::string,DDCompOp> cop;
  cop["=="] = DDCompOp::equals;
  cop["!="] = DDCompOp::not_equals;
  cop["<"]  = DDCompOp::smaller;
  cop[">"]  = DDCompOp::bigger;
  cop[">="]  = DDCompOp::bigger_equals;
  cop["<="]  = DDCompOp::smaller_equals;
  std::map<std::string,DDLogOp> lop;
  lop["AND"] = DDLogOp::AND;
  lop["OR"] = DDLogOp::OR;
  bool moreFilters = true;
  bool moreQueries = true;
  bool moreFilterCriteria = true;
  std::string flog, ls, p, cs, v, q;
  while(moreQueries) {
    std::vector<std::pair<DDLogOp,DDSpecificsFilter*> > vecF;
    while(moreFilters) { 
      DDSpecificsFilter * f = new DDSpecificsFilter();
      std::string flog;
      std::string asString;
      bool asStringBool = false;
      std::cout << "filter LogOp = ";
      std::cin >> flog;
      if(flog=="end") 
	break;
      vecF.push_back(std::make_pair(lop[flog],f));
      while (moreFilterCriteria) {
	std::cout << " logic   = ";
	std::cin >> ls;
	if (ls=="end") 
	  break;
	std::cout << " par-name= ";
	std::cin >> p;
	std::cout << " comp    = ";
	std::cin >> cs;
	std::cout << " par-val = ";
	std::cin >> v;
	std::cout << " as-std::string[y/n] = ";
	std::cin >> asString;
      
	if (asString=="y") 
	  asStringBool = true;
	
	double dv = 0.;
	try {
	  dv = ExprEvalSingleton::instance().eval("",v);
	}
	catch (const cms::Exception & e) {
	  dv = 0;
	}
	DDValue ddval(p,v,dv);
	vecF.back().second->setCriteria(ddval,cop[cs],lop[ls],asStringBool);
      
      }//<- moreFilterCriteria
    }//<- morFilters
   
    DDScope scope;
    DDQuery query(ccv);
    std::vector<std::pair<DDLogOp,DDSpecificsFilter*> >::size_type loop=0;
    for(; loop < vecF.size(); ++loop) {
      DDFilter * filter = vecF[loop].second;  
      const DDFilter & fi = *filter;
      query.addFilter( fi, vecF[loop].first );
    }  
    std::cout << "The Scope is now: " << std::endl << scope << std::endl;
    std::string ans;
    ans = "";
    DDCompactView aaaaa;
  
    std::cout << "go persistent (filename,n)?";
    std::cin >> ans;
    if (ans !="n") {
      goPersistent(aaaaa,ans);
    }
  
    std::cout << "default num-scheme? ";
    std::cin >> ans;
    if (ans=="y") {
      std::cout << "creating the default numbering scheme ..." << std::endl;
  
      DDExpandedView eeeee(aaaaa);
      DDDefaultNumberingScheme nums(eeeee);
  
      std::cout << "do sibling-stack navigation?";
      std::cin >> ans;
      if (ans=="y") {
	bool go = true;
	while (go) {
	  DDCompactView c;
	  DDExpandedView e(c);
	  DDExpandedView::nav_type n;
	  std::cout << "  size ( 0 to stop )=";
	  int s;
	  std::cin >> s;
	  go = (bool)s;
	  int i=0;
	  for (; i<s; ++i) {
	    int k;
	    std::cin >> k;
	    n.push_back(k);
	  }
	  std::cout << "input=" << n << std::endl;
	  if (e.goTo(n)) {
	    std::cout << "node=" << e.geoHistory() << std::endl;
	    std::cout << "  id=" << nums.id(e) << std::endl;  
	  }
	  else {
	    std::cout << "no match!" << std::endl;
	  }  
	}
      } 
  
      std::cout << "node calculation based on id?" << std::endl;
      std::cin >> ans;
      if (ans=="y") {
	bool go = true;
	while(go) {
	  DDCompactView c;
	  DDExpandedView e(c);
	  DDExpandedView::nav_type n;
	  std::cout << "  id ( 0 to stop)=";
	  int s;
	  std::cin >> s;
	  go = (bool)s;
	  if (nums.node(s,e)) {
	    std::cout << "node=" << e.geoHistory() << std::endl;
	  }
	  else {
	    std::cout << "no match!" << std::endl;
	  }    
	}
      }  
    }
    std::cout << "exec a query based on the filter(s) (y/n) ?";
    std::cin >> ans;
    if (ans=="y") {
      const std::vector<DDExpandedNode> & res = query.exec();  
      std::cout << "the query results in " << res.size() << " nodes." << std::endl;
      if (res.size()) {
	std::cout << " the first node is:" << std::endl
		  << "  " << res[0] << " transl=" << res[0].absTranslation() << std::endl;
	std::cout << " the last node is:" << std::endl
		  << "  " << res.back() << " transl=" << res.back().absTranslation() << std::endl << std::endl;	   
      }
    
    }
  
    std::cout << "iterate the FilteredView (y/n)";
    std::cin >> ans;
    DDCompactView compactview;
    DDFilteredView fv(compactview);

    if (ans=="y") {
      for (std::vector<std::pair<DDLogOp,DDSpecificsFilter*> >::size_type j=0; j<vecF.size(); ++j) {
	fv.addFilter(*(vecF[j].second), DDLogOp::AND);
      }
    
      //bool looop = true;
      int count =0;
      /*
	while(looop) {
	looop = fv.next();
	++count;
	}
      */
      std::cout << "The filtered-view contained " << count << " nodes." << std::endl;
      fv.reset();
      std::cout << "Now entering interactive navigation: f = (f)irstChild," << std::endl
		<< "                                     n = (n)extSibling," << std::endl 
		<< "                                     p = (p)arent," << std::endl
		<< "                                     s = (s)tatus," << std::endl
		<< "                                     w = (w)eigth[kg]," << std::endl
		<< "                                     h = (h)istory debugging," << std::endl
		<< "                                     e = (e)nd" << std::endl;
      std::string nav="";
      DDCompactView wcpv;
      while (nav!="e") {
	std::cout << "  nav = ";
	std::cin >> nav;
	char c = nav[0];
	typedef std::vector<std::pair<const DDPartSelection*, const DDsvalues_type *> > spectype;
	spectype v = fv.logicalPart().attachedSpecifics();
	std::vector<const DDsvalues_type *> vlp = fv.specifics();
	std::vector<const DDsvalues_type *> only = fv.logicalPart().specifics();
	DDsvalues_type merged = fv.mergedSpecifics();
	DDLogicalPart curlp = fv.logicalPart();
	double curweight=0;
	bool result = false;
	switch (c) {
	case 'f':
	  result = fv.firstChild();
	  break;
	case 'n':
	  result = fv.nextSibling();
	  break;
	case 'p':
	  result = fv.parent();
	  break;
	case 'h':
	  debugHistory(fv.geoHistory()); 
	  break;
	case 's':
	  fv.print();
	  std::cout << std::endl <<"specifics sets = " << v.size() << ":" << std::endl;
	  for (spectype::size_type o=0;o<v.size();++o) {
	    std::cout << *(v[o].first) 
		      << " = " 
		      << *(v[o].second) 
		      << std::endl;// << std::endl;
	  }
	  std::cout << std::endl;
	  std::cout << "merged-specifics:" << std::endl;
	  std::cout << merged << std::endl;
	 
	  std::cout << "specifics only at logicalPart:" << std::endl;
	  for (std::vector<const DDsvalues_type *>::size_type o=0;o<only.size();++o) {
	    std::cout << *(only[o]) << std::endl;
	  }
	  std::cout << std::endl;	 
	  std::cout << "translation: " << fv.translation() << std::endl;
	  std::cout << "       calc: " << calc(fv.geoHistory()) << std::endl;
	  {	 
	    DDAxisAngle   aa(fv.rotation());
	    std::cout << "rotation: axis=" << aa.Axis() 
		      << " angle=" << aa.Angle()/deg << std::endl << std::endl;
	  }
	  std::cout << "sibling-stack=" << fv.navPos() << std::endl << std::endl;
	  std::cout << "material=" << fv.logicalPart().material().ddname() << std::endl;
	  std::cout << "solid=" << fv.logicalPart().solid().ddname() <<
	    " volume[m3]=" << fv.logicalPart().solid().volume()/m3 << std::endl;	      
	 
	  //std::cout << "id from default numbering-scheme=" << nums.id(fv) << std::endl;
	  break;
	case 'w':
	  curweight = wcpv.weight(curlp);
	  std::cout << " The weight of " << curlp.ddname() << " is " 
		    << curweight/kg << "kg" << std::endl;
	  std::cout << " The average density is " << curweight/curlp.solid().volume()/g*cm3 << "g/cm3" << std::endl;
	  break;     
	case 'e':
	  break;	 
	default:
	  std::cout << "  > not understood, try again < " << std::endl;
	}
	std::cout << "  node = " << fv.geoHistory().back() << " isNew=" << result << std::endl;
      }
    
    }  
  
    std::cout << " again (y/n)?";
    std::cin >> ans;
    if (ans!="y") 
      moreQueries = false;
   
    int fv_count=0;
    fv.reset();
    clock_t Start, End;
    Start = clock();
    //while (NEXT(fv,fv_count)) ;
    int cc=0;
    walker_type  g = walker_type(DDCompactView().graph(),DDCompactView().root());
    while(g.next()) ++cc;
    End = clock();
    std::cout << "Time : " << ((double) (End-Start)) / double(CLOCKS_PER_SEC) << " sec" << std::endl;
    //std::cout << fv.history().back() << std::endl;
    std::cout << "Nodes: " << cc << std::endl;
    std::cout << "Using navigation the filtered-view has " << fv_count << " nodes." << std::endl;
   
    loop=0;
    for(; loop<vecF.size(); ++loop) {
      delete vecF[loop].second; // delete the filters
      vecF[loop].second=0;
    } 
  }

  /*
    std::cout << "Deleting transient store!" << std::endl;
    DDLogicalPart::clear();
    DDRotation::clear();
    DDAlgo::clear();
    DDMaterial::clear();
    DDSolid::clear();
    DDSpecifics::clear();
    std::cout << "Transient store deleted!" << std::endl;
  */
  return;
  // Or we look for any specific data attached to any nodes 
  // while iterating the expanded-view:
  DDCompactView cp;
  DDExpandedView ex(cp);
  bool goon(true); // just for truncation of lengthy output
  int trunc=0;     // just for truncation of lengthy output   
  std::cout << "retrieving all nodes with specific data attached ..." << std::endl;
  while (ex.next() && goon) {
    // ask each expanded-not for its specifics 
    // std::vector<..>.size() will be 0 if there are no specifics
    std::vector<const DDsvalues_type *>  spec = ex.specifics();
    if (spec.size()) {
      std::cout << spec.size() << " different specific-data sets found for " << std::endl; 
      dumpHistory(ex.geoHistory(),true) ;    
      std::cout << std::endl;
      if (trunc>3) {
	std::cout << " ... truncated .. " << std::endl;
	goon=false;
      }  
      ++trunc;
    }  
  }

  // At last we create a graphical represention of the compact view by
  // iterating over the PosPart-Registry

#ifdef DD_OLD_SCHEME

  std::ofstream file("graph.dot");
  std::cout << "Writing a dot-file for the graph-structure of the compact-view .." << std::endl
	    << "File: graph.dot" << std::endl;
  file << "digraph G {" << std::endl;
  DDPosPartReg::instance_t::const_iterator PPIT = DDPosPartReg::instance().begin();
  for (; PPIT != DDPosPartReg::instance().end(); ++PPIT) {
    file << PPIT->second.myself().name()
	 << " -> " 
	 << PPIT->second.mother().name()
	 << " ; " 
	 << std::endl;
  }
  file << "}" << std::endl;
  file.close();

#endif // DD_OLD_SCHEME
     
  std::cout << std::endl << std::endl 
	    << "   >>>       End of DDD user-code in tutorial.cc       <<< " << std::endl 
	    << "            -------------------------------------" << std::endl 
	    << "                         ============            " << std::endl;

  std::cout << std::endl;
  int n_c = 500;

  exit(1);

  std::cout << "DUMPING the first " << n_c << " node-histories of the expanded-view" << std::endl << std::endl;

  DDCompactView scratch;
  DDExpandedView exx(scratch);

  while (exx.next() && n_c--) {
    dumpHistory(exx.geoHistory());
  }
}

