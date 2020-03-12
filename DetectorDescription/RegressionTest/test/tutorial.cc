#include <cstdlib>
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDEnums.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDScope.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Graph.h"
#include "DetectorDescription/Core/interface/ClhepEvaluator.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace geant_units::operators;

namespace {
  class GroupFilter : public DDFilter {
  public:
    GroupFilter(std::vector<DDSpecificsFilter*>& filters) : filters_(filters) {}

    bool accept(const DDExpandedView& cv) const final {
      bool returnValue = true;
      for (const auto& f : filters_) {
        returnValue = returnValue and f->accept(cv);
        if (not returnValue) {
          break;
        }
      }
      return returnValue;
    };

  private:
    std::vector<DDSpecificsFilter*> filters_;
  };
}  // namespace

DDTranslation calc(const DDGeoHistory& aHist) {
  const DDGeoHistory& h = aHist;
  unsigned int sz = h.size();
  std::vector<DDRotationMatrix> vr;
  std::vector<DDTranslation> vt;
  DDRotationMatrix r;
  vr.emplace_back(r);

  if (h.size() > 1) {
    vt.emplace_back(h[1].posdata()->translation());
    unsigned int i = 1;
    for (; i <= sz - 2; ++i) {
      vr.emplace_back(vr.back() * h[i].posdata()->ddrot().rotation());
      vt.emplace_back(h[i + 1].posdata()->translation());
    }
  }

  DDTranslation t;
  for (unsigned int i = 0; i < vt.size(); ++i) {
    t += vr[i] * vt[i];
  }
  return t;
}

void debugHistory(const DDGeoHistory& h) {
  static constexpr char const c = 'a';
  DDGeoHistory::const_iterator it = h.begin();
  std::string fname("hdebug_");
  std::ofstream file((fname + c).c_str());
  std::vector<DDRotationMatrix> rmv;
  std::vector<DDTranslation> tv;
  for (; it != h.end(); ++it) {
  }
}

void goPersistent(const DDCompactView& cv, const std::string& file) {
  std::ofstream f(file.c_str());
  const auto& g = cv.graph();
  unsigned int node = 0;
  auto it = g.begin();
  for (; it != g.end(); ++it) {
    auto eit = it->begin();
    for (; eit != it->end(); ++eit) {
      unsigned int eindex = eit->first;
      int copyno = g.edgeData(eit->second)->copyno();
      double x, y, z;

      x = g.edgeData(eit->second)->trans().x();
      y = g.edgeData(eit->second)->trans().y();
      z = g.edgeData(eit->second)->trans().z();
      f << node << " " << eindex << " " << copyno << " " << x << " " << y << " " << z << " "
        << g.edgeData(eit->second)->ddrot().ddname().ns() << " " << g.edgeData(eit->second)->ddrot().ddname().name()
        << std::endl;
    }
    ++node;
  }
  f.close();
}

bool NEXT(DDFilteredView& fv, int& count) {
  bool result = false;

  if (fv.firstChild()) {
    result = true;
  } else if (fv.nextSibling()) {
    result = true;
  } else {
    while (fv.parent()) {
      if (fv.nextSibling()) {
        result = true;
        break;
      }
    }
  }

  if (result)
    ++count;
  return result;
}

void dumpHistory(const DDGeoHistory& h, bool short_dump = false) {
  DDGeoHistory::size_type i = 0;
  for (; i < h.size(); ++i) {
    std::cout << h[i].logicalPart().name() << "[" << h[i].copyno() << "]-";
    if (!short_dump) {
      DDAxisAngle ra(h[i].absRotation());
      std::cout << h[i].absTranslation() << ra.Axis() << convertRadToDeg(ra.Angle());
    }
  }
}

// function object to compare to ExpandedNodes
struct NodeComp {
  bool operator()(const DDExpandedNode& n1, const DDExpandedNode& n2) {
    const DDTranslation& t1 = n1.absTranslation();
    const DDTranslation& t2 = n2.absTranslation();

    bool result = false;

    // 'alphabetical ordering' according to absolute position

    if (t1.z() < t2.z()) {
      result = true;
    } else if ((t1.z() == t2.z()) && (t1.y() < t2.y())) {
      result = true;
    } else if ((t1.z() == t2.z()) && (t1.y() == t2.y()) && (t1.x() < t2.x())) {
      result = true;
    }

    return result;
  }
};

#include <algorithm>  // sort

using DDNodes = std::vector<DDExpandedNode>;

void dump_nodes(DDNodes& nodes, int max = 100) {
  DDNodes::iterator it = nodes.begin();
  DDNodes::iterator ed = nodes.end();

  sort(it, ed, NodeComp());
  int nodeCount = 1;
  for (; it != nodes.end(); ++it) {
    std::cout << nodeCount << " " << it->logicalPart() << " trans=" << it->absTranslation() << " [mm]" << std::endl;
    std::cout << "  specifics: " << it->logicalPart().specifics().size() << std::endl;
    if (nodeCount == max)
      break;
    ++nodeCount;
  }
  if (nodeCount == max)
    std::cout << " ... truncated ... " << std::endl;
}

void tutorial() {
  std::cout << std::endl
            << std::endl
            << "   >>>      Entering DDD user-code in tutorial.cc       <<< " << std::endl
            << "            -------------------------------------" << std::endl
            << std::endl;

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
  const auto& cpvGraph = cpv.graph();

  // Using the Graph.h interface, some basic information is available:
  std::cout << "CompactView, basic information: " << std::endl
            << "   LogicalParts = " << cpvGraph.size() << std::endl
            << "   PosParts( converted to DDPosData *) = [not yet available]"  //<< cpvGraph.edgeCount()
            << std::endl
            << std::endl;

  // Now we navigate a bit:
  // The root of the graph can be obtained from DDCompactView:
  std::cout << "Root volume of the Detector Description is named [" << cpv.root() << "]" << std::endl << std::endl;

  // The same, but creating a reference to it:
  const DDLogicalPart& root = cpv.root();
  const DDLogicalPart& world = root;  //(DDName("CMS","cms"));
  std::cout << "The world volume is described by following solid:" << std::endl;
  std::cout << world.solid() << std::endl << std::endl;
  const DDMaterial& worldMaterial = root.material();
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
  while (exv.next()) {  // loop over the !whole! ExpandedView ...
    if (exv.logicalPart().category() == DDEnums::sensitive) {
      ++sensVolCount;
    }
    ++overall;
  }
  EndT = clock();
  std::cout << "Time: " << ((double)(EndT - StartT)) / double(CLOCKS_PER_SEC) << " sec" << std::endl;
  std::cout << "There were " << sensVolCount << " sensitive volumes counted" << std::endl
            << "out of " << overall << " expanded nodes! " << std::endl
            << std::endl;

  // Test the SCOPE
  DDCompactView ccv;
  DDExpandedView ev(ccv);
  ev.firstChild();
  ev.firstChild();
  ev.firstChild();
  ev.nextSibling();
  ev.firstChild();
  ev.firstChild();
  std::cout << "now: " << ev.logicalPart() << std::endl;
  DDGeoHistory h = ev.geoHistory();
  std::cout << "now-hist: " << h << std::endl;

  DDExpandedView sev(ccv);
  std::cout << "scope-set=" << sev.setScope(h, 1) << std::endl;
  std::cout << "scope-root=" << sev.logicalPart() << std::endl;
  std::cout << "scope-scope=" << sev.scope() << std::endl;
  int si = 0;
  while (sev.next()) {
    std::cout << "scope-next=" << sev.geoHistory() << std::endl;
    ++si;
  }
  std::cout << "counted " << si << " nodes in the scope " << std::endl << std::endl;

  // test the filter

  std::map<std::string, DDCompOp> cop;
  cop["=="] = DDCompOp::equals;
  cop["!="] = DDCompOp::not_equals;
  bool moreFilters = true;
  bool moreQueries = true;
  bool moreFilterCriteria = true;
  std::string flog, ls, p, cs, v, q;
  while (moreQueries) {
    std::vector<DDSpecificsFilter*> vecF;
    while (moreFilters) {
      DDSpecificsFilter* f = new DDSpecificsFilter();
      std::string flog;
      std::string asString;
      std::cout << "filter LogOp = ";
      std::cin >> flog;
      if (flog == "end")
        break;
      vecF.emplace_back(f);
      while (moreFilterCriteria) {
        std::cout << " logic   = ";
        std::cin >> ls;
        if (ls == "end")
          break;
        std::cout << " par-name= ";
        std::cin >> p;
        std::cout << " comp    = ";
        std::cin >> cs;
        std::cout << " par-val = ";
        std::cin >> v;

        double dv = 0.;
        try {
          dv = ClhepEvaluator().eval("", v);
        } catch (const cms::Exception& e) {
          dv = 0;
        }
        DDValue ddval(p, v, dv);
        vecF.back()->setCriteria(ddval, cop[cs]);

      }  //<- moreFilterCriteria
    }    //<- morFilters

    std::string ans;
    ans = "";
    DDCompactView aaaaa;

    std::cout << "go persistent (filename,n)?";
    std::cin >> ans;
    if (ans != "n") {
      goPersistent(aaaaa, ans);
    }

    std::cout << "default num-scheme? ";
    std::cin >> ans;
    if (ans == "y") {
      std::cout << "creating the default numbering scheme ..." << std::endl;

      DDExpandedView eeeee(aaaaa);

      std::cout << "do sibling-stack navigation?";
      std::cin >> ans;
      if (ans == "y") {
        bool go = true;
        while (go) {
          DDCompactView c;
          DDExpandedView e(c);
          DDExpandedView::nav_type n;
          std::cout << "  size ( 0 to stop )=";
          int s;
          std::cin >> s;
          go = (bool)s;
          int i = 0;
          for (; i < s; ++i) {
            int k;
            std::cin >> k;
            n.emplace_back(k);
          }
          std::cout << "input=" << n << std::endl;
          if (e.goTo(n)) {
            std::cout << "node=" << e.geoHistory() << std::endl;
          } else {
            std::cout << "no match!" << std::endl;
          }
        }
      }

      std::cout << "node calculation based on id?" << std::endl;
      std::cin >> ans;
      if (ans == "y") {
        bool go = true;
        while (go) {
          DDCompactView c;
          DDExpandedView e(c);
          DDExpandedView::nav_type n;
          std::cout << "  id ( 0 to stop)=";
          int s;
          std::cin >> s;
          go = (bool)s;
        }
      }
    }
    std::cout << "iterate the FilteredView (y/n)";
    std::cin >> ans;
    DDCompactView compactview;

    if (ans == "y") {
      GroupFilter gf(vecF);
      DDFilteredView fv(compactview, gf);

      int count = 0;
      std::cout << "The filtered-view contained " << count << " nodes." << std::endl;
      fv.reset();
      std::cout << "Now entering interactive navigation: f = (f)irstChild," << std::endl
                << "                                     n = (n)extSibling," << std::endl
                << "                                     p = (p)arent," << std::endl
                << "                                     s = (s)tatus," << std::endl
                << "                                     h = (h)istory debugging," << std::endl
                << "                                     e = (e)nd" << std::endl;
      std::string nav = "";
      DDCompactView wcpv;
      while (nav != "e") {
        std::cout << "  nav = ";
        std::cin >> nav;
        char c = nav[0];
        typedef std::vector<std::pair<const DDPartSelection*, const DDsvalues_type*> > spectype;
        spectype v = fv.logicalPart().attachedSpecifics();
        std::vector<const DDsvalues_type*> vlp = fv.specifics();
        std::vector<const DDsvalues_type*> only = fv.logicalPart().specifics();
        DDsvalues_type merged = fv.mergedSpecifics();
        DDLogicalPart curlp = fv.logicalPart();
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
            std::cout << std::endl << "specifics sets = " << v.size() << ":" << std::endl;
            for (const auto& o : v) {
              std::cout << *(o.first) << " = " << *(o.second) << std::endl;  // << std::endl;
            }
            std::cout << std::endl;
            std::cout << "merged-specifics:" << std::endl;
            std::cout << merged << std::endl;

            std::cout << "specifics only at logicalPart:" << std::endl;
            for (const auto& o : only) {
              std::cout << *o << std::endl;
            }
            std::cout << std::endl;
            std::cout << "translation: " << fv.translation() << std::endl;
            std::cout << "       calc: " << calc(fv.geoHistory()) << std::endl;
            {
              DDAxisAngle aa(fv.rotation());
              std::cout << "rotation: axis=" << aa.Axis() << " angle=" << convertRadToDeg(aa.Angle()) << std::endl
                        << std::endl;
            }
            std::cout << "sibling-stack=" << fv.navPos() << std::endl << std::endl;
            std::cout << "material=" << fv.logicalPart().material().ddname() << std::endl;
            std::cout << "solid=" << fv.logicalPart().solid().ddname()
                      << " volume[m3]=" << convertMm3ToM3(fv.logicalPart().solid().volume()) << std::endl;
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
    if (ans != "y")
      moreQueries = false;

    int fv_count = 0;

    clock_t Start, End;
    Start = clock();
    //while (NEXT(fv,fv_count)) ;
    int cc = 0;
    auto g = math::GraphWalker<DDLogicalPart, DDPosData*>(DDCompactView().graph(), DDCompactView().root());
    while (g.next())
      ++cc;
    End = clock();
    std::cout << "Time : " << ((double)(End - Start)) / double(CLOCKS_PER_SEC) << " sec" << std::endl;
    //std::cout << fv.history().back() << std::endl;
    std::cout << "Nodes: " << cc << std::endl;
    std::cout << "Using navigation the filtered-view has " << fv_count << " nodes." << std::endl;
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
  bool goon(true);  // just for truncation of lengthy output
  int trunc = 0;    // just for truncation of lengthy output
  std::cout << "retrieving all nodes with specific data attached ..." << std::endl;
  while (ex.next() && goon) {
    // ask each expanded-not for its specifics
    // std::vector<..>.size() will be 0 if there are no specifics
    std::vector<const DDsvalues_type*> spec = ex.specifics();
    if (!spec.empty()) {
      std::cout << spec.size() << " different specific-data sets found for " << std::endl;
      dumpHistory(ex.geoHistory(), true);
      std::cout << std::endl;
      if (trunc > 3) {
        std::cout << " ... truncated .. " << std::endl;
        goon = false;
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
    file << PPIT->second.myself().name() << " -> " << PPIT->second.mother().name() << " ; " << std::endl;
  }
  file << "}" << std::endl;
  file.close();

#endif  // DD_OLD_SCHEME

  std::cout << std::endl
            << std::endl
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
