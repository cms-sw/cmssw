#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/OfflineDBLoader/interface/GeometryInfoDump.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDExpandedNode.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cassert>
#include <fstream>
#include <map>
#include <set>
#include <vector>

using Graph = DDCompactView::Graph;
using adjl_iterator = Graph::const_adj_iterator;

// For output of values to four decimal places, round negative values
// equivalent to 0 within the precision to 0 to prevent printing "-0".
template <class valType>
static constexpr valType roundNeg0(valType value) {
  if (value < 0. && value > -5.0e-5)
    return (0.0);
  else
    return (value);
}

GeometryInfoDump::GeometryInfoDump() {}

GeometryInfoDump::~GeometryInfoDump() {}

void GeometryInfoDump::dumpInfo(
    bool dumpHistory, bool dumpSpecs, bool dumpPosInfo, const DDCompactView& cpv, std::string fname, int nVols) {
  fname = "dump" + fname;
  DDExpandedView epv(cpv);
  std::cout << "Top Most LogicalPart =" << epv.logicalPart() << std::endl;
  if (dumpHistory || dumpPosInfo) {
    if (dumpPosInfo) {
      std::cout << "After the GeoHistory in the output file dumpGeoHistoryOnRead you will see x, y, z, r11, r12, r13, "
                   "r21, r22, r23, r31, r32, r33"
                << std::endl;
    }
    typedef DDExpandedView::nav_type nav_type;
    typedef std::map<nav_type, int> id_type;
    id_type idMap;
    int id = 0;
    std::ofstream dump(fname.c_str());
    bool notReachedDepth(true);
    char buf[256];

    do {
      nav_type pos = epv.navPos();
      idMap[pos] = id;
      //      dump << id
      dump << " - " << epv.geoHistory();
      DD3Vector x, y, z;
      epv.rotation().GetComponents(x, y, z);
      if (dumpPosInfo) {
        size_t s = snprintf(buf,
                            256,
                            ",%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f,%12.4f",
                            roundNeg0(epv.translation().x()),
                            roundNeg0(epv.translation().y()),
                            roundNeg0(epv.translation().z()),
                            roundNeg0(x.X()),
                            roundNeg0(y.X()),
                            roundNeg0(z.X()),
                            roundNeg0(x.Y()),
                            roundNeg0(y.Y()),
                            roundNeg0(z.Y()),
                            roundNeg0(x.Z()),
                            roundNeg0(y.Z()),
                            roundNeg0(z.Z()));
        assert(s < 256);
        dump << buf;
      }
      dump << "\n";
      ;
      ++id;
      if (nVols != 0 && id > nVols)
        notReachedDepth = false;
    } while (epv.next() && notReachedDepth);
    dump << std::flush;
    dump.close();
  }
  if (dumpSpecs) {
    // dump specifics at every compact-view nodes to have the most detailed "true"
    // final destination of the DDSpecifics
    std::string dsname = "dumpSpecs" + fname;
    std::ofstream dump(dsname.c_str());
    const auto& gra = cpv.graph();
    std::set<DDLogicalPart> lpStore;
    adjl_iterator git = gra.begin();
    adjl_iterator gend = gra.end();
    for (; git != gend; ++git) {
      const DDLogicalPart& ddLP = gra.nodeData(git);
      if (lpStore.find(ddLP) != lpStore.end() && !ddLP.attachedSpecifics().empty()) {
        dump << ddLP.toString() << ": ";
        dumpSpec(ddLP.attachedSpecifics(), dump);
      }
      lpStore.insert(ddLP);

      if (!git->empty()) {
        // ask for children of ddLP
        for (const auto& cit : *git) {
          const DDLogicalPart& ddcurLP = gra.nodeData(cit.first);
          if (lpStore.find(ddcurLP) != lpStore.end() && !ddcurLP.attachedSpecifics().empty()) {
            dump << ddcurLP.toString() << ": ";
            dumpSpec(ddcurLP.attachedSpecifics(), dump);
          }
          lpStore.insert(ddcurLP);
        }  // iterate over children
      }    // if (children)
    }      // iterate over graph nodes
    dump.close();
  }
}

void GeometryInfoDump::dumpSpec(const std::vector<std::pair<const DDPartSelection*, const DDsvalues_type*> >& attspec,
                                std::ostream& dump) {
  std::vector<std::pair<const DDPartSelection*, const DDsvalues_type*> >::const_iterator bit(attspec.begin()),
      eit(attspec.end());
  for (; bit != eit; ++bit) {
    //  DDPartSelection is a std::vector<DDPartSelectionLevel>
    std::vector<DDPartSelectionLevel>::const_iterator psit(bit->first->begin()), pseit(bit->first->end());
    for (; psit != pseit; ++psit) {
      switch (psit->selectionType_) {
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
          dump << "//" << psit->lp_.toString() << "[" << psit->copyno_ << "]";
          break;
        case ddchildlogp:
          dump << "/" << psit->lp_.toString();
          break;
        case ddchildposp:
          dump << "/" << psit->lp_.toString() << "[" << psit->copyno_ << "]";
          break;
        default:
          throw cms::Exception("DetectorDescriptionSpecPar")
              << "Can not end up here! default of switch on selectionTyp_";
      }
    }
    dump << " ";
    // DDsvalues_type is typedef std::vector< std::pair<unsigned int, DDValue> > DDsvalues_type;
    DDsvalues_type::const_iterator bsit(bit->second->begin()), bseit(bit->second->end());
    for (; bsit != bseit; ++bsit) {
      dump << bsit->second.name() << " ";
      dump << (bsit->second.isEvaluated() ? "eval " : "NOT eval ");
      size_t sdind(0);
      for (; sdind != bsit->second.strings().size(); ++sdind) {
        if (bsit->second.isEvaluated()) {
          dump << bsit->second.doubles()[sdind];
        } else {
          dump << bsit->second.strings()[sdind];
        }
        if (sdind != bsit->second.strings().size() - 1)
          dump << ", ";
      }
      if (!bsit->second.strings().empty() && bsit + 1 != bseit)
        dump << " | ";
    }
    if (!bit->second->empty() && bit + 1 != eit)
      dump << " | ";
  }
  dump << std::endl;
}
