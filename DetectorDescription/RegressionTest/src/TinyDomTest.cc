#include "DetectorDescription/RegressionTest/src/TinyDomTest.h"

#include <utility>

#include "DataFormats/Math/interface/Graph.h"
#include "DetectorDescription/RegressionTest/src/TagName.h"


TinyDomTest::TinyDomTest(const TinyDom & d) 
 : dom_(d) 
 { }
 
unsigned int TinyDomTest::allNodes(const NodeName & tagName, std::vector<const AttList *> & result)
{
   result.clear();
   TinyDom::const_adj_iterator it = dom_.begin();  
   TinyDom::const_adj_iterator ed = dom_.end();  
   for (; it != ed; ++it) {
     const TinyDom::edge_list & el = *it;
     TinyDom::edge_list::const_iterator el_it = el.begin();
     TinyDom::edge_list::const_iterator el_ed = el.end();
     for (; el_it != el_ed; ++el_it) {
       if ( dom_.nodeData(el_it->first).sameName(tagName) ) {
         result.push_back(& dom_.edgeData(el_it->second));
       }
     }
   }
   return result.size();
}
