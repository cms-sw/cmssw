#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include "DataFormats/Math/interface/GraphUtil.h"

#include <iostream>
#include <string>

using namespace std;

using graph_type = math::Graph<string, string>;
using walker_type = math::GraphWalker<string, string>;

void build_graph(graph_type& g) {
  /*
  
       A     B     
      //\    |  
      C  D   E   
      \ /        
       G F      
       |/
       H 
 edge direction is from top to down, e.g. from D to G  
 edge-names e.g.:  B-E: e1
                   F-H: h1
 
 The graph has 3 possible roots: A, B, F
*/

  g.addEdge("B", "E", "e1");
  g.addEdge("G", "H", "h1");
  g.addEdge("A", "C", "c1");
  g.addEdge("D", "G", "g1");
  g.addEdge("A", "C", "c2");
  g.addEdge("C", "G", "g1");
  g.addEdge("F", "H", "f1");
  g.addEdge("A", "D", "d1");
}

void build_graph2(graph_type& g) {
  /*
       AA
      /  \     EE
     BB  CC   / \
     \\  /   FF GG
       DD
  */

  g.addEdge("AA", "BB", "bb1");
  g.addEdge("AA", "CC", "cc1");
  g.addEdge("BB", "DD", "dd1");
  g.addEdge("BB", "DD", "dd2");
  g.addEdge("CC", "DD", "dd3");
  g.addEdge("EE", "FF", "ff1");
  g.addEdge("EE", "GG", "gg2");
}

/* invert the graph of build_graph2():

      DD
    //  \     FF  GG
    BB  CC     \ /
     \ /        EE
      A
*/
void build_graph3(const graph_type& input, graph_type& output) { input.invert(output); }

void list_roots(const graph_type& g, ostream& os) {
  graph_type::edge_list roots;
  g.findRoots(roots);
  while (!roots.empty()) {
    os << g.nodeData(roots.back().first) << ' ';
    roots.pop_back();
  }
}

void serialize(const graph_type& g, const string& root, ostream& os) {
  walker_type w(g, root);
  bool go = true;
  while (go) {
    os << w.current().first << ' ';
    go = w.next();
  }
}
void serialize(const graph_type& g, ostream& os) {
  walker_type w(g);
  bool go = true;
  while (go) {
    os << w.current().first << ' ';
    go = w.next();
  }
}

void dfs_bfs(const graph_type& g, const string& root, ostream& os) {
  walker_type w1(g, root);
  walker_type w2(g, root);

  bool doit = true;
  os << "bfs iteration:" << endl;
  while (doit) {
    os << w2.current_bfs().first << ' ';
    doit = w2.next_bfs();
  }
  os << endl;
  doit = true;
  os << "dfs iteration:" << endl;
  while (doit) {
    os << w2.current().first << ' ';
    doit = w2.next();
  }
  os << endl;
}

int main() {
  ostream& os = cout;

  graph_type g1;
  build_graph(g1);
  dfs_bfs(g1, "A", os);

  os << "roots of the graph are: ";
  list_roots(g1, os);
  os << endl;

  os << "tree serialization: ";
  serialize(g1, "A", os);
  os << "tree hierarchy: " << endl;
  graph_tree_output(g1, string("A"), os);

  os << "exchanging node C through node Y." << endl;
  unsigned int idx = g1.replace("C", "Y");
  os << idx << endl;
  graph_tree_output(g1, string("A"), os);

  os << "replacing edge h1 with exchanged_h1 " << endl;
  g1.replaceEdge("h1", "exchanged_h1");
  graph_tree_output(g1, string("A"), os);

  graph_type g2;
  build_graph2(g2);
  os << "second graph:" << endl;
  serialize(g2, "AA", os);

  os << endl << "combining g1 and g2:" << endl;
  os << "g1: ";
  serialize(g1, "A", os);
  os << endl;
  os << "g2: ";
  serialize(g2, "AA", os);
  os << endl;
  graph_type g3;
  graph_combine<string, string>(g1, g2, "A", "AA", "NewRoot", g3);
  os << "g3: ";
  serialize(g3, "NewRoot", os);
  os << endl;
  graph_tree_output(g3, string("NewRoot"), os);

  os << endl << "inverting g2:" << endl;
  graph_type g4;
  g2.invert(g4);
  graph_tree_output(g4, string("DD"), os);
  graph_tree_output(g4, string("FF"), os);
  graph_tree_output(g4, string("GG"), os);

  os << endl << "graph-iterator test: loop over g1" << endl;
  graph_type gg1;
  build_graph(gg1);
  graph_type::const_iterator it(gg1.begin_iter());
  graph_type::const_iterator ed(gg1.end_iter());
  for (; it != ed; ++it) {
    cout << "looping! from=" << (*it).from() << " to=" << flush;
    cout << (*it).to() << " edge=" << flush;
    cout << it->edge() << endl;
  }

  return 0;
}
