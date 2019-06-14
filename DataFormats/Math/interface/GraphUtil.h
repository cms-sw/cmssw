#ifndef DATA_FORMATS_MATH_GRAPH_UTIL_H
#define DATA_FORMATS_MATH_GRAPH_UTIL_H

#include "DataFormats/Math/interface/Graph.h"
#include "DataFormats/Math/interface/GraphWalker.h"
#include <iostream>
#include <string>

template <class N, class E>
void output(const math::Graph<N, E>& g, const N& root) {
  math::GraphWalker<N, E> w(g, root);
  bool go = true;
  while (go) {
    std::cout << w.current().first << ' ';
    go = w.next();
  }
  std::cout << std::endl;
}

template <class N, class E>
void graph_combine(const math::Graph<N, E>& g1,
                   const math::Graph<N, E>& g2,
                   const N& n1,
                   const N& n2,
                   const N& root,
                   math::Graph<N, E>& result) {
  result = g1;
  result.replace(n1, n2);
  math::GraphWalker<N, E> walker(g2, n2);
  while (walker.next()) {
    const N& parent = g2.nodeData((++walker.stack().rbegin())->first->first);
    result.addEdge(parent, walker.current().first, walker.current().second);
  }
  result.replace(n2, root);
}

template <class N, class E>
void graph_tree_output(const math::Graph<N, E>& g, const N& root, std::ostream& os) {
  math::GraphWalker<N, E> w(g, root);
  bool go = true;
  unsigned int depth = 0;
  while (go) {
    std::string s(2 * depth, ' ');
    os << ' ' << s << w.current().first << '(' << w.current().second << ')' << std::endl;
    go = w.firstChild();
    if (go) {
      ++depth;
    } else if (w.stack().size() > 1 && w.nextSibling()) {
      go = true;
    } else {
      go = false;
      while (w.parent()) {
        --depth;
        if (w.stack().size() > 1 && w.nextSibling()) {
          go = true;
          break;
        }
      }
    }
  }
}

#endif
