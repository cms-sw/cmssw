#ifndef DATA_FORMATS_MATH_GRAPH_WALKER_H
#define DATA_FORMATS_MATH_GRAPH_WALKER_H

#include "DataFormats/Math/interface/Graph.h"
#include <queue>
#include <vector>

namespace math {

  /** a walker for an acyclic directed multigraph */
  template <class N, class E>
  class GraphWalker {
  public:
    using index_type = typename math::Graph<N, E>::index_type;
    using index_result = typename math::Graph<N, E>::index_result;
    using edge_type = typename math::Graph<N, E>::edge_type;
    using edge_list = typename math::Graph<N, E>::edge_list;
    using edge_iterator = typename math::Graph<N, E>::edge_iterator;
    using const_edge_iterator = typename math::Graph<N, E>::const_edge_iterator;

    // only a const-edge_range!
    using edge_range = std::pair<const_edge_iterator, const_edge_iterator>;

    using stack_type = std::vector<edge_range>;
    using bfs_type = std::queue<edge_type>;

    using result_type = bool;
    using value_type = typename math::Graph<N, E>::value_type;

  public:
    //! creates a walker rooted by the first candidate root found in the underlying Graph
    GraphWalker(const Graph<N, E> &);

    //! creates a walker rooted by the node given
    GraphWalker(const Graph<N, E> &, const N &);

    // operations

    result_type firstChild();

    result_type nextSibling();

    result_type parent();

    result_type next();

    inline value_type current() const;

    result_type next_bfs();
    value_type current_bfs() const;

    void reset();

    const stack_type &stack() const { return stack_; }

  protected:
    // stack_.back().first corresponds to index of the current node!
    stack_type stack_;  // hierarchical stack used in navigation
    bfs_type queue_;    // breath first search queue
    edge_list root_;    // root of the walker
    const Graph<N, E> &graph_;

  private:
    GraphWalker() = delete;
  };

  template <class N, class E>
  GraphWalker<N, E>::GraphWalker(const Graph<N, E> &g) : graph_(g) {  // complexity = (no nodes) * (no edges)
    graph_.findRoots(root_);
    stack_.emplace_back(edge_range(root_.begin(), root_.end()));
    if (!root_.empty()) {
      queue_.push(root_[0]);
    }
  }

  template <class N, class E>
  GraphWalker<N, E>::GraphWalker(const Graph<N, E> &g, const N &root) : graph_(g) {
    index_result rr = graph_.nodeIndex(root);
    if (!rr.second)  // no such root node, no walker can be created!
      throw root;

    root_.emplace_back(edge_type(rr.first, 0));
    stack_.emplace_back(edge_range(root_.begin(), root_.end()));
    queue_.push(root_[0]);
  }

  template <class N, class E>
  typename GraphWalker<N, E>::value_type GraphWalker<N, E>::current() const {
    const edge_range &er = stack_.back();
    return value_type(graph_.nodeData(er.first->first), graph_.edgeData(er.first->second));
  }

  template <class N, class E>
  typename GraphWalker<N, E>::value_type GraphWalker<N, E>::current_bfs() const {
    const edge_type &e = queue_.front();
    return value_type(graph_.nodeData(e.first), graph_.edgeData(e.second));
  }

  template <class N, class E>
  void GraphWalker<N, E>::reset() {
    stack_.clear();
    stack_.emplace_back(edge_range(root_.begin(), root_.end()));
    queue_.clear();
    if (root_.size()) {
      queue_.push(root_[0]);
    }
  }

  template <class N, class E>
  typename GraphWalker<N, E>::result_type GraphWalker<N, E>::firstChild() {
    result_type result = false;
    const edge_range &adjEdges = graph_.edges(stack_.back().first->first);
    if (adjEdges.first != adjEdges.second) {
      stack_.emplace_back(adjEdges);
      result = true;
    }
    return result;
  }

  template <class N, class E>
  typename GraphWalker<N, E>::result_type GraphWalker<N, E>::nextSibling() {
    result_type result = false;
    edge_range &siblings = stack_.back();
    if (siblings.first != (siblings.second - 1)) {
      ++siblings.first;
      result = true;
    }
    return result;
  }

  template <class N, class E>
  typename GraphWalker<N, E>::result_type GraphWalker<N, E>::parent() {
    result_type result = false;
    if (stack_.size() > 1) {
      stack_.pop_back();
      result = true;
    }
    return result;
  }

  template <class N, class E>
  typename GraphWalker<N, E>::result_type GraphWalker<N, E>::next() {
    result_type result = false;
    if (firstChild()) {
      result = true;
    } else if (stack_.size() > 1 && nextSibling()) {
      result = true;
    } else {
      while (parent()) {
        if (stack_.size() > 1 && nextSibling()) {
          result = true;
          break;
        }
      }
    }
    return result;
  }

  template <class N, class E>
  typename GraphWalker<N, E>::result_type GraphWalker<N, E>::next_bfs() {
    result_type result(false);
    if (!queue_.empty()) {
      const edge_type &e = queue_.front();
      const edge_range &er = graph_.edges(e.first);
      const_edge_iterator it(er.first), ed(er.second);
      for (; it != ed; ++it) {
        queue_.push(*it);
      }
      queue_.pop();
      if (!queue_.empty()) {
        result = true;
      }
    }
    return result;
  }

}  // namespace math

#endif
