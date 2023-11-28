#include "DetectorDescription/Core/interface/DDCompactViewImpl.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDCompactViewImpl::DDCompactViewImpl(const DDLogicalPart& rootnodedata) : root_(rootnodedata) {
  LogDebug("DDCompactViewImpl") << "Root node data = " << rootnodedata << std::endl;
}

DDCompactViewImpl::~DDCompactViewImpl() {
  Graph::adj_list::size_type it = 0;
  if (graph_.size() == 0) {
    LogDebug("DDCompactViewImpl") << "In destructor, graph is empty." << std::endl;
  } else {
    LogDebug("DDCompactViewImpl") << "In destructor, graph is NOT empty."
                                  << " graph_.size() = " << graph_.size() << std::endl;
    for (; it < graph_.size(); ++it) {
      Graph::edge_range erange = graph_.edges(it);
      for (; erange.first != erange.second; ++(erange.first)) {
        DDPosData* pd = graph_.edgeData(erange.first->second);
        delete pd;
        pd = nullptr;
      }
    }
  }
  edm::LogInfo("DDCompactViewImpl") << std::endl
                                    << "DDD transient representation has been destructed." << std::endl
                                    << std::endl;
}

DDCompactViewImpl::GraphWalker DDCompactViewImpl::walker() const { return GraphWalker(graph_, root_); }

void DDCompactViewImpl::position(const DDLogicalPart& self,
                                 const DDLogicalPart& parent,
                                 int copyno,
                                 const DDTranslation& trans,
                                 const DDRotation& rot,
                                 const DDDivision* div) {
  DDPosData* pd = new DDPosData(trans, rot, copyno, div);
  graph_.addEdge(parent, self, pd);
}

void DDCompactViewImpl::swap(DDCompactViewImpl& implToSwap) { graph_.swap(implToSwap.graph_); }

DDCompactViewImpl::DDCompactViewImpl() {}
