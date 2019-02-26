#ifndef DETECTOR_DESCRIPTION_DD_FILTERED_VIEW_H
#define DETECTOR_DESCRIPTION_DD_FILTERED_VIEW_H

// -*- C++ -*-
//
// Package:    DetectorDescription/DDFilteredView
// Class:      DDFilteredView
// 
/**\class DDFilteredView

 Description: Filtered View of a Tree

 Implementation:
     Filter criteria is defined in XML
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 30 Jan 2019 09:24:30 GMT
//
//
#include "DetectorDescription/DDCMS/interface/DDExpandedNode.h"

#include <vector>

namespace cms {

  class DDDetector;

  //! Geometrical 'path' of the current node up to the root-node
  using DDGeoHistory = std::vector<DDExpandedNode>;
  using Volume = dd4hep::Volume;
    
  struct DDFilteredView {
    
    struct ExpandedNodes {
      std::vector<double> tags;
      std::vector<double> offsets;
      std::vector<int> copyNos;
    } nodes;
    
    DDFilteredView(const DDDetector*);
    
    //! The logical-part of the current node in the filtered-view
    const DDVolume & volume() const;
    
    //! The absolute translation of the current node
    const DDTranslation & translation() const;
    
    //! The absolute rotation of the current node
    const DDRotationMatrix & rotation() const;

    //! User specific data
    void mergedSpecifics(DDSpecParRefs const&);
    
    //! set the current node to the first child
    bool firstChild();

    //! set the current node to the next sibling
    bool nextSibling();

    //! set current node to the next node in the filtered tree
    bool next();
    
    //! The list of ancestors up to the root-node of the current node
    const DDGeoHistory & geoHistory() const;

    bool accepted(std::string_view, std::string_view) const;
    bool accepted(std::vector<std::string_view> const&, std::string_view) const;
    bool acceptedM(std::vector<std::string_view>&, std::string_view) const;
    std::vector<std::string_view> const& topNodes() const { return topNodes_; }
    std::vector<std::string_view> const& nextNodes() const { return nextNodes_; }
    std::vector<std::string_view> const& afterNextNodes() const { return afterNextNodes_; }
    
    std::vector<double> extractParameters() const;
    std::vector<std::string_view> paths(const char*) const;
    bool checkPath(std::string_view, TGeoNode *);
    bool checkNode(TGeoNode *);
    void unCheckNode();
    void filter(DDSpecParRefs&, std::string_view, std::string_view) const;
    std::vector<std::string_view> vPathsTo(const DDSpecPar&, unsigned int) const;
    std::vector<std::string_view> tails(const std::vector<std::string_view>& fullPath) const;

    DDGeoHistory parents_;
    
  private:
    const DDSpecParRegistry* registry_;
    
    bool isRegex(std::string_view) const;
    int contains(std::string_view, std::string_view) const;
    std::string_view realTopName(std::string_view input) const;
    int copyNo(std::string_view input) const;
    std::string_view noCopyNo(std::string_view input) const;
    std::string_view noNamespace(std::string_view input) const;
    std::vector<std::string_view> split(std::string_view, const char*) const;
    bool acceptRegex(std::string_view, std::string_view) const;
    
    std::vector<std::string_view> topNodes_;
    std::vector<std::string_view> nextNodes_;
    std::vector<std::string_view> afterNextNodes_;

    TGeoVolume *topVolume_ = nullptr;
    TGeoNode *node_ = nullptr;
  };
}

#endif
