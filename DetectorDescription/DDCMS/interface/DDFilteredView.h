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

  //! Geometrical 'path' of the current node up to the root-node
  using DDGeoHistory = std::vector<DDExpandedNode>;
  using Volume = dd4hep::Volume;
    
  struct DDFilteredView {

    DDFilteredView(DDVolume const&, DDTranslation const&, DDRotationMatrix const&);
    
    //! The logical-part of the current node in the filtered-view
    const DDVolume & volume() const;
    
    //! The absolute translation of the current node
    const DDTranslation & translation() const;
    
    //! The absolute rotation of the current node
    const DDRotationMatrix & rotation() const;

    //! User specific data
    void mergedSpecifics(DDSpecParRefMap const&);
    
    //! set the current node to the first child
    bool firstChild();

    //! set the current node to the next sibling
    bool nextSibling();

    //! set current node to the next node in the filtered tree
    bool next();
    
    //! The list of ancestors up to the root-node of the current node
    const DDGeoHistory & geoHistory() const;

    bool accepted(std::string_view, std::string_view) const;
    bool accepted(std::vector<std::string_view>, std::string_view) const;
    std::vector<std::string_view> const& topNodes() const { return topNodes_; }
    
    std::vector<double> extractParameters(Volume) const;
    
  private:
    bool isRegex(std::string_view) const;
    int contains(std::string_view, std::string_view) const;
    std::string_view realTopName(std::string_view input) const;
    
    DDGeoHistory parents_;
    std::vector<std::string_view> topNodes_;
  };
}

#endif
