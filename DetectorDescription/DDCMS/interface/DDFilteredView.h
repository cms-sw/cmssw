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
#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include <DD4hep/Volumes.h>
#include <memory>
#include <vector>

namespace cms {

  class DDDetector;

  using Volume = dd4hep::Volume;
  using PlacedVolume = dd4hep::PlacedVolume;
  
  class DDFilteredView {

  public:
    
    DDFilteredView(const DDDetector*, const Volume);
    DDFilteredView() = delete;
    
    struct ExpandedNodes {
      std::vector<double> tags;
      std::vector<double> offsets;
      std::vector<int> copyNos;
    };

    struct Filter {
      std::vector<std::string_view> keys;
      struct std::unique_ptr<Filter> next;
      struct Filter* up;
    };
    
    //! The numbering history of the current node
    const ExpandedNodes& history() const {
      return nodes_;
    }
    
    //! The physical volume of the current node
    const PlacedVolume volume() const;
    
    //! The absolute translation of the current node
    const Double_t* trans() const;
    
    //! The absolute rotation of the current node
    const Double_t* rot() const;

    //! User specific data
    void mergedSpecifics(DDSpecParRefs const&);
    
    //! set the current node to the first child
    bool firstChild();

    //! set the current node to the first sibling
    bool firstSibling();

    //! set the current node to the next sibling
    bool nextSibling();

    //! set the current node to the next sub sibling
    bool sibling();
    bool siblingNoCheck();
    
    //! count the number of children matching selection
    bool checkChild();
    
    //! set the current node to the parent node ...
    bool parent();

    //! set current node to the next node in the filtered tree
    bool next(int);

    //! set current node to the child node in the filtered tree
    void down();

    //! set current node to the parent node in the filtered tree
    void up();
   
    //! pop current node
    void unCheckNode();

    //! extract shape parameters
    std::vector<double> extractParameters() const;

  private:
    
    ExpandedNodes nodes_;
    std::vector<std::unique_ptr<Filter>> filters_;
    Filter* currentFilter_ = nullptr;

    bool accepted(std::string_view);
    bool accepted(std::string_view, std::string_view) const;
    bool accepted(std::vector<std::string_view> const&, std::string_view) const;
    bool addPath(std::string_view, TGeoNode* const);
    bool addNode(TGeoNode* const);

    const DDSpecParRegistry* registry_;
    
    bool isRegex(std::string_view) const;
    int contains(std::string_view, std::string_view) const;
    std::string_view realTopName(std::string_view input) const;
    int copyNo(std::string_view input) const;
    std::string_view noCopyNo(std::string_view input) const;
    std::string_view noNamespace(std::string_view input) const;
    std::vector<std::string_view> split(std::string_view, const char*) const;
    bool acceptRegex(std::string_view, std::string_view) const;
    
    TGeoNode *node_ = nullptr;
    std::vector<TGeoIterator> it_;
  };
}

#endif
