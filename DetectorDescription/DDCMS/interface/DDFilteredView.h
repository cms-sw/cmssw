#ifndef DETECTOR_DESCRIPTION_DD_FILTERED_VIEW_H
#define DETECTOR_DESCRIPTION_DD_FILTERED_VIEW_H

// -*- C++ -*-
//
// Package:    DetectorDescription/Core
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
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DetectorDescription/DDCMS/interface/ExpandedNodes.h"
#include "DetectorDescription/DDCMS/interface/Filter.h"
#include <DD4hep/Volumes.h>
#include <memory>
#include <vector>

namespace cms {

  struct DDSolid {
    explicit DDSolid(dd4hep::Solid s) : solid_(s) {}
    dd4hep::Solid solid() const { return solid_; }
    dd4hep::Solid solidA() const;
    dd4hep::Solid solidB() const;
    const std::vector<double> parameters() const;

  private:
    dd4hep::Solid solid_;
  };

  class DDDetector;
  class DDCompactView;

  using Volume = dd4hep::Volume;
  using PlacedVolume = dd4hep::PlacedVolume;
  using ExpandedNodes = cms::ExpandedNodes;
  using Filter = cms::Filter;
  using Iterator = TGeoIterator;
  using Node = TGeoNode;
  using Translation = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double>>;
  using RotationMatrix = ROOT::Math::Rotation3D;
  using DDFilter = std::string_view;

  class DDFilteredView {
  public:
    using nav_type = std::vector<int>;

    DDFilteredView(const DDDetector*, const Volume);
    DDFilteredView(const DDCompactView&, const DDFilter& = "");
    DDFilteredView() = delete;

    //! The numbering history of the current node
    const ExpandedNodes& history();

    //! The physical volume of the current node
    const PlacedVolume volume() const;

    //! The full path to the current node
    const std::string path() const;

    //! The list of the volume copy numbers
    //  along the full path to the current node
    const std::vector<int> copyNos() const;

    const std::vector<int> copyNumbers() { return copyNos(); }

    //! The absolute translation of the current node
    // Return value is Double_t translation[3] with x, y, z elements.
    const Double_t* trans() const;
    const Translation translation() const;

    //! The absolute rotation of the current node
    const Double_t* rot() const;
    const RotationMatrix rotation() const;
    void rot(dd4hep::Rotation3D& matrixOut) const;

    //! User specific data
    void mergedSpecifics(DDSpecParRefs const&);

    //! set the current node to the first child
    bool firstChild();

    //! set the current node to the next sibling
    bool nextSibling();

    //! set the current node to the next sub sibling
    bool sibling();

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

    // Shape of current node
    bool isABox() const;
    bool isAConeSeg() const;
    bool isAPseudoTrap() const;
    bool isATrapezoid() const;
    bool isATruncTube() const;
    bool isATubeSeg() const;
    bool isASubtraction() const;

    // Get shape pointer of current node.
    // Caller must check that current node matches desired type
    // before calling this function.

    template <class Shape>
    const Shape* getShapePtr() const {
      Volume currVol = node_->GetVolume();
      return (dynamic_cast<Shape*>(currVol->GetShape()));
    }

    template <class Shape>
    bool isA() const {
      return dd4hep::isA<Shape>(solid());
    }

    dd4hep::Solid solid() const;

    // Name of current node
    std::string_view name() const;

    // Copy number of current node
    unsigned short copyNum() const;

    // Material name of current node
    std::string_view materialName() const;

    //! extract shape parameters
    const std::vector<double> parameters() const;

    const cms::DDSolidShape shape() const;

    // Convert new DD4hep shape id to an old DD one
    LegacySolidShape legacyShape(const cms::DDSolidShape shape) const;

    //! extract attribute value
    template <typename T>
    T get(const std::string&) const;

    //! extract attribute value in SpecPar
    template <typename T>
    T get(const std::string&, const std::string&) const;

    std::string_view getString(const std::string&) const;

    //! return the stack of sibling numbers which indicates
    //  the current position in the DDFilteredView
    nav_type navPos() const;

  private:
    bool accept(std::string_view);
    bool addPath(Node* const);
    bool addNode(Node* const);
    const TClass* getShape() const;

    //! set the current node to the first sibling
    bool firstSibling();

    ExpandedNodes nodes_;
    std::vector<Iterator> it_;
    std::vector<std::unique_ptr<Filter>> filters_;
    Filter* currentFilter_ = nullptr;
    Node* node_ = nullptr;
    const DDSpecParRegistry* registry_;
    DDSpecParRefs refs_;
  };
}  // namespace cms

#endif
