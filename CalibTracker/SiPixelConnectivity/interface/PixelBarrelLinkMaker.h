#ifndef PixelBarrelLinkMaker_H
#define PixelBarrelLinkMaker_H

/** \class PixelBarrelLinkMaker
 * Assign barrel pixel modules (defined by name and unit) to links
 */

#include <vector>

#include "CalibTracker/SiPixelConnectivity/interface/TRange.h"

class PixelModuleName;
class PixelBarrelName;
#include "CondFormats/SiPixelObjects/interface/PixelFEDCabling.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include <cstdint>

class PixelBarrelLinkMaker {
public:
  typedef sipixelobjects::PixelFEDCabling PixelFEDCabling;
  typedef sipixelobjects::PixelFEDLink PixelFEDLink;
  typedef sipixelobjects::PixelROC PixelROC;

  typedef std::vector<PixelModuleName*> Names;
  typedef std::vector<uint32_t> DetUnits;
  typedef PixelFEDCabling::Links Links;
  typedef TRange<int> Range;

  /// ctor from owner
  PixelBarrelLinkMaker(const PixelFEDCabling* o) : theOwner(o) {}

  /// construct links
  /// Each barrel module triggers one or two link Items.
  /// They are sorted according to Order().
  /// The ROCs corresponding to items are created. The link is
  /// form from link items and ROCS.
  Links links(const Names& n, const DetUnits& u) const;

private:
  const PixelFEDCabling* theOwner;

  /// link item.
  /// defined by DetUnit with name (representing Pixel Detector module)
  /// and Ids of associated ROCs. The Id is the ROC number in module according
  /// to PixelDatabase.
  struct Item {
    const PixelBarrelName* name;
    uint32_t unit;
    Range rocIds;
  };

  /// define order of items in a link.
  /// Highest priority to layer id.
  /// Second priority for ladder id (phi).
  /// Third priority by abs(module) (ie. along |z|)
  /// If all equal id of ROC matters
  struct Order {
    bool operator()(const Item&, const Item&) const;
  };

private:
};

#endif
