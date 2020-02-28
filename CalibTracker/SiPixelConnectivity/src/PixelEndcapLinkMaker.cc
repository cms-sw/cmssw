#include "CalibTracker/SiPixelConnectivity/interface/PixelEndcapLinkMaker.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <ostream>

using namespace std;
using namespace sipixelobjects;

bool PixelEndcapLinkMaker::Order::operator()(const Item& u1, const Item& u2) const {
  bool res = true;
  const PixelEndcapName& n1 = *u1.name;
  const PixelEndcapName& n2 = *u2.name;

  if (n1.halfCylinder() < n2.halfCylinder())
    res = true;
  else if (n1.halfCylinder() > n2.halfCylinder())
    res = false;
  else if (n1.diskName() < n2.diskName())
    res = true;
  else if (n1.diskName() > n2.diskName())
    res = false;
  else if (n1.bladeName() < n2.bladeName())
    res = true;
  else if (n1.bladeName() > n2.bladeName())
    res = false;
  else if (n1.pannelName() < n2.pannelName())
    res = true;
  else if (n1.pannelName() > n2.pannelName())
    res = false;
  else if (n1.plaquetteName() < n2.plaquetteName())
    res = true;
  else if (n1.plaquetteName() > n2.plaquetteName())
    res = false;

  return res;
}

PixelEndcapLinkMaker::Links PixelEndcapLinkMaker::links(const Names& n, const DetUnits& u) const {
  Links result;
  typedef Names::const_iterator CIN;

  //
  // split names to links
  //
  vector<Item> linkItems;
  typedef vector<Item>::const_iterator CIU;

  for (unsigned int idx = 0; idx < n.size(); idx++) {
    Item item;
    PixelEndcapName* e = dynamic_cast<PixelEndcapName*>(n[idx]);
    uint32_t d = u[idx];
    item.name = e;
    item.unit = d;
    Range rocIds(-1, -1);
    PixelModuleName::ModuleType type = e->moduleType();
    switch (type) {
      case (PixelModuleName::v1x2): {
        rocIds = Range(0, 1);
        break;
      }
      case (PixelModuleName::v1x5): {
        rocIds = Range(0, 4);
        break;
      }
      case (PixelModuleName::v2x3): {
        rocIds = Range(0, 5);
        break;
      }
      case (PixelModuleName::v2x4): {
        rocIds = Range(0, 7);
        break;
      }
      case (PixelModuleName::v2x5): {
        rocIds = Range(0, 9);
        break;
      }
      default:
        edm::LogError("PixelEndcapLinkMaker") << " *** UNEXPECTED roc: " << e->name();
    };
    item.rocIds = rocIds;
    linkItems.push_back(item);
  }
  //
  // sort names to get the order as in links
  //

  sort(linkItems.begin(), linkItems.end(), Order());

  //
  // DEBUG
  //
  ostringstream str;
  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    str << (*it).name->name() << " r=" << (*it).rocIds << endl;
  }
  LogDebug(" sorted ENDCAP links: ") << str.str();

  result.reserve(36);
  int lastPannelId = -1;
  int idLink = 0;
  int idRoc = 0;
  PixelFEDLink link(idLink);  // dummy object, id=0

  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    PixelFEDLink::ROCs rocs;
    int pannelId = it->name->pannelName();

    if (pannelId != lastPannelId) {
      lastPannelId = pannelId;
      if (idLink > 0)
        result.push_back(link);
      idRoc = 0;
      link = PixelFEDLink(++idLink);  // real link, to be filled
    }

    for (int id = (*it).rocIds.min(); id <= (*it).rocIds.max(); id++) {
      ++idRoc;
      rocs.push_back(PixelROC(it->unit, id, idRoc));
    }

    link.add(rocs);
  }

  if (idLink > 0)
    result.push_back(link);
  return result;
}
