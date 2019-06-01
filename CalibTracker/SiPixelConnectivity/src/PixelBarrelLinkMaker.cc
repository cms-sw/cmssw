#include "CalibTracker/SiPixelConnectivity/interface/PixelBarrelLinkMaker.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <ostream>
using namespace std;
using namespace sipixelobjects;

bool PixelBarrelLinkMaker::Order::operator()(const Item& u1, const Item& u2) const {
  const PixelBarrelName& n1 = *u1.name;
  const PixelBarrelName& n2 = *u2.name;

  bool res = false;

  if (n1.layerName() < n2.layerName())
    res = true;
  else if (n1.layerName() > n2.layerName())
    res = false;
  else if (n1.ladderName() < n2.ladderName())
    res = true;
  else if (n1.ladderName() > n2.ladderName())
    res = false;
  else if (abs(n1.moduleName()) < abs(n2.moduleName()))
    res = true;
  else if (abs(n1.moduleName()) > abs(n2.moduleName()))
    res = false;
  else if (u1.rocIds.min() < u2.rocIds.min())
    res = true;
  else if (u1.rocIds.min() > u2.rocIds.min())
    res = false;

  return res;
}

PixelBarrelLinkMaker::Links PixelBarrelLinkMaker::links(const Names& n, const DetUnits& u) const {
  Links result;
  typedef Names::const_iterator CIN;

  //
  // construct link items from names.
  // the item is equivalent to name for layer=3.
  // for layer=1,2 each module has 2 links
  //
  vector<Item> linkItems;
  typedef vector<Item>::const_iterator CIU;

  for (unsigned int idx = 0; idx < n.size(); idx++) {
    Item item;
    PixelBarrelName* b = dynamic_cast<PixelBarrelName*>(n[idx]);
    uint32_t d = u[idx];
    item.name = b;
    item.unit = d;

    if (b->isHalfModule()) {
      item.rocIds = Range(0, 7);  // half modules
      linkItems.push_back(item);
    } else if (b->layerName() <= 2) {
      item.rocIds = Range(0, 7);  // first link for modules in Layer=1,2
      linkItems.push_back(item);
      item.rocIds = Range(8, 15);  // second link for modules in Layer=1,2
      linkItems.push_back(item);
    } else {
      item.rocIds = Range(0, 15);  // one module per link
      linkItems.push_back(item);
    }
  }

  //
  // sort link items to get the order as in links
  //

  Order myLess;
  sort(linkItems.begin(), linkItems.end(), myLess);

  //
  // DEBUG
  //
  ostringstream str;
  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    str << (*it).name->name() << " r=" << (*it).rocIds << endl;
  }
  LogDebug(" sorted BARREL links: ") << str.str();

  //
  // create corresponding PixelROC and link
  //
  int idLink = 0;
  result.reserve(linkItems.size());
  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    PixelFEDLink::ROCs rocs;
    PixelFEDLink link(++idLink);
    int idRoc = 0;
    for (int id = (*it).rocIds.min(); id <= (*it).rocIds.max(); id++) {
      idRoc++;
      rocs.push_back(PixelROC(it->unit, id, idRoc));
    }
    link.add(rocs);
    result.push_back(link);
  }

  return result;
}
