#include "CalibTracker/SiPixelConnectivity/interface/PixelEndcapLinkMaker.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

using namespace std;
bool PixelEndcapLinkMaker::Order::operator()
    (const Item &u1, const Item& u2) const
{
  bool res = true;
  const PixelEndcapName & n1 = *u1.name;
  const PixelEndcapName & n2 = *u2.name;

  if (n1.endcapName() < n2.endcapName() ) res = true;
  else if(n1.endcapName() > n2.endcapName() ) res = false;
  else if (n1.diskName() < n2.diskName() ) res =  true;
  else if (n1.diskName() > n2.diskName() ) res =  false;
  else if (n1.bladeName() < n2.bladeName() ) res =  true;
  else if (n1.bladeName() > n2.bladeName() ) res =  false;
  else if (n1.pannelName() < n2.pannelName() ) res =  true;
  else if (n1.pannelName() > n2.pannelName() ) res =  false;
  else if (n1.plaquetteName() < n2.plaquetteName() ) res =  true;
  else if (n1.plaquetteName() > n2.plaquetteName() ) res =  false;

  return res;
}

PixelEndcapLinkMaker::Links PixelEndcapLinkMaker::links(
    const Names & n, const DetUnits & u) const
{
    
  Links result; 
  typedef Names::const_iterator CIN;

  //
  // split names to links
  //
  vector<Item> linkItems;
  typedef vector<Item>::const_iterator CIU;


  for(unsigned int idx = 0; idx < n.size(); idx++) {
    Item item;
    PixelEndcapName * e = dynamic_cast<PixelEndcapName * >(n[idx]);
    uint32_t d = u[idx];
    item.name = e;
    item.unit = d;
    Range rocIds(-1,-1);
    Plaquette type = v1x2;

    if (e->pannelName() == 0) {
      if (e->plaquetteName() == 0)      { rocIds = Range(0,1); type = v1x2; }
      else if (e->plaquetteName() == 1) { rocIds = Range(0,5); type = v2x3; }
      else if (e->plaquetteName() == 2) { rocIds = Range(0,7); type = v2x4; }
      else if (e->plaquetteName() == 3) { rocIds = Range(0,4); type = v1x5; }
      else { cout << " *** UNEXPECTED roc: " << e->name() << endl; }
    }
    else {
      if (e->plaquetteName() == 0)      { rocIds = Range(0,5); type = v2x3; }
      else if (e->plaquetteName() == 1) { rocIds = Range(0,7); type = v2x4; }
      else if (e->plaquetteName() == 2) { rocIds = Range(0,9); type = v2x5; }
      else { cout << " *** UNEXPECTED roc: " << e->name() << endl; }
    }
    item.rocIds = rocIds;
    item.type = type;
    linkItems.push_back(item);
  }
  //
  // sort names to get the order as in links
  //

  sort( linkItems.begin(), linkItems.end(), Order() );
  bool debug = false;
  if (debug) {
    cout  << " ** PixelEndcapLinkMaker ** sorted: " << endl;
    for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
      cout << (*it).name->name() <<" r="<< (*it).rocIds << endl;
    }
    cout << endl;
  }

  result.reserve(36);
  PixelFEDLink * link = 0;
  int lastPannelId = -1;
  int idLink = -1;
  int idRoc = -1;
  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    PixelFEDLink::ROCs rocs;
    int pannelId = it->name->pannelName();
    if (!link) {
      lastPannelId = pannelId;
      idRoc = -1;
      link = new PixelFEDLink(++idLink, theOwner);
    }
    if ( pannelId != lastPannelId ) {
      lastPannelId = pannelId;
      result.push_back(link);
      idRoc = -1;
      link = new PixelFEDLink(++idLink, theOwner);
    }
    for (int id = (*it).rocIds.min(); id <= (*it).rocIds.max(); id++) {
      int rocInY, rocInX;
      if ( (*it).type < v2x3) {    //narrow modules
        rocInY = 0;
        rocInX = (*it).rocIds.max()-id;
      }
      else {
        if (2*id < (*it).rocIds.max()) {
          rocInY = 0;
          rocInX = id;
        }
        else {
          rocInY = 1;
          rocInX = (*it).rocIds.max()-id;
        }
      }
      rocs.push_back(
           new PixelROC( it->unit, link, id, ++idRoc, rocInX, rocInY));
    }
    PixelFEDLink::Connection con;
    con.name = it->name;
    con.unit = it->unit;
    con.rocs = it->rocIds;
    link->add( con, rocs);
    rocs.clear();
  }
  if (link) result.push_back(link);




  return result;
}



