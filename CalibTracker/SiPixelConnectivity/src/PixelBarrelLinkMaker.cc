#include "CalibTracker/SiPixelConnectivity/interface/PixelBarrelLinkMaker.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"

using namespace std;
bool PixelBarrelLinkMaker::Order::operator() 
    (const Item &u1, const Item& u2) const
{
  const PixelBarrelName & n1 = *u1.name;
  const PixelBarrelName & n2 = *u2.name;

  bool res = false;

       if ( n1.layerName() < n2.layerName() ) res = true;
  else if ( n1.layerName() > n2.layerName() ) res = false;
  else if ( n1.ladderName() < n2.ladderName() ) res = true;
  else if ( n1.ladderName() > n2.ladderName() ) res = false;
  else if ( abs(n1.moduleName()) < abs(n2.moduleName()) ) res =  true;
  else if ( abs(n1.moduleName()) > abs(n2.moduleName()) ) res =  false;
  else if ( u1.rocIds.min() < u2.rocIds.min() ) res = true;
  else if ( u1.rocIds.min() > u2.rocIds.min() ) res = false;

  return res;
}

PixelBarrelLinkMaker::Links PixelBarrelLinkMaker::links( 
    const Names & n, const DetUnits & u) const
{

  Links result;
  typedef Names::const_iterator CIN;

  //
  // construct link items from names. 
  // the item is equivalent to name for layer=3.
  // for layer=1,2 each module has 2 links
  //
  vector<Item> linkItems;
  typedef vector<Item>::const_iterator CIU;

  for(unsigned int idx = 0; idx < n.size(); idx++) {
    Item item;
    PixelBarrelName * b = dynamic_cast<PixelBarrelName * >(n[idx]);
    uint32_t d = u[idx];
    item.name = b;
    item.unit = d;
/*
    if (b->layerName() <= 2) {
      item.rocIds = Range(0,7);      // first link for modules in Layer=1,2
      linkItems.push_back(item);
      item.rocIds = Range(8,15);     // second link for modules in Layer=1,2
      linkItems.push_back(item);
    } else  {
      item.rocIds = Range(0,15);
      linkItems.push_back(item);     // link for modules in layer = 3 
    }
*/
      item.rocIds = Range(0,15);   // FIXME temporary!
      linkItems.push_back(item);     
  }



  //
  // sort link items to get the order as in links
  //
  
  Order myLess;
  sort( linkItems.begin(), linkItems.end(), myLess );

  bool debug = false;
  if (debug) {
    cout  << " sorted: " << endl;
    for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
      cout << (*it).name->name() <<" r="<< (*it).rocIds << endl;
    }
    cout << endl;
  }


  //
  // create corresponding PixelROC and link
  //
  int idLink = -1;
  result.reserve(linkItems.size());
  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    PixelFEDLink::ROCs rocs; 
    PixelFEDLink * link = new PixelFEDLink(++idLink, theOwner);
    int idRoc = -1;
    for (int id = (*it).rocIds.min(); id <= (*it).rocIds.max(); id++) {
       //
       //
       int rocInX, rocInY;
       if (it->name->moduleName() < 0) { // negative barrel
         if (id <= 7) {
           rocInX = id;
           rocInY = 0;
         } 
         else {
           rocInX = 15-id;
           rocInY = 1;
         } 
       } else {
         if (id <=7) {
           rocInX = 7-id; 
           rocInY = 1;
         }
         else {
           rocInX = id-8;
           rocInY = 0;
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
    result.push_back(link); 
  }

  return result;
}
