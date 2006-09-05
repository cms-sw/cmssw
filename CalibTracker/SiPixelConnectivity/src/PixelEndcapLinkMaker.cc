#include "CalibTracker/SiPixelConnectivity/interface/PixelEndcapLinkMaker.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <ostream>

using namespace std;
using namespace sipixelobjects;

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

    if (e->pannelName() == 1) {
      if (e->plaquetteName() == 1)      { rocIds = Range(0,1); type = v1x2; }
      else if (e->plaquetteName() == 2) { rocIds = Range(0,5); type = v2x3; }
      else if (e->plaquetteName() == 3) { rocIds = Range(0,7); type = v2x4; }
      else if (e->plaquetteName() == 4) { rocIds = Range(0,4); type = v1x5; }
      else { edm::LogError("PixelEndcapLinkMaker")<< " *** UNEXPECTED roc: " << e->name() ; }
    }
    else {
      if (e->plaquetteName() == 1)      { rocIds = Range(0,5); type = v2x3; }
      else if (e->plaquetteName() == 2) { rocIds = Range(0,7); type = v2x4; }
      else if (e->plaquetteName() == 3) { rocIds = Range(0,9); type = v2x5; }
      else { edm::LogError("PixelEndcapLinkMaker")<< " *** UNEXPECTED roc: " << e->name() ; }
    }
    item.rocIds = rocIds;
    item.type = type;
    linkItems.push_back(item);
  }
  //
  // sort names to get the order as in links
  //

  sort( linkItems.begin(), linkItems.end(), Order() );

  //
  // DEBUG
  //
  ostringstream str;
  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    str << (*it).name->name() <<" r="<< (*it).rocIds << endl;
  }
  LogDebug(" sorted ENDCAP links: ") << str.str();


  result.reserve(36);
  int lastPannelId = -1;
  int idLink = -1;
  int idRoc = -1;
  PixelFEDLink link(idLink); // dummy object, id=-1

  for (CIU it = linkItems.begin(); it != linkItems.end(); it++) {
    PixelFEDLink::ROCs rocs;
    int pannelId = it->name->pannelName();

    if ( pannelId != lastPannelId ) {
      lastPannelId = pannelId;
      if (idLink >= 0) result.push_back(link);
      idRoc = -1;
      link = PixelFEDLink(++idLink); // real link, to be filled
    }
    for (int id = (*it).rocIds.min(); id <= (*it).rocIds.max(); id++) {
      int rocInX, rocInY;
      if ( (*it).type < v2x3) {    //narrow modules
        rocInX = 0;
        rocInY = id;
      }
      else {
        if (2*id < (*it).rocIds.max()) {
          rocInX = 0;
          rocInY = id;
        }
        else {
          rocInX = 1;
          rocInY = (*it).rocIds.max()-id;
        }
      }
      rocs.push_back( PixelROC( it->unit, id, ++idRoc, rocInX, rocInY) );
    }
    PixelFEDLink::Connection connection = {it->unit, it->type, it->name->name(), it->rocIds, rocs};
    link.add( connection);
  }
  if (idLink >= 0) result.push_back(link);
  return result;
}



