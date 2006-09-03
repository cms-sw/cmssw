#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
#include "DataFormats/SiPixelDetId/interface/PixelModuleName.h"
#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include <sstream>
#include <iostream>

using namespace std;
using namespace sipixelobjects;


bool PixelFEDLink::checkRocNumbering() const
{
  bool result = true;
  for (int idx = 0; idx < numberOfROCs();  idx++) {
    int co = theIndices[idx].first;
    int ro = theIndices[idx].second;
    const PixelROC & roc = theConnections[co].rocs[ro];
    if (idx != roc.idInLink() ) {
      result = false;
      cout << "** PixelFEDLink, idInLink in ROC, expected: "
           << idx <<" has: "<< roc.idInLink() << endl;
    }
  }
  return result;
}

const PixelROC * PixelFEDLink::roc(unsigned int id) const
{
// return & theConnections[theIndices[id].first].rocs[theIndices[id].second];

 if (id < 0 || id >= theIndices.size()) return 0;
 const ConnectionIndex & conIdx = theIndices[id];

 if (conIdx.first < theConnections.size() &&
     conIdx.second < theConnections[conIdx.first].rocs.size() )
   return &(theConnections[conIdx.first].rocs[conIdx.second]);

  return 0;
}

void PixelFEDLink::add(const Connection & con)
{
  theConnections.push_back(con);
  unsigned int connection_id = theConnections.size()-1;
  unsigned int nrocs = con.rocs.size();
  for (unsigned int idx=0; idx < nrocs; idx++) 
    theIndices.push_back( ConnectionIndex(connection_id,idx) ); 
}

string PixelFEDLink::print(int depth) const
{
  ostringstream out;
  typedef Connections::const_iterator IT;

  int idx=-1;
  if (depth-- >=0 ) {
    out <<"====== PixelFEDLink, ID: "<<id()<< endl;
    for (IT ic=theConnections.begin(); ic != theConnections.end(); ic++) {
      out <<"       "<<(*ic).name
          <<",r=("<< (*ic).range.first<<","<<(*ic).range.second<<")"
          <<"  ids:";
      for (int i = (*ic).range.first; i <= (*ic).range.second; i++) {
        idx++;
        out <<"_" << idx;
      }
      out << endl;
      if (idx != numberOfROCs()) edm::LogError(" problem with ROC numbering!"); 
      const ROCs & rocs = (*ic).rocs;
      typedef ROCs::const_iterator CIR;
      for (CIR ir = rocs.begin(); ir != rocs.end(); ir++) out<< (ir)->print(depth); 
    }
    out <<"       total number of ROCs: "<< numberOfROCs() << endl;
  }
  out << endl;
  return out.str();

}

