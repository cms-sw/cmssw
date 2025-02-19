#ifndef GUARD_RecoIdealGeometry_H
#define GUARD_RecoIdealGeometry_H

#include <vector>
#include <algorithm>
#include <cassert>

#include <DataFormats/DetId/interface/DetId.h>

/** @class 
 *
 *  @author:  Michael Case               Initial Version
 *  @version: 0.0
 *  @date:    21 May 2007
 * 
 *  Description:
 *  
 *  Users are expected to read the publicly available data members themselves
 *  though insert is provided for convenience when adding to this container.
 * 
 *  2009-11-19:  Removed full vector copies and instead force the use of *Start and *End methods
 *               which return a const_iterator for the range of elements for each "vector"
 *
 */

class RecoIdealGeometry {
 public:

  RecoIdealGeometry() { }
  ~RecoIdealGeometry() { }

  bool insert( DetId id, const std::vector<double>& trans, const std::vector<double>& rot, const std::vector<double>& pars ) {
    if ( trans.size() != 3 || rot.size() != 9 ) return false;
    pDetIds.push_back(id);
    pNumShapeParms.push_back(pars.size()); // number of shape specific parameters.
    pParsIndex.push_back(pPars.size()); // start of this guys "blob"
    pPars.reserve(pPars.size() + trans.size() + rot.size() + pars.size());
    std::copy ( trans.begin(), trans.end(), std::back_inserter(pPars));
    std::copy ( rot.begin(), rot.end(), std::back_inserter(pPars));
    std::copy ( pars.begin(), pars.end(), std::back_inserter(pPars));
    return true;
  }

  bool insert( DetId id, const std::vector<double>& trans, const std::vector<double>& rot, const std::vector<double>& pars, const std::vector<std::string>& spars ) {
    if ( trans.size() != 3 || rot.size() != 9 ) return false;
    pDetIds.push_back(id);
    pNumShapeParms.push_back(pars.size()); // number of shape specific parameters.
    pParsIndex.push_back(pPars.size()); // start of this guys "blob"
    pPars.reserve(pPars.size() + trans.size() + rot.size() + pars.size());
    std::copy ( trans.begin(), trans.end(), std::back_inserter(pPars));
    std::copy ( rot.begin(), rot.end(), std::back_inserter(pPars));
    std::copy ( pars.begin(), pars.end(), std::back_inserter(pPars));

    sNumsParms.push_back(spars.size());
    sParsIndex.push_back(strPars.size());
    strPars.reserve(strPars.size()+spars.size());
    std::copy ( spars.begin(), spars.end(), std::back_inserter(strPars));
    return true;
  }

  size_t size() { 
    assert ( (pDetIds.size() == pNumShapeParms.size()) && (pNumShapeParms.size() == pParsIndex.size()) );
    return pDetIds.size(); 
  }

  // HOW to use this stuff... first, get hold of the reference to the detIds like:
  // const std::vector<double>& myds = classofthistype.detIds()
  // Then iterate over the detIds using 
  // for ( size_t it = 0 ; it < myds.size(); ++it ) 
  // and ask for the parts ...
  // {
  //   std::vector<double>::const_iterator xyzB = classofthistype.transStart(it);
  //   std::vector<double>::const_iterator xyzE = classofthistype.transEnd(it);
  // }
  const std::vector<DetId>& detIds () const {
    return pDetIds;
  }

  std::vector<double>::const_iterator tranStart( size_t ind ) const { 
     return pPars.begin() + pParsIndex[ind]; 
  }

  std::vector<double>::const_iterator tranEnd ( size_t ind ) const { 
     return pPars.begin() + pParsIndex[ind] + 3; 
  }

  std::vector<double>::const_iterator rotStart ( size_t ind ) const {
    return pPars.begin() + pParsIndex[ind] + 3;
  }

  std::vector<double>::const_iterator rotEnd ( size_t ind ) const {
    return pPars.begin() + pParsIndex[ind] + 3 + 9;
  }

  std::vector<double>::const_iterator shapeStart ( size_t ind ) const {
    return pPars.begin() + pParsIndex[ind] + 3 + 9;
  }

  std::vector<double>::const_iterator shapeEnd ( size_t ind ) const {
    return pPars.begin() + pParsIndex[ind] + 3 + 9 + pNumShapeParms[ind];
  }

  std::vector<std::string>::const_iterator strStart ( size_t ind ) const {
    return strPars.begin() + sParsIndex[ind];
  }

  std::vector<std::string>::const_iterator strEnd ( size_t ind ) const {
    return strPars.begin() + sParsIndex[ind] + sNumsParms[ind];
  }


 private:    
  // translation always 3; rotation 9 for now; pars depends on shape_type.
  std::vector<DetId> pDetIds;

  std::vector<double> pPars;  // trans, rot then shape parms.
  // 0 for first pDetId, 3 + 9 + number of shape parameters for second & etc.
  // just save pPars size BEFORE adding next stuff.
  std::vector<int> pParsIndex; 
  std::vector<int> pNumShapeParms; // save the number of shape parameters.

  std::vector<std::string> strPars;
  std::vector<int> sParsIndex;
  std::vector<int> sNumsParms;
};

#endif

