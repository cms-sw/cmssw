#ifndef PixelFEDLink_H
#define PixelFEDLink_H

/** \class PixelFEDLink
 * Represents Link connected to PixelFED. Owns ROCs
 */

#include <utility>
#include <vector>
#include <iostream>
#include <boost/cstdint.hpp>

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
class PixelModuleName;

namespace sipixelobjects {

class PixelFEDLink {
public:

  typedef std::pair<int,int> Range;

  typedef std::pair<unsigned int,unsigned int> ConnectionIndex;

  /// ROCs served be this link
  typedef std::vector<PixelROC> ROCs;

  /// specifies minimal object connected to link (ranges of ROCs in module)
  struct Connection { uint32_t unit; std::string name; Range range; ROCs rocs; };

  /// all objects connected to link
  typedef std::vector<Connection> Connections;

  /// ctor with id of link and parent FED
  PixelFEDLink(int id = -1) : theId(id) { } 

  ///  add connection (defined by connection spec and ROCs)
  void add(const Connection & con);

  /// link id
  int id() const { return theId; }

  /// number of ROCs in fed
  int numberOfROCs() const { return theIndices.size(); }

  /// return ROC identified by id. ROC ids are ranged [0,numberOfROCs)
  const PixelROC * roc(unsigned int id) const;

  const Connections & connected() const { return theConnections; }
  
  /// check ROC in link numbering consistency, ie. that ROC position in
  /// vector is the same as its id. To be called by owner
  bool checkRocNumbering() const;

  std::string print(int depth = 0) const;

private:
  int theId;
  std::vector<ConnectionIndex> theIndices;
  Connections theConnections;
};

}

//std::ostream & operator<<( std::ostream& out, const PixelFEDLink & l);

#endif
