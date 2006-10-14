#ifndef PixelFEDLink_H
#define PixelFEDLink_H

/** \class PixelFEDLink
 * Represents Link connected to PixelFED. Owns ROCs
 */

#include <utility>
#include <vector>
#include <iostream>
#include <boost/cstdint.hpp>

class PixelModuleName;
class PixelFEDCabling;
class PixelROC;

class PixelFEDLink {
public:

  typedef std::pair<int,int> Range;
  typedef std::vector<PixelROC* > ROCs;

  /// specifies minimal object connected to link (ranges of ROCs in module)
  struct Connection { const PixelModuleName * name;
                      uint32_t unit;
                      Range rocs; };
  typedef std::vector<Connection> Connections;

  /// ctor with id of link and parent FED
  PixelFEDLink(int id, const PixelFEDCabling * fed) : theId(id), theFed(fed) { } 

  ~PixelFEDLink() { clearRocs(); }

  ///  add connection (defined by connection spec and ROCs)
  void add(const Connection & con, const ROCs & rocs) {
    theConnections.push_back(con);
    theROCs.insert( theROCs.end(), rocs.begin(), rocs.end() );
  }

  /// link id
  int id() const { return theId; }

  /// parent fed 
  const PixelFEDCabling * fed() const { return theFed; }

  /// number of ROCs in fed
  int numberOfROCs() const { return theROCs.size(); }

  /// return ROC identified by id. ROC ids are ranged [0,numberOfROCs)
  PixelROC * roc(unsigned int id) const 
    { return (id >= 0 && id < theROCs.size() ) ?  theROCs[id] : 0; }

  const Connections & connected() const { return theConnections; }
  
  /// check ROC in link numbering consistency, ie. that ROC position in
  /// vector is the same as its id. To be called by owner
  bool checkRocNumbering() const;

private:
  void clearRocs();

private:
  int theId;
  const PixelFEDCabling * theFed;
  ROCs theROCs;
  Connections theConnections;
};

std::ostream & operator<<( std::ostream& out, const PixelFEDLink & l);

#endif
