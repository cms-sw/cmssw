#ifndef PixelFEDLink_H
#define PixelFEDLink_H

/** \class PixelFEDLink
 * Represents Link connected to PixelFED. Owns ROCs
 */

#include <utility>
#include <vector>
#include <iostream>

#include "CondFormats/SiPixelObjects/interface/PixelROC.h"
class PixelModuleName;

namespace sipixelobjects {

  class PixelFEDLink {
  public:
    /// ROCs served be this link
    typedef std::vector<PixelROC> ROCs;

    /// ctor with id of link and parent FED
    PixelFEDLink(unsigned int id = 0) : theId(id) {}

    ///  add connection (defined by connection spec and ROCs)
    void add(const ROCs& rocs);

    /// link id
    unsigned int id() const { return theId; }

    /// number of ROCs in fed
    unsigned int numberOfROCs() const { return theROCs.size(); }

    /// return ROC identified by id. ROC ids are ranged [1,numberOfROCs]
    const PixelROC* roc(unsigned int id) const { return (id > 0 && id <= theROCs.size()) ? &theROCs[id - 1] : nullptr; }

    /// check ROC in link numbering consistency, ie. that ROC position in
    /// vector is the same as its id. To be called by owner
    bool checkRocNumbering() const;

    std::string print(int depth = 0) const;

    void addItem(const PixelROC& roc);

  private:
    unsigned int theId;
    ROCs theROCs;
    std::string printForMap() const;
  };

}  // namespace sipixelobjects

//std::ostream & operator<<( std::ostream& out, const PixelFEDLink & l);

#endif
