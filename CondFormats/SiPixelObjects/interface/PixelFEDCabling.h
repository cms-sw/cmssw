#ifndef PixelFEDCabling_H
#define PixelFEDCabling_H

/** \class PixelFEDCabling
 *  Represents Pixel FrontEndDriver. 
 *  Owns links (of type PixelFEDCablingLink) attached to FED
 */

#include <vector>

#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
class PixelModuleName;

namespace sipixelobjects {

class PixelFEDCabling {
public:

  typedef std::vector<PixelFEDLink> Links;
  
  PixelFEDCabling(unsigned int id = 0) : theFedId(id) { }

  void setLinks(Links & links);

  void addLink(const PixelFEDLink & link);

  /// return link identified by id. Link id's are ranged [1, numberOfLinks]
  const PixelFEDLink * link(unsigned int id) const 
    { return (id > 0 && id <= theLinks.size()) ? &theLinks[id-1] : 0; }

  /// number of links in FED
  unsigned int numberOfLinks() const { return theLinks.size(); }

  unsigned int id() const { return theFedId; } 

  std::string print(int depth = 0) const;

  void  addItem(unsigned int linkId, const PixelROC & roc);

  /// check link numbering consistency, ie. that link position in vector
  /// is the same as its id. Futhermore it checks numbering consistency for
  /// ROCs belonging to Link. 
  bool checkLinkNumbering() const;
private:
  
private:

  unsigned int   theFedId;
  Links theLinks;

}; 
}

#endif
