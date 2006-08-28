#ifndef PixelFEDCabling_H
#define PixelFEDCabling_H

/** \class PixelFEDCabling
 *  Represents Pixel FrontEndDriver. 
 *  Owns links (of type PixelFEDCablingLink) attached to FED
 */

#include <vector>

#include "CondFormats/SiPixelObjects/interface/PixelFEDLink.h"
class PixelModuleName;
class PixelROC;

class PixelFEDCabling {
public:
  typedef std::vector<PixelFEDLink> Links;
  
  PixelFEDCabling(int id = -1) : theFedId(id) { }

  void setLinks(Links & links);

  void addLink(const PixelFEDLink & link);

  /// return link identified by id. Link id's are ranged [0, numberOfLinks)
  const PixelFEDLink * link(unsigned int id) const 
    { return (id >= 0 && id < theLinks.size()) ? &theLinks[id] : 0; }

  /// number of links in FED
  int numberOfLinks() const { return theLinks.size(); }

  int id() const { return theFedId; } 

  std::string print(int depth = 0) const;

private:
  /// check link numbering consistency, ie. that link position in vector
  /// is the same as its id. Futhermore it checks numbering consistency for
  /// ROCs belonging to Link. Called by constructor 
  bool checkLinkNumbering() const;
  
private:

  int   theFedId;
  Links theLinks;

}; 

#endif
