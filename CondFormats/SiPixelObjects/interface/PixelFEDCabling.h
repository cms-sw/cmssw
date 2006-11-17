#ifndef PixelFEDCabling_H
#define PixelFEDCabling_H

/** \class PixelFEDCabling
 *  Represents Pixel FrontEndDriver. 
 *  Owns links (of type PixelFEDCablingLink) attached to FED
 */

#include <vector>

class PixelModuleName;
class PixelFEDLink;
class PixelROC;

class PixelFEDCabling {
public:
  typedef std::vector<PixelModuleName *> ModuleNames;
  typedef std::vector<PixelFEDLink *> Links;
  
  PixelFEDCabling(int id, ModuleNames & names);
  virtual ~PixelFEDCabling();

  void setLinks(Links & links);

  /// return link identified by id. Link id's are ranged [0, numberOfLinks)
  PixelFEDLink * link(unsigned int id) const 
    { return (id >= 0 && id < theLinks.size()) ? theLinks[id] : 0; }

  /// number of links in FED
  int numberOfLinks() const { return theLinks.size(); }

  int id() const { return theFedId; } 

private:
  /// check link numbering consistency, ie. that link position in vector
  /// is the same as its id. Futhermore it checks numbering consistency for
  /// ROCs belonging to Link. Called by constructor 
  bool checkLinkNumbering() const;
  
  void clearLinks(bool warn = true);

private:

  int theFedId;
  ModuleNames theModuleNames;
  Links       theLinks;

}; 

#endif
