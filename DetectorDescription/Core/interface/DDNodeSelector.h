#ifndef DD_DDNodeSelector_h 
#define DD_DDNodeSelector_h

#include <string>

/** given a part selection std::string, a node selector calculates all expanded nodes in the geometry tree */
class DDNodeSelector
{
public:
  
  /** The compact view given in the constructor represents the geometry on which the node selector works */
  explicit DDNodeSelector(const DDCompactView &);
  
  /** you can derive your own node selector */
  virtual ~DDNodeSelector();
  
  /** sets the selection std::string from which all expanded nodes in the geometry tree should be
      calculated */
  bool setPartSelection(const std::string &, bool useRegex=false);

  /** the current node matching the selection std::string; copy it, if you want to keep it! */
  const DDGeoHistory & current() const;
  
  /** the next node matching the selection std::string, returns false, if there is no next node */
  bool next();
  
  
};

#endif
