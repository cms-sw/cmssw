#ifndef DD_DDNodeSelector_h 
#define DD_DDNodeSelector_h

#include <string>

/** given a part selection string, a node selector calculates all expanded nodes in the geometry tree */
class DDNodeSelector
{
public:
  
  /** The compact view given in the constructor represents the geometry on which the node selector works */
  explicit DDNodeSelector(const DDCompactView &);
  
  /** you can derive your own node selector */
  virtual ~DDNodeSelector();
  
  /** sets the selection string from which all expanded nodes in the geometry tree should be
      calculated */
  bool setPartSelection(const std::string &, bool useRegex=false);

  /** the current node matching the selection string; copy it, if you want to keep it! */
  const DDGeoHistory & current() const;
  
  /** the next node matching the selection string, returns false, if there is no next node */
  bool next();
  
  
};

#endif
