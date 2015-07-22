#ifndef DDSpecifics_h
#define DDSpecifics_h

#include <map>
#include <string>
#include <vector>

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDBase.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"

class DDSpecifics;
class DDPartSelection;
class DDNodes;
namespace DDI { class Specific; }

typedef std::vector<std::string> selectors_type;

std::ostream & operator<<(std::ostream &, const std::vector<std::string> &);
std::ostream & operator<<(std::ostream &, const DDSpecifics &);

/**
  used to attach specific (user defined) data to nodes in the expanded view.
  - only a std::map<std::string,std::string> (std::map of name,value) 

*/
//! Interface to attach user specific data to nodes in the expanded-view
/** User data (currently only of type a \c std::map<std::string,std::string> ) can be attached
    to single nodes or set of nodes in the detector tree (represented in DDExpandedView).
    Nodes where user data has to be attached are selected by a very simplified XPath similar 
    notation.
    
    DDSpecifics are lightweighted reference-objects. For further information concerning
    reference-objects refere to the documentation of DDLogicalPart. 
*/
class DDSpecifics : public DDBase<DDName,DDI::Specific*>
{
  friend std::ostream & operator<<( std::ostream  &, const DDSpecifics &);

public:
  //! Creates a uninitialized reference-object (see DDLogicalPart documentation for details on reference objects)
  DDSpecifics();
     
  //! Creates a initialized reference-object or a reference to an allready defined specifcs.
  /** If a DDSpecifics with \a name was already defined, this constructor creates a 
      lightweighted reference-object to it. Otherwise a (default) initialized reference-object
      is registered named \a name. 
      For further details concerning the usage of reference-objects refere
      to the documentation of DDLogicalPart.
  */
  DDSpecifics(const DDName & name); 
  
  //! Creates a defined reference-object or replaces a already defined reference-object named \a name
  /**
      \arg \c name unique name 
      \arg \c partSelections collection of selection-strings which select expanded-nodes
      \arg \c svalues user data attached to nodes selected by \a partSelections
      
      <h3> Syntax of the selection std::string </h3>
      bla, bla, bla
  */
  DDSpecifics(const DDName & name,
              const std::vector<std::string> & partSelections,
	      const DDsvalues_type & svalues,
	      bool doRegex=true);
  
  ~DDSpecifics();
  
  //! Gives a reference to the collection of part-selections
  const std::vector<DDPartSelection> & selection() const;
  
  //! Reference to the user-data attached to all nodes selected by the selections-strings given through selection
  const DDsvalues_type & specifics() const;
  
  //! Calculates the geometrical history of a fully specified PartSelector
  std::pair<bool,DDExpandedView> node() const;
};

#endif
