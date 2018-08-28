#ifndef DDI_Specific_h
#define DDI_Specific_h

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDExpandedView;
class DDLogicalPart;
class DDNodes;
class DDPartSelection;

namespace DDI {
  
  class Specific
  {
  public:
    Specific(const std::vector<std::string> & selections,
             const DDsvalues_type & specs,
	     bool doRegex=true) ;
	      
    Specific(const std::vector<DDPartSelection> & selections,
             const DDsvalues_type & specs);	      
    
    const std::vector<DDPartSelection> & selection() const;
    
    void updateLogicalPart(std::vector<std::pair<DDLogicalPart, std::pair<const DDPartSelection*, const DDsvalues_type*> > >&) const;
    
    void tokenize();
    
    const DDsvalues_type & specifics() const { return specifics_; }
    
    //! gives the geometrical history of a fully specified PartSelector   
    std::pair<bool,DDExpandedView> node() const;
    
    void stream(std::ostream &) const;    
  protected:
    void createPartSelections(const std::string & selString);
    void addSelectionLevel(std::vector<DDLogicalPart> & lpv, int copyno, ddselection_type st, 
			   std::vector<DDPartSelection> & selv);
    DDsvalues_type specifics_;
    std::vector<DDPartSelection> partSelections_;
    bool valid_;    
    bool doRegex_;
  };
}

#endif // DDI_Specific_h
