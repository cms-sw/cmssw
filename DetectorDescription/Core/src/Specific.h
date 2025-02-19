#ifndef DDI_Specific_h
#define DDI_Specific_h

#include "DetectorDescription/Core/interface/DDPartSelection.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"

#include <iostream>
#include <vector>
#include <string>
#include <utility>

class DDPartSelection;
class DDNodes;

namespace DDI {
  
  class Specific
  {
  public:
    typedef std::vector<DDPartSelection> partsel_type;
    typedef std::vector<std::string> selectors_type; 	     

    Specific(const std::vector<std::string> & selections,
             const DDsvalues_type & specs,
	      bool doRegex=true) ;
	      
    Specific(const std::vector<DDPartSelection> & selections,
             const DDsvalues_type & specs);	      
    
    ~Specific();// { } 
    	     
    const std::vector<DDPartSelection> & selection() const;
    
    void updateLogicalPart(std::vector<std::pair<DDLogicalPart, std::pair<DDPartSelection*,DDsvalues_type*> > >&) const;
    
    void tokenize();
    
    const DDsvalues_type & specifics() const { return specifics_; }
    
    //! CURRENTLY NOT IMPLEMENTED! 
    bool nodes(DDNodes &) const { return false; }
    
    //! gives the geometrical history of a fully specified PartSelector   
    std::pair<bool,DDExpandedView> node() const;
    
    void stream(std::ostream &) const;    
  protected:
  void createPartSelections(const std::string & selString);
  void addSelectionLevel(std::vector<DDLogicalPart> & lpv, int copyno, ddselection_type st, 
                       std::vector<DDPartSelection> & selv);
    //std::vector<std::string> selection_;
    DDsvalues_type specifics_;
    partsel_type partSelections_;
    bool valid_;    
    bool doRegex_;
  };

}

#endif // DDI_Specific_h
