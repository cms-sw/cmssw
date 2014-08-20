#ifndef DDI_LogicalPart_h
#define DDI_LogicalPart_h

#include <iostream>
#include <vector>
#include <utility>
#include <map>

#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDEnums.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"

//class DDsvalues_type;
class DDPartSelection;

namespace DDI {
  class LogicalPart 
  {
  public:
    LogicalPart(const DDMaterial &,
                const DDSolid &,
		DDEnums::Category  =  DDEnums::unspecified);
    const DDMaterial & material() const;
    const DDSolid & solid() const;
    DDEnums::Category category() const; 
    std::vector<const DDsvalues_type*> specifics() const;
    void specificsV(std::vector<const DDsvalues_type*> & result) const;
    DDsvalues_type mergedSpecifics() const;
    void mergedSpecificsV(DDsvalues_type & res) const;
    void addSpecifics(const std::pair<const DDPartSelection*, const DDsvalues_type*> &);
    void removeSpecifics(const std::pair<const DDPartSelection*, const DDsvalues_type*> &);
    const std::vector<std::pair<const DDPartSelection*,const DDsvalues_type* > > &
         attachedSpecifics() const { return specifics_; }
    bool hasDDValue(const DDValue &) const;
    //const std::vector<DDPartSelection*> & partSelections(const DDValue &) const;	 
    void stream(std::ostream &);
    double & weight();
  private:
    DDMaterial material_;
    DDSolid solid_;
    DDEnums::Category cat_;      
    double weight_;  		

    //std::vector<DDSpecifics> specifics_;
    std::map<DDValue, std::vector<DDPartSelection*> > valToParsel_;
    std::vector<std::pair<const DDPartSelection*, const DDsvalues_type* > > specifics_;
    std::vector<bool> hasDDValue_;
  };
}
#endif
