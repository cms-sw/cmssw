#include "DetectorDescription/Core/interface/DDFilter.h"

#include <stddef.h>
#include <iterator>
#include <string>
#include <utility>

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "DetectorDescription/Core/interface/DDComparator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDFilter::DDFilter() 
{ }

DDFilter::~DDFilter()
{ }

// =======================================================================
// =======================================================================

/** Mike Case 2007x12x18
 * Important note:  The removal of throwing and catchine a DDException is
 * based more on "what is" rather than the ideal.  We have managed to force
 * users to put all SpecPar types and filters at the end of the set of XML
 * files.  Therefore a DDValue of the XML syntax "[myvar]-2*[myothervar]"
 * will have a correctly evaluated number and valid doubles by the time this
 * code is ever run.  What will happen now, is possibly a crash if someone mis-
 * identifies their variables to the DD.  I use edm::LogWarning to attempt to catch
 * one type of such problems.
 **/

// ================================================================================================
DDSpecificsFilter::DDSpecificsFilter() 
  : DDFilter()
{ }

DDSpecificsFilter::~DDSpecificsFilter() {}


void DDSpecificsFilter::setCriteria(const DDValue & nameVal, // name & value of a variable 
                                    DDCompOp op)
{
  criteria_.push_back(SpecificCriterion(nameVal,op));
 }		   

bool DDSpecificsFilter::accept(const DDExpandedView & node) const
{
  return accept_impl(node);
} 

bool DDSpecificsFilter::accept_impl(const DDExpandedView & node) const
{
  bool result = true;
  const DDLogicalPart & logp = node.logicalPart();

  for( auto it = begin(criteria_); it != end(criteria_); ++it) {
    bool locres=false;
    if (logp.hasDDValue(it->nameVal_)) { 
      
      const auto& specs = logp.attachedSpecifics();

      const auto& hist = node.geoHistory();
      bool decided = false;
      for(auto const& spec: specs) {
        if(DDCompareEqual(hist,*spec.first)()) {
          for(auto const& v: *(spec.second) ) {
            if(it->nameVal_.id() == v.first) {
              switch (it->comp_) {
              case DDCompOp::equals:
                {
                  locres= (it->nameVal_.strings() == v.second.strings());
                  break;
                }
              case DDCompOp::not_equals:
                {
                  locres= ( it->nameVal_.strings() != v.second.strings() );
                  break;
                }
              default:
                return false;
              }
              decided = true;
              break;
            }
          }
          if(decided) {
            break;
          }
        }
      }
    }
    result &= locres;
    // avoid useless evaluations
    if(!result) {
      break;
    }
  }
  return result;
}

bool
DDSpecificsAnyValueFilter::accept(const DDExpandedView& node) const {
  const DDLogicalPart & logp = node.logicalPart();

  if (logp.hasDDValue(attribute_)) { 
      
    const auto& specs = logp.attachedSpecifics();
    
    const auto& hist = node.geoHistory();
    for(auto const& spec: specs) {
      if(DDCompareEqual(hist,*spec.first)()) {
        for(auto const& v: *(spec.second) ) {
          if(attribute_.id() == v.first) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool
DDSpecificsMatchesValueFilter::accept(const DDExpandedView& node) const {
  const DDLogicalPart & logp = node.logicalPart();

  if (logp.hasDDValue(value_)) { 
    
    const auto& specs = logp.attachedSpecifics();
    
    const auto& hist = node.geoHistory();
    for(auto const& spec: specs) {
      if(DDCompareEqual(hist,*spec.first)()) {
        for(auto const& v: *(spec.second) ) {
          if(value_.id() == v.first) {
            return (value_.strings() == v.second.strings());
          }
        }
      }
    }
  }
  return false;

}
