#include "DetectorDescription/Core/interface/DDFilter.h"

#include <stddef.h>
#include <iterator>
#include <string>
#include <utility>

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
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

namespace {
  
// SpecificsFilterString-comparison (sfs)
  inline bool sfs_compare(const DDSpecificsFilter::SpecificCriterion & crit,
			  const DDsvalues_type & sv) {
    DDsvalues_type::const_iterator it = find(sv,crit.nameVal_.id());
    if (it == sv.end()) return false;
    switch (crit.comp_) {
    case DDCompOp::equals: case DDCompOp::matches:
      return ( crit.nameVal_.strings() == it->second.strings() );
    case DDCompOp::not_equals: case DDCompOp::not_matches:
      return ( crit.nameVal_.strings() != it->second.strings() );
    default:
      return false;
    }
    return false;
  }		   
}

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
  DDsvalues_type  sv;
  std::vector<const DDsvalues_type *> specs;
  for( auto it = begin(criteria_); it != end(criteria_); ++it) {

    bool locres=false;
    if (logp.hasDDValue(it->nameVal_)) { 
      
      if (sv.empty())  node.mergedSpecificsV(sv);
      
      locres = sfs_compare(*it,sv); 

    }
    result &= locres;
    // avoid useless evaluations
    if(!result) {
      break;
    }
  }
  return result;
}
