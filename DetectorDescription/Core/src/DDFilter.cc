#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"

// Message logger.
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
    case DDSpecificsFilter::equals: case DDSpecificsFilter::matches:
      return ( crit.nameVal_.strings() == it->second.strings() );
    case DDSpecificsFilter::not_equals: case DDSpecificsFilter::not_matches:
      return ( crit.nameVal_.strings() != it->second.strings() );
    case DDSpecificsFilter::smaller_equals:
      return ( it->second.strings() <= crit.nameVal_.strings() );
    case DDSpecificsFilter::smaller:
      return ( it->second.strings() < crit.nameVal_.strings() );
    case DDSpecificsFilter::bigger_equals:
      return ( it->second.strings() >= crit.nameVal_.strings() );
    case DDSpecificsFilter::bigger:	 
      return ( it->second.strings() > crit.nameVal_.strings() );
    default:
      return false;
    }
    return false;
  }		 
  
  inline bool sfd_compare(const DDSpecificsFilter::SpecificCriterion & crit,
		   const DDsvalues_type & sv)
  {
    DDsvalues_type::const_iterator it = find(sv,crit.nameVal_.id());
    if (it == sv.end()) return false;
    if (it->second.isEvaluated() ) {
      switch (crit.comp_) {
      case DDSpecificsFilter::equals: case DDSpecificsFilter::matches:
	return (crit.nameVal_.doubles() == it->second.doubles() );
      case DDSpecificsFilter::not_equals: case DDSpecificsFilter::not_matches:
	return (crit.nameVal_.doubles() != it->second.doubles() );
      case DDSpecificsFilter::smaller_equals:
	return ( it->second.doubles() <= crit.nameVal_.doubles() );
      case DDSpecificsFilter::smaller:
	return ( it->second.doubles() < crit.nameVal_.doubles() );
      case DDSpecificsFilter::bigger_equals:
	return ( it->second.doubles() >= crit.nameVal_.doubles() );
      case DDSpecificsFilter::bigger:	 
	return ( it->second.doubles() > crit.nameVal_.doubles() );
      default:
	return false;
      }
    } else {
      edm::LogWarning("DD:Filter") << "Attempt to access a DDValue as doubles but DDValue is NOT Evaluated!" 
				   << crit.nameVal_.name();
    }
    return false;
  }		 
  
  inline bool sfs_compare_nm (const DDSpecificsFilter::SpecificCriterion & crit,
		       const std::vector<const DDsvalues_type *> & specs)
  {
    // The meaning of this is defined for each operator below.
    bool result = false; 
    
    std::vector<const DDsvalues_type *>::const_iterator spit = specs.begin();
    
    // go through all specifics
    //   edm::LogInfo("DDFilter") << "Checking all specifics." << std::endl;
    for (;  spit != specs.end(); ++spit) {
      DDsvalues_type::const_iterator it = (**spit).begin();
      
      // go through all DDValues of the current specific.     
      for (;  it != (**spit).end(); ++it) {
	
	size_t ci = 0; // criteria values index
	size_t si = 0; // specs values index
	// compare all the crit values to all the specs values when the name matches.
	
	while ( !result && ci < crit.nameVal_.strings().size()) {
	  if (it->second.id() == crit.nameVal_.id()) {	
	    //	   edm::LogInfo("DDFilter") << "  Actually checking " << it->second.name() << std::endl;     
	    while ( !result && si < it->second.strings().size()) {
	      //	       edm::LogInfo("DDFilter") << "**** is " << crit.nameVal_.strings()[ci] << " = " << it->second.strings()[si] << " ?" << std::endl;
	      
	      switch (crit.comp_) {
		// equals means at least one value in crit matches one value
		// in specs.
		//
		// not equals means that no values in crit match no values in specs
		//  (see below the ! (not))
		// 
		// Maybe we should have an 'in' or 'contains' operator and equals would meaning
		// would be changed to mean ALL equal.
	      case DDSpecificsFilter::equals: 
	      case DDSpecificsFilter::matches:
	      case DDSpecificsFilter::not_equals: 
	      case DDSpecificsFilter::not_matches: 
		result = ( crit.nameVal_.strings()[ci] == it->second.strings()[si] );
		break;
		
		// less than or equals means that ALL values in specs
		// are less than or equals ALL values in crit.  therefore
		// if even ONE is bigger, then this is false.
	      case DDSpecificsFilter::smaller_equals:
		result = ( it->second.strings()[si] > crit.nameVal_.strings()[ci] );
		break;
		
		// less than means that all are strictly less than, therefore
		// if one is greater than or equal, then this is false
	      case DDSpecificsFilter::smaller:
		result = ( it->second.strings()[si] >= crit.nameVal_.strings()[ci] );
		break;
		
		// greater or equal to means that all values in specs are
		// greater than or equals to all values in crit.  therefore
		// if even ONE is less than, then this is false
	      case DDSpecificsFilter::bigger_equals:
		result = ( it->second.strings()[si] < crit.nameVal_.strings()[ci] );
		break;
		
		// greater means that all values in specs are greater than
		// all values in crit.  therefore if even one is less than or
		// equal to crit, then this is false;
	      case DDSpecificsFilter::bigger:	 
		result = ( it->second.strings() <= crit.nameVal_.strings() );
	      }
	      //edm::LogInfo("DDFilter") << "       si = " << si << "  result = " << result ;
	      si ++;
	    }
	    //edm::LogInfo("DDFilter") << "  ci = " << ci << std::endl;
	    
	    if ( crit.comp_ == DDSpecificsFilter::not_equals 
		 || crit.comp_ == DDSpecificsFilter::not_matches
		 || crit.comp_ == DDSpecificsFilter::smaller_equals 
		 || crit.comp_ == DDSpecificsFilter::smaller
		 || crit.comp_ == DDSpecificsFilter::bigger)
	      result = !result;
	    
	    //edm::LogInfo("DDFilter") << "           it = " << it->second.name() << " final result = " << result << std::endl;
	  }
	  ++ci;
	}
      }
    }
    
    return result;
  }		 
  
  inline bool sfd_compare_nm(const DDSpecificsFilter::SpecificCriterion & crit,
		      const std::vector<const DDsvalues_type *> & specs)
  {
    // The meaning of this is defined for each operator below.
    bool result = false; 
    
    //   edm::LogInfo("DDFilter") << "specs.size()=" << specs.size() << std::endl;
    //   edm::LogInfo("DDFilter") << "crit.nameVal_.doubles().size()=" << crit.nameVal_.doubles().size() << std::endl;
    std::vector<const DDsvalues_type *>::const_iterator spit = specs.begin();
    
    // go through all specifics
    for (;  spit != specs.end(); ++spit) {
      
      DDsvalues_type::const_iterator it = (**spit).begin();
      
      // go through all DDValues of the current specific.     
      for (;  it != (**spit).end(); ++it) {
	if ( !it->second.isEvaluated() ) {
	  edm::LogWarning("DD:Filter") << "(nm) Attempt to access a DDValue as doubles but DDValue is NOT Evaluated!" 
				       << crit.nameVal_.name();
	  continue; // go on to next one, do not attempt to acess doubles()
	}
	size_t ci = 0; // criteria values index
	size_t si = 0; // specs values index
	// compare all the crit values to all the specs values when the name matches.
	
	while ( !result && ci < crit.nameVal_.doubles().size()) {
	  
	  if (it->second.id() == crit.nameVal_.id()) {	     
	    //	   edm::LogInfo("DDFilter") << "  Actually checking " << it->second.name() << std::endl;     
	    while ( !result && si < it->second.doubles().size()) {
	      //	     edm::LogInfo("DDFilter") << "**** is " << crit.nameVal_.doubles()[ci] << " = " << it->second.doubles()[ci] << " ?" << std::endl;
	      switch (crit.comp_) {
		
		// equals means at least one value in crit matches one value
		// in specs.
		//
		// not equals means that no values in crit match no values in specs
		//  (see below the ! (not))
	      case DDSpecificsFilter::equals: 
	      case DDSpecificsFilter::matches:
	      case DDSpecificsFilter::not_equals: 
	      case DDSpecificsFilter::not_matches: 
		result = ( crit.nameVal_.doubles()[ci] == it->second.doubles()[si] );
		break;
		
		// less than or equals means that ALL values in specs
		// are less than or equals ALL values in crit.  therefore
		// if even ONE is bigger, then this is false.
	      case DDSpecificsFilter::smaller_equals:
		result = ( it->second.doubles()[si] > crit.nameVal_.doubles()[ci] );
		break;
		
		// less than means that all are strictly less than, therefore
		// if one is greater than or equal, then this is false
	      case DDSpecificsFilter::smaller:
		result = ( it->second.doubles()[si] >= crit.nameVal_.doubles()[ci] );
		break;
		
		// greater or equal to means that all values in specs are
		// greater than or equals to all values in crit.  therefore
		// if even ONE is less than, then this is false
	      case DDSpecificsFilter::bigger_equals:
		result = ( it->second.doubles()[si] < crit.nameVal_.doubles()[ci] );
		break;
		
		// greater means that all values in specs are greater than
		// all values in crit.  therefore if even one is less than or
		// equal to crit, then this is false;
	      case DDSpecificsFilter::bigger:	 
		result = ( it->second.doubles() <= crit.nameVal_.doubles() );
	      }
	      //	     edm::LogInfo("DDFilter") << "       si = " << si << "  result = " << result ;
	      si ++;
	      
	    }
	    //	   edm::LogInfo("DDFilter") << "  ci = " << ci << std::endl;
	    
	    if ( crit.comp_ == DDSpecificsFilter::not_equals 
		 || crit.comp_ == DDSpecificsFilter::not_matches
		 || crit.comp_ == DDSpecificsFilter::smaller_equals 
		 || crit.comp_ == DDSpecificsFilter::smaller
		 || crit.comp_ == DDSpecificsFilter::bigger)
	      result = !result;
	    
	    //	   edm::LogInfo("DDFilter") << "           it = " << it->second.name() << " final result = " << result << std::endl;
	    
	  }
	  
	  ++ci;
	}
      }
    }
    
    return result;
  }		 
  
  

}

// ================================================================================================

DDSpecificsFilter::DDSpecificsFilter() 
  : DDFilter()
{ }

DDSpecificsFilter::~DDSpecificsFilter() {}


void DDSpecificsFilter::setCriteria(const DDValue & nameVal, // name & value of a variable 
                   comp_op op, 
		   log_op l, 
		   bool asStrin, // compare std::strings otherwise doubles
		   bool merged // use merged-specifics or simple-specifics
		   )
{
  criteria_.push_back(SpecificCriterion(nameVal,op,asStrin,merged));
  logOps_.push_back(l);
 }		   

bool DDSpecificsFilter::accept(const DDExpandedView & node) const
{
  return accept_impl(node);
} 


bool DDSpecificsFilter::accept_impl(const DDExpandedView & node) const
{
  bool result = true;
  criteria_type::const_iterator it = criteria_.begin();
  criteria_type::const_iterator crEnd = criteria_.end();
  logops_type::const_iterator logOpIt = logOps_.begin();
  //logops_type::iterator logOpEnd = logOps_.end();
  const DDLogicalPart & logp = node.logicalPart();
  DDsvalues_type  sv;
  std::vector<const DDsvalues_type *> specs;
  for (; it != crEnd; ++it, ++logOpIt) {
    // avoid useless evaluations
    if ( (   result &&(*logOpIt)==OR ) ||
	 ( (!result)&&(*logOpIt)==AND) ) continue; 

    bool locres=false;
    if (logp.hasDDValue(it->nameVal_)) { 
      
      if (it->merged_) {
	
	if (sv.empty())  node.mergedSpecificsV(sv);
	
	if (it->asString_) { // merged specifics & compare std::strings
	  locres = sfs_compare(*it,sv); 
	}
	else { // merged specifics & compare doubles
	  locres = sfd_compare(*it,sv);
	}
      }
      else {
	
	if (specs.empty()) node.specificsV(specs);
	
	if (it->asString_) { // non-merged specifics & compare std::strings
	  locres = sfs_compare_nm(*it, specs);
	}
	else { // non-merged specifics & compare doubles
	  locres = sfd_compare_nm(*it, specs);
	}  
      }
    }
    if (*logOpIt==AND) {
      result &= locres;
    }
    else {
      result |= locres;
    }
  }
  return result;
}
