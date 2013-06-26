
#include "DetectorDescription/Core/interface/DDQuery.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
//#include "DetectorDescription/Base/interface/DDdebug.h"

DDQuery::DDQuery(const DDCompactView & cpv)
 :   epv_(cpv), scope_(0)
{ }
 

DDQuery::~DDQuery() 
{ }


/** the standard DDQuery only support one single filter. 
 Memory management of DDFilter* is NOT delegated to DDQuery 
*/
void DDQuery::addFilter(const DDFilter & f, log_op op)
{
  // cheating a bit with const ....
  // Filters have a non-const ::accept(..) member function to allow
  // a possible internal state change in a particular filter implementation ...
  DDFilter * nonConstFilter = const_cast<DDFilter *>(&f);
  criteria_.push_back(std::make_pair(false,nonConstFilter)); 
  logOps_.push_back(op);
  //DCOUT('F',"DDQuery::addFilter(): log-op=" << op );
}


void DDQuery::setScope(const DDScope & s) 
{
  scope_ = &s;
}


const std::vector<DDExpandedNode> & DDQuery::exec()
{
   result_.clear();
   epv_.reset();

   //bool filtered = bool(filters_.size());
   //bool unfiltered = !filtered;
   
   // currently at least one filter must be set, because
   // the query simply consists in applying at least one filter!
   if(criteria_.size()) { // <- remove the 'if' when implementing 'the QUERY'
     int depth = 0;
     bool scoped = false;
     DDScope::scope_type dummy;
     DDScope::scope_type::const_iterator it = dummy.begin();
     DDScope::scope_type::const_iterator it_end = dummy.end();
     if (scope_) {
       const DDScope & sc = *scope_;
       it = sc.scope().begin();
       depth = sc.depth();
       scoped = bool(sc.scope().end()-it);  
     }
     
     bool runonce = scoped ? false : true;
            
     while (runonce || (it != it_end) ) {
        if (scoped) epv_.setScope(*it,depth); // set the subtree-scope & depth within
        bool run = true;
	while(run) {
	  //DCOUT('F', "DDQuery: examining " << epv_.geoHistory().back() );
	  
	  std::vector<const DDsvalues_type *> specs = epv_.specifics();
// 	  std::vector<const DDsvalues_type *>::const_iterator sit = specs.begin();
	  //FIXME: find a solution when more then one specifics_type is attached to
	  //FIXME: particlular nodes ... (merging the specifics-map, etc ...)
	  //FIXME: for (; sit != specs.end() ; ++sit) {
// 	    DDsvalues_type dummy;
// 	    const DDsvalues_type * specifics;
// 	    if (sit==specs.end())
// 	      specifics = &dummy;
// 	    else
// 	      specifics = *sit;
	      
	    criteria_type::iterator it = criteria_.begin();
	    logops_type::iterator logOpIt = logOps_.begin();
            // loop over all user-supplied criteria (==filters)
            bool result=true;
            for (; it != criteria_.end(); ++it, ++logOpIt) {
              DDFilter * filter = it->second;
 	      if (filter->accept(epv_)) {
	                              //.geoHistory().back(), // expanded node
	 		 	      //epv_.geoHistory(),
				      //specifics)) { // geom.history
	       
	        it->first=true;
	        //DCOUT('F', " Filter(" << criteria_.end()-it << ") accepted: " << epv_.geoHistory().back());
	      }
	      else {
	        it->first=false;
	      }
	      	
	      // now do the logical-operations on the results encountered so far:
              if (*logOpIt==AND) { // AND
                result &= it->first; 
              }
              else { // OR
                result |= it->first;  
              }
	    } // <-- loop over filters 
	    //DCOUT('f', "-------------------");
	  if (result) {
	    // HERE THE ACTUAL QUERY SHOULD BE INVOKED!!!
	    result_.push_back(epv_.geoHistory().back());  
	  }  
	  //} <-- loop over std::vector of specifics_type ...
	  run = epv_.next();
	}
        if (scoped)
	  ++it;
	else
	  runonce=false;  	
     } 
   }  
   return result_;
}
