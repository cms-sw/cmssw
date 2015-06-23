#include "DetectorDescription/Core/interface/DDQuery.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"

DDQuery::DDQuery(const DDCompactView & cpv)
 :   epv_(cpv), scope_(0)
{ }
 

DDQuery::~DDQuery() 
{ }


/** the standard DDQuery only support one single filter. 
 Memory management of DDFilter* is NOT delegated to DDQuery 
*/
void DDQuery::addFilter(const DDFilter & f, DDLogOp op)
{
  // cheating a bit with const ....
  // Filters have a non-const ::accept(..) member function to allow
  // a possible internal state change in a particular filter implementation ...
  DDFilter * nonConstFilter = const_cast<DDFilter *>(&f);
  criteria_.push_back(std::make_pair(false,nonConstFilter)); 
  logOps_.push_back(op);
}


void DDQuery::setScope(const DDScope & s) 
{
  scope_ = &s;
}


const std::vector<DDExpandedNode> & DDQuery::exec()
{
   result_.clear();
   epv_.reset();
   
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
	  std::vector<const DDsvalues_type *> specs = epv_.specifics();
	    auto logOpIt = logOps_.begin();
            // loop over all user-supplied criteria (==filters)
            bool result=true;
            for( auto it = begin(criteria_); it != end(criteria_); ++it, ++logOpIt) {
              DDFilter * filter = it->second;
 	      if (filter->accept(epv_)) {
	        it->first=true;
	      }
	      else {
	        it->first=false;
	      }
	      	
	      // now do the logical-operations on the results encountered so far:
              if (*logOpIt == DDLogOp::AND) { // AND
                result &= it->first; 
              }
              else { // OR
                result |= it->first;  
              }
	    } // <-- loop over filters 
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
