#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

DDFilteredView::DDFilteredView(const DDCompactView & cpv)
 : epv_(cpv)
{
   parents_.push_back(epv_.geoHistory());
}

DDFilteredView::~DDFilteredView()
{ }

const DDLogicalPart & DDFilteredView::logicalPart() const
{
  return epv_.logicalPart();
}

void DDFilteredView::addFilter(const DDFilter & f, DDLogOp op)
{
  criteria_.push_back(&f); 
  logOps_.push_back(op);
}
 
const DDTranslation & DDFilteredView::translation() const
{
   return epv_.translation();
}
	 
const DDRotationMatrix & DDFilteredView::rotation() const		           
{
   return epv_.rotation();
}
   
const DDGeoHistory &  DDFilteredView::geoHistory() const
{
   return epv_.geoHistory();
}

std::vector<const DDsvalues_type * > DDFilteredView::specifics() const
{
  std::vector<const DDsvalues_type * > result; epv_.specificsV(result);
  return result;
}

void  DDFilteredView::specificsV(std::vector<const DDsvalues_type * > & result) const 
{
  epv_.specificsV(result);
}

void  DDFilteredView::mergedSpecificsV(DDsvalues_type & merged) const 
{
  epv_.mergedSpecificsV(merged);
}

DDsvalues_type DDFilteredView::mergedSpecifics() const
{
  DDsvalues_type merged;
  epv_.mergedSpecificsV(merged);
  return merged;
}

int DDFilteredView::copyno() const
{
  return epv_.copyno();
}

const DDGeoHistory & DDFilteredView::scope() const
{
  return epv_.scope();
}

bool DDFilteredView::setScope(const DDGeoHistory & hist)
{
  bool result = epv_.setScope(hist,0);
  if (result) {
    parents_.clear();
    parents_.push_back(hist);
  }  
  return result;
}  

void DDFilteredView::clearScope()
{
   epv_.clearScope();
   parents_.clear();
   parents_.push_back(epv_.geoHistory());
}

bool DDFilteredView::next()
{
   bool result = false;
   int i=0;
   while(epv_.next()) {
     ++i;
     if ( filter() ) {
       result = true;
       break;
     }
   }
   return result;
}

/**
 Algorithm:
  
  - find the first child matching the filter down the current subtree using next()
    which transverses already in the required hierarchical sequence ... 
*/
bool DDFilteredView::firstChild() 
{
   bool result = false;
   
   // save the current position
   DDGeoHistory savedPos = epv_.geoHistory();
           
   // save the current scope
   DDGeoHistory savedScope = epv_.scope_;
   int savedDepth = epv_.depth_;
   
   // set current node to be the scope
   epv_.scope_ = epv_.geoHistory();
   epv_.depth_ = 0;
   
   // search the subtree for the first matching node
   if (next()) { 
     result = true;
     epv_.scope_ = savedScope;
     epv_.depth_ = savedDepth;
   }
   else { // restore the starting point
     epv_.scope_ = savedScope;
     epv_.depth_ = savedDepth;
     epv_.goToHistory(savedPos);
   }
   
   if (result) {
     parents_.push_back(epv_.geoHistory());
   }
     
   return result;
}

/**
  Algorithm:
  
  - find the first node - which matches the filter - in the subtrees of the siblings of
    the current node
*/  
bool DDFilteredView::nextSibling()
{
  //PRE:  the current node A is one matching the filter or the (scoped) root
  //POST: current node is A if no filter has matched, or B with matching filter
  //      B is the firstChild matching the filter in the subtrees of A's siblings
  bool result = false;
  DDGeoHistory savedPos = epv_.geoHistory();
  
  bool flag = true;
  //bool shuffleParent = false;
  while (flag) {
    if (epv_.nextSibling()) {
      if ( filter() ) {
        result = true;
        break;
      }
      else if (firstChild()) {
        result = true;
	// firstChild increases parents!
	parents_.pop_back(); 
	break;
      }
    } // <-- epv_.nextSibling      
    else if (!epv_.parent()) {
      flag = false;
    }
    if (epv_.geoHistory().size() == parents_[parents_.size()-2].size()) {
      flag = false;
    }  
  }   
      
  if (!result)
    epv_.goToHistory(savedPos);
  else
    parents_.back() = epv_.geoHistory();
       
  return result;
}

bool DDFilteredView::parent()
{
   bool result = false;
 
   if (parents_.size()==1) {
     result = false;
   }
   else {
     parents_.pop_back();
     epv_.goToHistory(parents_.back());
     result = true;
   }  
   
   // =====> CHECK WHETHER THE FOLLOWING REMARKS STILL REPRESENT THE IMPLEMENTATION ABOVE <===   
   // case 1: we've reached the (scoped) root of the underlying expanded-view
   //         from the first child of the fitered-view; result must be true
   // case 2: we've reached a parent-node where the filter said 'yes'; 
   //         result must be true
   // case 3: we are already at the (scoped) root of the underlying expanded-view
   //         result must be false
      
   return result;
}

void DDFilteredView::reset()
{
  epv_.reset();
  parents_.clear();
  parents_.push_back(epv_.geoHistory());          
}

bool DDFilteredView::filter()
{
  bool result = true;
  auto logOpIt = logOps_.begin();
  // loop over all user-supplied criteria (==filters)
  for( auto it = begin(criteria_); it != end(criteria_); ++it, ++logOpIt) {
    // avoid useless evaluations
    if(( result && ( *logOpIt ) == DDLogOp::OR ) ||
       (( !result ) && ( *logOpIt ) == DDLogOp::AND )) continue; 
    
    bool locres = (*it)->accept(epv_);
    
    // now do the logical-operations on the results encountered so far:
    if (*logOpIt == DDLogOp::AND) { // AND
      result &= locres; 
    }
    else { // OR
      result |= locres;  
    }
  } // <-- loop over filters     
  return result;
}

DDFilteredView::nav_type DDFilteredView::navPos() const
{
  return epv_.navPos();
}

DDFilteredView::nav_type DDFilteredView::copyNumbers() const
{
  return epv_.copyNumbers();
}

bool DDFilteredView::goTo(const DDFilteredView::nav_type & /*n*/)
{
 // WARNING!!!!!!!!!!
 // NOT IMPLEMENTED!!!!!!!
 bool result(false);
 return result;
}

void DDFilteredView::print() {
  edm::LogInfo("DDFliteredView") << "FilteredView Status" << std::endl
       << "-------------------" << std::endl
       << "scope = " << epv_.scope_ << std::endl
       << "parents:" << std::endl;
  for (unsigned int i=0; i<parents_.size(); ++i)
    edm::LogInfo("DDFliteredView") << "  " << parents_[i] << std::endl;
    
}

const std::vector<DDGeoHistory> & DDFilteredView::history() const
{
  return parents_;
}
