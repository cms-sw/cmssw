
#include "DetectorDescription/ExprAlgo/interface/AlgoPos.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoImpl.h"
#include "DetectorDescription/ExprAlgo/interface/AlgoCheck.h"
#include "DetectorDescription/Base/interface/DDException.h"

using namespace std;

AlgoPos::AlgoPos(AlgoCheck * check)
 : start_(0), end_(0), incr_(0), curr_(0), count_(0),
   terminate_(false), selAlgo_(0), checkAlgo_(check)
{ }


AlgoPos::~AlgoPos()
{ 
  //FIXME: delete all registered AlgoImpls!!!
  std::vector<AlgoImpl*>::iterator it = regAlgos_.begin();
  for (; it != regAlgos_.end(); ++it) {
    delete *it;
  }  
  delete checkAlgo_; 
}


void AlgoPos::setParameters(int start, int end, int incr,
                            const parS_type & ps, const parE_type & pe)
{
  // init the algorithm when parameters are set
  terminate_ = false;
  start_     = start;
  end_       = end;
  incr_      = incr;
  if (incr>0) curr_ = start;
  if (incr<0) curr_ = end;  
  count_ = 1;

  // now check mandatory parameters, then let the algorithm check all parametes itself.
  // collect all error messages in std::string err_
  
  bool check = true;
  if (incr) {
    if (start>=end) {
      check = false;
    }  
  }
  if (!check) err_ = "\twrong range specification: (start<end && incr!=0)==false, [start,end,incr]=["
                   + AlgoImpl::d2s(start) + "," 
                   + AlgoImpl::d2s(end) + "," 
		   + AlgoImpl::d2s(incr) + "]\n" ;

  if (incr==0) {
    if ( (start!=0) || (end!=0) ) {
      err_ += "\tincr==0 requires start==0 and end==0 in the range. range: [start,end,incr]=["
                   + AlgoImpl::d2s(start) + "," 
                   + AlgoImpl::d2s(end) + "," 
		   + AlgoImpl::d2s(incr) + "]\n" ;
      check = false;		   

    }
  }	
  	   		   
  // select one of the registered algorithm implementations
  ParS_ = ps;
  ParE_ = pe;
  if (!select()) check = false;
  
  if (!check) 
    throw DDException(err_);
}


void AlgoPos::registerAlgo(AlgoImpl*a)
{
  regAlgos_.push_back(a);
}

size_t AlgoPos::numRegistered()
{
  return regAlgos_.size();
}

int AlgoPos::copyno() const
{
  return selAlgo_->copyno();
}


/**
  In the case of incr_!=0 this function will set curr_ to the next point in
  the range [start_,end_,incr_] or terminate the algorithm if the next point
  is going to be out of the range bounds.
  
  In the case of incr_=0 this function calls checkTermination() in which
  the algorithm itself must check whether to terminate or not
*/  
void AlgoPos::next()
{
  // increase the invocation count of the algorithm
  ++count_;
  
  
  // iterate to the next position in the range [start_,end_,incr_]
  // only if incr_ != 0
  
  if (incr_>0) {
    curr_ += incr_;
    if (curr_>end_) {
      terminate();
    }  
  }
  
  if (incr_<0) {
    curr_ += incr_;
    if (curr_<start_) {
      terminate();
    }  
  }
  
  // incr_==0: the algorithm has to self-check whether to terminate
  if (incr_==0) {
    checkTermination();
  }
}



void AlgoPos::checkTermination()
{ 
  selAlgo_->checkTermination();
}


void AlgoPos::terminate()
{ 
  terminate_=true; 
}


bool AlgoPos::go() const 
{ 
  return !terminate_; 
}


DD3Vector AlgoPos::translation()
{
  return selAlgo_->translation();
}


DDRotationMatrix AlgoPos::rotation()
{
  return selAlgo_->rotation();
}


bool AlgoPos::select()
{
  bool result = false;
  
  // if constraints checking is enabled (object checkAlgo_ is there) ,
  // check the contraints of the parameters as specified in the schema.
  if (checkAlgo_) {
    result = (checkAlgo_->check(ParS_,ParE_,err_));
  }
  else {
    result = true;
  }  
  
  
  if (result) { // select an algorithm-implementation only if the parameters are ok
    std::vector<AlgoImpl*>::iterator it = regAlgos_.begin();
    for (; it != regAlgos_.end(); ++it) {
      std::string::size_type s = err_.size();
      result = (*it)->checkParameters();
      if (s!=err_.size()) { // uups, error-std::string was modified! tell it, where:
        err_ += std::string("\tin algo.implementation labeled \"") + (*it)->label_ + std::string("\"\n");
      }
      if (result) { // select the algorithm
        selAlgo_ = *it;
        break;
      }  
    }
  }  
  // parameters are not ok, put something into the error message
  /*
  if (!result) { 
   err_ += "\tin algorithm named " + std::string("[") 
        +  ns() + std::string(":") + name() + std::string("]\n");
  }	
  */   
  
  return result;
}

// stream information about the alog-parameters
void AlgoPos::stream(ostream & os) const
{
  os <<  "start=" << start_ << " end=" << end_ << " incr=" << incr_ << std::endl;
  parE_type::const_iterator eit = ParE_.begin();
  for (; eit != ParE_.end() ; ++eit) {
    std::vector<double>::const_iterator sit = eit->second.begin();
    os << "parE name=" << eit->first;
    for (; sit != eit->second.end(); ++sit) {
       os << " val=" << *sit << std::endl;
    }
  }
  parS_type::const_iterator stit = ParS_.begin();
  for (; stit != ParS_.end() ; ++stit) {
    std::vector<std::string>::const_iterator sit = stit->second.begin();
    os << "parS name=" << stit->first;
    for (; sit != stit->second.end(); ++sit) {
       os << " val=" << *sit << std::endl;
    }
  }
}
