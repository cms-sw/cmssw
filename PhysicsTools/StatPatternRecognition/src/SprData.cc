//$Id: SprData.cc,v 1.2 2007/09/21 22:32:09 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <cassert>
#include <utility>
#include <functional>
#include <algorithm>

using namespace std;


SprData::~SprData()
{
  this->clear();
}


void SprData::clear()
{
  if( ownPoints_ ) {
    for( unsigned int i=0;i<data_.size();i++ )
      delete data_[i];
  }
  data_.clear();
}


SprPoint* SprData::insert(unsigned index, int cls, 
			  const std::vector<double>& v)
{
  // check dimensionality
  assert( !v.empty() );
  if( data_.empty() && dim_==0 ) 
    dim_ = v.size();
  else {
    if( dim_ != v.size() ) {
      cerr << "Dimensionality of input vector does not match dimensionality " 
	   << "of space: " << v.size() << " " << dim_ << endl;
      return 0;
    }
  }
   
  // insert 
  SprPoint* p = new SprPoint(index,cls,v);
  data_.push_back(p);

  // exit
  return p;
}

SprPoint* SprData::insert(SprPoint* p)
{
   // check dimensionality
  assert( !p->empty() );
  if( data_.empty() && dim_==0 ) 
    dim_ = p->dim();
  else {
    if( dim_ != p->dim() ) {
      cerr << "Dimensionality of input vector does not match dimensionality " 
	   << "of space: " << p->dim() << " " << dim_ << endl;
      return 0;
    }
  }
   
  // insert 
  data_.push_back(p);

  // exit
  return p;
}


SprPoint* SprData::insert(int cls, const std::vector<double>& v)
{
  return this->insert(data_.size(),cls,v);
}


bool SprData::setVars(const std::vector<std::string>& vars)
{
  assert( !vars.empty() );
  if( dim_ == 0 ) 
    dim_ = vars.size();
  else {
    if( dim_ != vars.size() ) {
      cerr << "Number of variables does not match dimensionality " 
	   << "of space: " << vars.size() << " " << dim_ << endl;
      return false;
    }
  }
  vars_ = vars;
  return true;
}


unsigned SprData::ptsInClass(const SprClass& cls) const
{
  return count_if(data_.begin(),data_.end(),
		  bind2nd(mem_fun(&SprPoint::class_eq),cls));
}


SprPoint* SprData::find(unsigned index) const
{
  vector<SprPoint*>::const_iterator iter 
    = find_if(data_.begin(),data_.end(),
	      bind2nd(mem_fun(&SprPoint::index_eq),index));
  if( iter == data_.end() )
    return 0;
  return *iter;
}


int SprData::dimIndex(const char* var) const
{
  vector<string>::const_iterator iter = ::find(vars_.begin(),vars_.end(),var);
  if( iter == vars_.end() )
    return -1;
  return (iter-vars_.begin());
}


SprData* SprData::emptyCopy() const
{
  SprData* copy = new SprData(false);// do not own points
  copy->setDim(dim_);
  copy->setVars(vars_);
  copy->setLabel(label_.c_str());
  return copy;
}


SprData* SprData::copy() const
{
  SprData* copy = this->emptyCopy();
  copy->data_ = data_;
  return copy;
}

