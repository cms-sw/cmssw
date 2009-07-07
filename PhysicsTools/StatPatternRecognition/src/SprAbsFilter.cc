//$Id: SprAbsFilter.cc,v 1.4 2007/12/01 01:29:46 narsky Exp $

#include "PhysicsTools/StatPatternRecognition/interface/SprExperiment.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprAbsFilter.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprPoint.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprUtils.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprStringParser.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprIntegerPermutator.hh"

#include <stdio.h>
#include <fstream>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>
#include <functional>
#include <utility>
#include <cmath>
#include <numeric>
#include <cassert>

using namespace std;


struct SAFCmpPairFirst
  : public binary_function<pair<double,double>,pair<double,double>,bool> {
  bool operator()(const pair<double,double>& l, const pair<double,double>& r)
    const {
    return (l.first < r.first);
  }
};


struct SAFCmpPairDIFirst 
  : public binary_function<pair<double,int>,pair<double,int>,bool> {
  bool operator()(const pair<double,int>& l, const pair<double,int>& r)
    const {
    return (l.first < r.first);
  }
};


struct SAFCmpPairDIFirstNumber
  : public binary_function<pair<double,int>,double,bool> {
  bool operator()(const pair<double,int>& l, double r) const {
    return (l.first < r);
  }
};


SprAbsFilter::SprAbsFilter(const SprData* data,
			   bool ownData) 
  : 
  data_(data), 
  copy_(data), 
  ownData_(ownData),
  ownCopy_(false),
  dataWeights_(),
  copyWeights_(),
  classes_(),
  imin_(0),
  imax_(0),
  median_()
{
  assert( data_ != 0 );
  this->setUniformWeights();
  dataWeights_ = copyWeights_;
}


SprAbsFilter::SprAbsFilter(const SprData* data,
			   const std::vector<SprClass>& classes,
			   bool ownData) 
  : 
  data_(data), 
  copy_(data), 
  ownData_(ownData),
  ownCopy_(false),
  dataWeights_(),
  copyWeights_(),
  classes_(classes),
  imin_(0),
  imax_(0),
  median_()
{
  assert( data_ != 0 );
  this->setUniformWeights();
  dataWeights_ = copyWeights_;
}


SprAbsFilter::SprAbsFilter(const SprData* data, 
			   const std::vector<double>& weights,
			   bool ownData) 
  : 
  data_(data), 
  copy_(data), 
  ownData_(ownData),
  ownCopy_(false),
  dataWeights_(),
  copyWeights_(),
  classes_(),
  imin_(0),
  imax_(0),
  median_()
{
  assert( data_ != 0 );
  bool status = this->setWeights(weights);
  assert( status );
  dataWeights_ = copyWeights_;
}


SprAbsFilter::SprAbsFilter(const SprData* data, 
			   const std::vector<SprClass>& classes,
			   const std::vector<double>& weights,
			   bool ownData) 
  : 
  data_(data), 
  copy_(data), 
  ownData_(ownData),
  ownCopy_(false),
  dataWeights_(),
  copyWeights_(),
  classes_(classes),
  imin_(0),
  imax_(0),
  median_()
{
  assert( data_ != 0 );
  bool status = this->setWeights(weights);
  assert( status );
  dataWeights_ = copyWeights_;
}


SprAbsFilter::SprAbsFilter(const SprAbsFilter& filter) 
  : 
  data_(0), 
  copy_(0), 
  ownData_(true),
  ownCopy_(false),
  dataWeights_(filter.copyWeights_),
  copyWeights_(filter.copyWeights_),
  classes_(filter.classes_),
  imin_(0),
  imax_(0),
  median_()
{
  data_ = filter.copy();
  copy_ = data_;
}


void SprAbsFilter::clear() 
{
  if( ownCopy_ ) {
    delete copy_;
    ownCopy_ = false;
  }
  copy_ = data_;
  this->resetIndexRange();
  this->reset();
  copyWeights_ = dataWeights_;
}


void SprAbsFilter::classes(std::vector<SprClass>& classes) const 
{
  if( classes_.empty() )
    this->allClasses(classes);
  else
    classes = classes_;
}


bool SprAbsFilter::filter()
{
  // make a copy
  SprData* copy = data_->emptyCopy();

  // find index range
  unsigned int istart = ( imin_<0 ? 0 : imin_ );
  unsigned int iend = ( (imax_>data_->size()||imax_==0) ? data_->size() : imax_ );

  // loop through points and accept
  copyWeights_.clear();
  for( unsigned int i=istart;i<iend;i++ ) {
    SprPoint* p = (*data_)[i];
    if( this->category(p) && this->pass(p) ) {
      copy->uncheckedInsert(p);
      copyWeights_.push_back(dataWeights_[i]);
    }
  }

  // save copy
  if( ownCopy_ ) delete copy_;
  copy_ = copy;
  ownCopy_ = true;

  // exit
  return true;
}


bool SprAbsFilter::irreversibleFilter()
{
  if( !this->filter() ) return false;
  if( ownData_ ) {
    if( data_->ownPoints() ) {
      cerr << "Cannot run an irreversible filter on data that owns points."
	   << endl;
      return false;
    }
    delete data_;
  }
  data_ = copy_;
  ownCopy_ = false;
  ownData_ = true;
  dataWeights_ = copyWeights_;
  imin_ = 0;
  imax_ = data_->size();
  return true;
}


bool SprAbsFilter::setW(int i, double w)
{
  if( i<0 || i>=(int)copyWeights_.size() ) return false;
  copyWeights_[i] = w;
  return true;
}


bool SprAbsFilter::category(const SprPoint* p) const
{
  if( classes_.empty() ) return true;
  if( ::find(classes_.begin(),classes_.end(),p->class_) == classes_.end() )
    return false;
  return true;
}


bool SprAbsFilter::normalizeWeights(const std::vector<SprClass>& classes,
				    double totalWeight)
{
  assert( copy_ != 0 );

  // save classes
  vector<SprClass> saveClasses = classes_;
  classes_ = classes;

  // get size
  unsigned int size = copy_->size();
  if( size == 0 ) return true;
  assert( size == copyWeights_.size() );

  // sum weights in the chosen classes
  double sum = 0;
  for( unsigned int i=0;i<size;i++ ) {
    if( this->category((*copy_)[i]) )
      sum += copyWeights_[i];
  }
  if( sum < SprUtils::eps() ) {
    classes_ = saveClasses;
    return false;
  }

  // normalize
  double factor = totalWeight/sum;
  for( unsigned int i=0;i<size;i++ ) {
    if( this->category((*copy_)[i]) )
      copyWeights_[i] *= factor;
  }

  // exit
  classes_ = saveClasses;
  return true;
}


bool SprAbsFilter::setWeights(const std::vector<double>& weights)
{
  if( this->size() != weights.size() )
    return false;
  copyWeights_ = weights;
  return true;
}


bool SprAbsFilter::setPermanentWeights(const std::vector<double>& weights)
{
  if( data_->size() != weights.size() )
    return false;
  dataWeights_ = weights;
  return this->filter();
}


void SprAbsFilter::setUniformWeights()
{
  if( !this->empty() ) {
    copyWeights_.clear();
    copyWeights_.resize(this->size(),1.);
  }
}


bool SprAbsFilter::resetWeights()
{
  assert( copy_ != 0 );

  // if no filter requirements imposed, copy the whole set
  if( copy_->size() == data_->size() ) {
    copyWeights_ = dataWeights_;
    return true;
  }

  // init
  int jstart = 0;

  // loop over points
  for( unsigned int i=0;i<copy_->size();i++ ) {
    const SprPoint* p = (*copy_)[i];
    bool matched = false;
    for( unsigned int j=jstart;j<data_->size();j++ ) {
      if( p == (*data_)[j] ) {
	matched = true;
	jstart = j;
	break;
      }
    }
    if( !matched ) {
      cerr << "resetWeights cannot find matching point." << endl;
      return false;
    }
    copyWeights_[i] = dataWeights_[jstart];
    jstart++;
  }

  // exit
  return true;
}


bool SprAbsFilter::remove(const SprData* data)
{
  // sanity check
  assert( data != 0 );
  if( data->dim() != this->dim() ) {
    cerr << "SprAbsFilter::remove data dimensionality does not match." << endl;
    return false;
  }

  // make a copy
  assert( copy_ != 0 );
  SprData* copy = copy_->emptyCopy();

  // loop through points and accept
  vector<double> copyWeights;
  for( unsigned int i=0;i<copy_->size();i++ ) {
    SprPoint* p = (*copy_)[i];
    if( data->find(p->index_) == 0 ) {
      copy->uncheckedInsert(p);
      copyWeights.push_back(copyWeights_[i]);
    }
  }

  // save copy
  if( ownCopy_ ) delete copy_;
  copy_ = copy;
  copyWeights_ = copyWeights;
  ownCopy_ = true;

  // exit
  return true;
}


bool SprAbsFilter::fastRemove(const SprData* data)
{
  // sanity check
  assert( data != 0 );
  if( data->dim() != this->dim() ) {
    cerr << "SprAbsFilter::remove data dimensionality does not match." << endl;
    return false;
  }

  // make a copy
  assert( copy_ != 0 );
  SprData* copy = copy_->emptyCopy();
  vector<double> copyWeights;

  // loop through points and accept
  unsigned int isplit(0), istart(0);
  for( unsigned int i=0;i<data->size();i++ ) {
    SprPoint* pToRemove = (*data)[i];
    for( isplit=istart;isplit<copy_->size();isplit++ ) {
      SprPoint* p = (*copy_)[isplit];
      if( p == pToRemove ) {
	istart = isplit+1;
	break;
      }
      else {
	copy->uncheckedInsert(p);
	copyWeights.push_back(copyWeights_[isplit]);
      }
    }
  }

  // save copy
  if( ownCopy_ ) delete copy_;
  copy_ = copy;
  copyWeights_ = copyWeights;
  ownCopy_ = true;

  // exit
  return true;
}


double SprAbsFilter::weightInClass(const SprClass& cls) const
{
  assert( copy_ != 0 );
  double w = 0;
  for( unsigned int i=0;i<copy_->size();i++ )
    if( (*copy_)[i]->class_ == cls ) w += copyWeights_[i];
  return w;
}


void SprAbsFilter::scaleWeights(const SprClass& cls, double w)
{
  if( w < SprUtils::eps() ) {
    cerr << "Unable to rescale weights for class " << cls 
	 << " by non-positive factor " << w << endl;
    return;
  }
  assert( data_->size() == dataWeights_.size() );
  for( unsigned int i=0;i<data_->size();i++ ) {
    if( (*data_)[i]->class_ == cls )
      dataWeights_[i] *= w;
  }
  this->filter();
}


void SprAbsFilter::print(std::ostream& os) const
{
  assert( copy_ != 0 );
  os << copy_->dim() << endl;
  vector<string> vars;
  copy_->vars(vars);
  assert( vars.size() == copy_->dim() );
  for( unsigned int d=0;d<vars.size();d++ )
    os << vars[d].c_str() << " ";
  os << endl;
  assert( copyWeights_.size() == copy_->size() );
  for( unsigned int i=0;i<copy_->size();i++ ) {
    os << "# " << i << endl;
    const SprPoint* p = (*copy_)[i];
    for( unsigned int d=0;d<copy_->dim();d++ )
      os << p->x_[d] << " ";
    os << copyWeights_[i] << " " << p->class_ << endl;
  }
}


bool SprAbsFilter::replaceMissing(const SprCut& validRange, int verbose)
{
  // sanity check
  if( validRange.empty() ) return true;

  // make a copy of data
  SprData* copy = data_->copy();
  copyWeights_ = dataWeights_;

  // median vector
  median_.clear();
  median_.resize(copy->dim());

  // loop thru dimensions
  for( unsigned int d=0;d<copy->dim();d++ ) {
    // vector of valid values used for median computation
    // first number in a pair is coordinate, 2nd number is weight
    vector<pair<double,double> > in;
    // points outside the valid range
    vector<SprPoint*> out;

    // print out
    if( verbose > 0 ) 
      cout << "Replacing missing values for dimension " << d << endl;

    // split data into points with and without missing values
    for( unsigned int i=0;i<copy->size();i++ ) {
      SprPoint* p = (*copy)[i];
      bool valid = false;
      double r = (p->x_)[d];
      for( unsigned int j=0;j<validRange.size();j++ ) {
	if( r>validRange[j].first && r<validRange[j].second ) {
	  valid = true;
	  break;
	}
      }
      if( valid )
	in.push_back(pair<double,double>(r,copyWeights_[i]));
      else
	out.push_back(p);
    }

    // sort valid values
    unsigned int validSize = in.size();
    if( validSize < 2 ) {
      cerr << "Less than 2 points with valid values found in dimension " 
	   << d << endl;
      return false;
    }
    stable_sort(in.begin(),in.end(),SAFCmpPairFirst());

    // compute cumulative weights
    double wtot = 0;
    vector<double> w(validSize);
    for( unsigned int i=0;i<validSize;i++ ) {
      w[i] = in[i].second;
      wtot += w[i];
    }

    // find median index
    int medIndex = -1;
    w[0] /= wtot;
    if( w[0] > 0.5 ) {
      medIndex = 0;
    }
    else {
      for( unsigned int i=1;i<validSize;i++ ) {
	w[i] = w[i-1] + w[i]/wtot;
	if( w[i] > 0.5 ) {
	  medIndex = i;
	  break;
	}
      }
    }
    if( medIndex<0 || (unsigned int) medIndex>(validSize-1)) {
      cerr << "Unable to find median index." << endl;
      return false;
    }

    // compute median
    double med = 0;
    if( medIndex == 0 ) 
      med = in[medIndex].first;
    else
      med = 0.5*(in[medIndex].first+in[medIndex-1].first);

    // store the median
    median_[d] = med;
      
    // assign median to all missing points
    for( unsigned int i=0;i<out.size();i++ )
      (out[i]->x_)[d] = med;
  }// end loop thru dimensions

  // store the copy
  if( ownCopy_ ) delete copy_;
  copy_ = copy;
  ownCopy_ = true;

  // exit
  return true;
}


bool SprAbsFilter::decodeClassString(const char* inputClassString,
				     std::vector<SprClass>& classes)
{
  // clear
  classes.clear();

  // parse to strings
  vector<vector<string> > members;
  SprStringParser::parseToStrings(inputClassString,members);

  // sanity check
  if( members.empty() ) {
    cout << "No input classes specified in string " 
	 << inputClassString << " Will use 0 and 1 by default." << endl;
    classes.resize(2);
    classes[0] = 0;
    classes[1] = 1;
    return true;
  }

  // Sanity check for dots and sizes.
  for( unsigned int i=0;i<members.size();i++ ) {
    if( members[i].empty() ) {
      cerr << "Class " << i << " is empty in string " 
	   << inputClassString << endl;
      return false;
    }
    if( ::find(members[i].begin(),members[i].end(),".")!=members[i].end() 
	&& members[i].size()>1 ) {
      cerr << "\".\" does not allow other classes in group " << i << endl;
      return false;
    }
  }

  // All classes are given as a list separated by commas.
  // Typical for multi-class problems.
  if( members.size() < 2 ) {
    if( members[0].size() < 2 ) {
      cerr << "Less than 2 input classes specified in string " 
	   << inputClassString << endl;
      return false;
    }
    classes.resize(members[0].size());
    for( unsigned int i=0;i<members[0].size();i++ )
      classes[i] = atoi(members[0][i].c_str());
    return true;
  }

  // Classes separated by colons.

  // Make sure the dot is used only once.
  int ndot = 0;
  int dottedClass = -1;
  for( unsigned int i=0;i<members.size();i++ ) {
    if( members[i][0] == "." ) {
      ndot++;
      dottedClass = i;
    }
  }
  if( ndot > 1 ) {
    cerr << "More than one class has a dot in its definition." << endl;
    return false;
  }

  // record all classes found in the expression
  vector<int> allButDot;
  for( unsigned int i=0;i<members.size();i++ ) {
    if( (int)i != dottedClass ) {
      for( unsigned int j=0;j<members[i].size();j++ ) {
	allButDot.push_back(atoi(members[i][j].c_str()));
      }
    }
  }

  // fill out classes
  classes.resize(members.size());
  for( unsigned int i=0;i<members.size();i++ ) {
    if( static_cast<int>(i) == dottedClass ) {
      classes[i] = SprClass(allButDot,true);// negate
    }
    else {
      vector<int> cls(members[i].size());
      for( unsigned int j=0;j<members[i].size();j++ ) {
	cls[j] = atoi(members[i][j].c_str());
      }
      classes[i] = SprClass(cls,false);
    }
  }

  // make sure the two class vectors do not contain the same class
  for( unsigned int i=0;i<classes.size()-1;i++ ) {
    for( unsigned int j=i+1;j<classes.size();j++ ) {
      if( classes[i].overlap(classes[j]) == 1 ) {
	cerr << "Classes " << i << " and " << j << " overlap." << endl;
	return false;
      }
    }
  }

  // exit
  return true;
}


bool SprAbsFilter::filterByClass(const char* inputClassString)
{
  // decode the string
  if( !this->chooseClassesFromString(inputClassString) ) 
    return false;

  // make a copy
  SprData* copy = copy_->emptyCopy();

  // loop through points and accept
  vector<double> copyWeights;
  for( unsigned int i=0;i<copy_->size();i++ ) {
    SprPoint* p = (*copy_)[i];
    if( this->category(p) ) {
      copy->uncheckedInsert(p);
      copyWeights.push_back(copyWeights_[i]);
    }
  }

  // save copy
  if( ownCopy_ ) delete copy_;
  copy_ = copy;
  ownCopy_ = true;
  copyWeights_ = copyWeights;

  // exit
  return true;
}


bool SprAbsFilter::store(const char* filename) const
{
  // init
  string fname = filename;
  string cmd;
 
  // check if file exists, delete and issue a warning
  struct stat buf;
  if( stat(fname.c_str(),&buf) == 0 ) {
    cerr << "Warning: file " << fname.c_str() << " will be deleted." << endl;
    cmd = "rm -f ";
    cmd += fname.c_str();
    if( system(cmd.c_str()) != 0 ) {
      cerr << "Attempt to delete file " << fname.c_str() 
           << " terminated with error " << errno << endl;
      return false;
    }
  }
 
  // open output stream
  ofstream outfile(fname.c_str());
  if( !outfile ) {
    cerr << "Cannot open file " << fname.c_str() << endl;
    return false;
  }

  // save data
  this->print(outfile);

  // exit
  return true;
}


bool SprAbsFilter::flatten(const SprClass& cls, 
			   const char* varname, 
			   const std::vector<double>& intervals)
{
  assert( copy_ != 0 );

  // sanity check
  if( intervals.size() < 3 ) {
    cerr << "No intervals are specified for flattening." << endl;
    return false;
  }
  for( unsigned int i=1;i<intervals.size();i++ ) {
    if( intervals[i] <= intervals[i-1] ) {
      cerr << "Intervals are incorrectly specified for flattening: " 
	   << (i-1) << "-" << i
	   << " " << intervals[i-1] << " " << intervals[i] << endl;
      return false;
    }
  }
  double totW = this->weightInClass(cls);
  if( totW < SprUtils::eps() ) {
    cerr << "No events found for flattening in class " << cls << endl;
    return false;
  }

  // find variable in the list of variables in data
  string var = varname;
  vector<string> vars;
  copy_->vars(vars);
  vector<string>::const_iterator iter = ::find(vars.begin(),vars.end(),var);
  if( iter == vars.end() ) {
    cerr << "Variable for flattening not found." << endl;
    return false;
  }
  const unsigned d = iter - vars.begin();

  // make an array of values in this variable
  vector<pair<double,int> > values(copy_->size());
  for( unsigned int i=0;i<copy_->size();i++ ) {
    if( (*copy_)[i]->class_ == cls )
      values[i] = pair<double,int>((*copy_)[i]->x_[d],i);
  }

  // sort the array
  stable_sort(values.begin(),values.end(),SAFCmpPairDIFirst());

  // check interval range
  vector<pair<double,int> >::iterator iter1 
    = find_if(values.begin(),values.end(),
	      not1(bind2nd(SAFCmpPairDIFirstNumber(),intervals[0])));
  if( iter1 == values.end() ) {
    cerr << "All points in class " << cls 
	 << " lie below specified range " << intervals[0] << endl;
    return false;
  }
  const int firstPoint = iter1 - values.begin();
  vector<pair<double,int> >::reverse_iterator iter2 
    = find_if(values.rbegin(),values.rend(),
	      bind2nd(SAFCmpPairDIFirstNumber(),
		      intervals[intervals.size()-1]));
  if( iter2 == values.rend() ) {
    cerr << "All points in class " << cls 
	 << " lie above specified range " << intervals[intervals.size()-1] 
	 << endl;
    return false;
  }
  const int lastPoint = values.size()-1 - (iter2-values.rbegin());

  // compute weights in bins
  assert( copy_->size() == copyWeights_.size() );
  vector<double> binWeights(intervals.size()-1,0);
  int ibin0 = 0;
  for( int i=firstPoint;i<=lastPoint;i++ ) {
    unsigned int ibin = ibin0;
    while( ibin < binWeights.size() ) {
      if( values[i].first>=intervals[ibin] 
	  && values[i].first<intervals[ibin+1] ) {
	binWeights[ibin] += copyWeights_[values[i].second];
	break;
      }
      else
	ibin0 = ++ibin;
    }
  }

  // normalize bin weights
  double averageW = accumulate(binWeights.begin(),binWeights.end(),double(0));
  assert( averageW > 0 );
  double length = intervals[intervals.size()-1] - intervals[0];
  assert( length > 0 );
  averageW /= length;
  for( unsigned int i=0;i<binWeights.size();i++ ) {
    double binLength = intervals[i+1] - intervals[i];
    if( binWeights[i]>0 && binLength>0 ) {
      binWeights[i] /= (binLength*averageW);
      binWeights[i] = 1./binWeights[i];
    }
  }

  // adjust weights
  ibin0 = 0;
  for( int i=firstPoint;i<=lastPoint;i++ ) {
    unsigned int ibin = ibin0;
    while( ibin < binWeights.size() ) {
      if( values[i].first>=intervals[ibin] 
	  && values[i].first<intervals[ibin+1] ) {
	copyWeights_[values[i].second] *= binWeights[ibin];
	break;
      }
      else
	ibin0 = ++ibin;
    }
  }

  // exit
  return true;
}


void SprAbsFilter::allClasses(std::vector<SprClass>& classes) const
{
  assert( copy_ != 0 );
  classes.clear();
  // This algorithm is inefficient - one would need to use std::set.
  // But I want to avoid defining operator< for SprClass.
  for( unsigned int i=0;i<copy_->size();i++ ) {
    const SprPoint* p = (*copy_)[i];
    if( ::find(classes.begin(),classes.end(),p->class_) == classes.end() )
      classes.push_back(SprClass(p->class_));
  }
}


SprData* SprAbsFilter::split(double fractionToKeep, 
			     std::vector<double>& splitWeights,
			     bool randomize,
			     int seed)
{
  assert( copy_ != 0 );

  // init
  splitWeights.clear();

  // sanity check
  if( fractionToKeep < SprUtils::eps() ) {
    cerr << "Fraction of events to keep too small: " << fractionToKeep << endl;
    return 0;
  }

  // if no classes specified, find all
  vector<SprClass> classes = classes_;
  if( classes.empty() )
    this->allClasses(classes);
  assert( !classes.empty() );

  // get weights for each class
  int nClass = classes.size();  
  vector<double> maxW(nClass);
  for( int ic=0;ic<nClass;ic++ )
    maxW[ic] = this->weightInClass(classes[ic]) * fractionToKeep;

  // randomize if necessary
  int N = copy_->size();
  vector<unsigned> indices;
  if( randomize ) {
    SprIntegerPermutator permu(N,seed);
    if( !permu.sequence(indices) ) {
      cerr << "Unable to permute input indices for splitting." << endl;
      return 0;
    }
  }
  else {
    indices.resize(N);
    for( int i=0;i<N;i++ ) indices[i] = i;
  }

  // loop through points for each class
  vector<double> weightInClass(nClass,0);
  vector<int> keepPoint(N,0);
  vector<int> processClass(nClass,1);
  for( int i=0;i<N;i++ ) {
    int ind = indices[i];
    const SprPoint* p = (*copy_)[ind];
    double w = copyWeights_[ind];
    for( int ic=0;ic<nClass;ic++ ) {
      if( classes[ic] == p->class_ ) {
	if( processClass[ic] == 1 ) {
	  weightInClass[ic] += w;
	  if( weightInClass[ic] > maxW[ic] )
	    processClass[ic] = 0;
	  else
	    keepPoint[i] = 1;
	}
	break;// exit loop over classes
      }
    }
  }

  // make data copies to keep and split
  SprData* keep  = copy_->emptyCopy();
  SprData* split = copy_->emptyCopy();

  // loop through points and assign them to copies
  vector<double> keepWeights;
  for( int i=0;i<N;i++ ) {
    int ind = indices[i];
    SprPoint* p = (*copy_)[ind];
    double w = copyWeights_[ind];
    if(      keepPoint[i] == 0 ) {
      split->uncheckedInsert(p);
      splitWeights.push_back(w);
    }
    else if( keepPoint[i] == 1 ) {
      keep->uncheckedInsert(p);
      keepWeights.push_back(w);
    }
  }

  // data to be kept
  if( ownCopy_ ) delete copy_;
  copy_ = keep;
  copyWeights_ = keepWeights;
  ownCopy_ = true;

  // data to be split
  return split;
}


