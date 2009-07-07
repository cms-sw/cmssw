//------------------------------------------------------------------------
// File and Version Information:
//      $Id: SprAbsFilter.hh,v 1.4 2007/12/01 01:29:45 narsky Exp $
//
// Description:
//      Class SprAbsFilter :
//         filters SprData according to a certain criterion
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _SprAbsFilter_HH
#define _SprAbsFilter_HH

#include "PhysicsTools/StatPatternRecognition/interface/SprData.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprDefs.hh"
#include "PhysicsTools/StatPatternRecognition/interface/SprClass.hh"

#include <vector>
#include <string>
#include <iostream>

class SprPoint;


class SprAbsFilter
{
public:
  virtual ~SprAbsFilter() {
    if( ownData_ ) {
      delete data_;
      ownData_ = false;
    }
    if( ownCopy_ ) {
      delete copy_;
      ownCopy_ = false;
    }
  }

  /*
    Original filter on data.
  */
  SprAbsFilter(const SprData* data, bool ownData=false);
  SprAbsFilter(const SprData* data, 
	       const std::vector<SprClass>& classes,
	       bool ownData=false);

  /*
    Filter with supplied weights.
  */
  SprAbsFilter(const SprData* data, const std::vector<double>& weights,
	       bool ownData=false);
  SprAbsFilter(const SprData* data, 
	       const std::vector<SprClass>& classes,
	       const std::vector<double>& weights,
	       bool ownData=false);

  /*
    Consecutive filter.
  */
  SprAbsFilter(const SprAbsFilter& filter);

  // accept or reject a point
  virtual bool pass(const SprPoint* p) const = 0;

  // specific reset
  virtual bool reset() = 0;

  // Filter input data: applied to original data.
  virtual bool filter();

  // irreversible filter => events that do not pass this filter are permanently
  // removed
  virtual bool irreversibleFilter();

  // Return filter to original state. All weights, category and index cuts,
  // and all cuts specified by setCut() are cleared.
  virtual void clear();

  // determine if the point belongs to one of the specified classes
  bool category(const SprPoint* p) const;

  // Choose classes. By default all classes from input data are used.
  void chooseClasses(const std::vector<SprClass>& classes) {
    classes_ = classes;
  }
  void classes(std::vector<SprClass>& classes) const;

  // Find all classes in data.
  void allClasses(std::vector<SprClass>& classes) const;

  /*
    This method allows to select an arbitrary number of classes split
    in an arbitrary number of groups.

    The input for this method is a list of classes separated by commas
    and grouped by colons. By default, the first group is treated as
    background and the 2nd group as signal. For example, '1,3:2,4'
    will force classes 1 and 3 to be treated as background and classes
    2 and 4 as signal. If you supply more than 2 groups, the first two
    will be used for two-class methods and all groups will be used
    for multi-class methods.

    A dot '.' means 'all classes but those in other groups'. For
    example, if you enter '.:2,3' for a two-class method, classes 2
    and 3 will be treated as signal and all other classes found in
    input data will be treated as background; vice versa for '2,3:.'.
    For a multi-class problem, if you enter, for example, '1:.:2,3',
    it means that the 1st category is class 1, the 3rd category are
    classes 2 and 3, and the 2nd category is all classes but 1, 2 and
    3.
  */
  bool chooseClassesFromString(const char* inputClassString) {
    return SprAbsFilter::decodeClassString(inputClassString,classes_);
  }
  static bool decodeClassString(const char* inputClassString,
				std::vector<SprClass>& classes);

  /*
    filterByClass() is a special method that is applied not to the basic
    data but to the current copy. It acts as a consecutive filter.
  */
  bool filterByClass(const char* inputClassString);

  // Define index range to look at part of data.
  // By default all data are looked at.
  void setIndexRange(int imin, int imax) { 
    imin_ = imin;
    imax_ = imax;
  }
  void resetIndexRange() {
    imin_ = 0;
    imax_ = 0;
  }

  // print out data content
  void print(std::ostream& os) const;

  // store data into a file
  bool store(const char* filename) const;

  // Splits data using the specified fraction.
  // Keeps fractionToKeep in this filter and splits 1.-fractionToKeep
  // off to a new SprData object. Returns weights for the split-off events.
  // If randomize is set to true, data points will be permuted.
  // Initial seed will be used for permutation.
  SprData* split(double fractionToKeep, 
		 std::vector<double>& splitWeights,
		 bool randomize,
		 int seed=0);

  // Remove a bunch of points from the sample.
  // This method compares points by their unique id (SprPoint::index_)
  // and removes points from the original sample.
  bool remove(const SprData* data);

  // Fast algorithm for removing points from data.
  // Assumes that the points are sorted in the same order in
  // the data being removed and the data from which they are removed.
  // Use with care!!!
  bool fastRemove(const SprData* data);

  /*
    Replaces missing values (i.e., values outside the given valid range)
    by medians of distributions of valid points in the supplied data.
    This method has an irreversible effect - like irreversibleFilter().
    The median() method returns computed medians.
  */
  bool replaceMissing(const SprCut& validRange, int verbose=0);
  void median(std::vector<double>& med) const { med = median_; }

  /*
    Flatten the input data in the specified variable using the supplied
    intervals for binning. This method adjusts weights to enforce the
    uniformity of the distribution in the given variable. The weights
    can be restored to original values using clear() or resetWeights().
  */
  bool flatten(const SprClass& cls, const char* varname, 
	       const std::vector<double>& intervals);

  // wrapper accessors to data
  const SprData* data() const { return copy_; }
  SprPoint* operator[](int i) const { return (*copy_)[i]; }
  inline SprPoint* at(int i) const { return copy_->at(i); }
  std::string label() const { return copy_->label(); }
  unsigned dim() const { return copy_->dim(); }
  void vars(std::vector<std::string>& vars) const { copy_->vars(vars); }
  unsigned size() const { return copy_->size(); }
  bool empty() const { return copy_->empty(); }
  unsigned ptsInClass(const SprClass& cls) const { 
    return copy_->ptsInClass(cls); 
  }
  SprPoint* find(unsigned index) const { return copy_->find(index); }
  int dimIndex(const char* var) const { return copy_->dimIndex(var); }
  SprData* emptyCopy() const { return copy_->emptyCopy(); }
  SprData* copy() const { return copy_->copy(); }

  // weight accessors
  double w(int i) const { return copyWeights_[i]; }
  void weights(std::vector<double>& weights) const { weights = copyWeights_; }
  inline double atw(int i) const;
  double weightInClass(const SprClass& cls) const;

  // weight modifiers
  //
  // Weights set by the following methods will be cleared by clear().
  // resetWeights(), which is also run as part of clear(), will restore
  // weights to the values supplied to the constructor.
  //
  void uncheckedSetW(int i, double w) { copyWeights_[i] = w; }
  bool setW(int i, double w);
  bool resetWeights();
  bool normalizeWeights() {
    return this->normalizeWeights(classes_,1.);
  }
  bool normalizeWeights(const std::vector<SprClass>& classes,
			double totalWeight=1.);
  void setUniformWeights();// sets all weights equal to 1
  bool setWeights(const std::vector<double>& weights);

  //
  // The following modifiers have a permanent effect. Weights set
  // by these modifiers cannot be cleared.
  //
  void scaleWeights(const SprClass& cls, 
		    double w);// scales weights in cls by factor w
  bool setPermanentWeights(const std::vector<double>& weights);

protected:
  const SprData* data_;
  const SprData* copy_;
  bool ownData_;
  bool ownCopy_;
  std::vector<double> dataWeights_;
  std::vector<double> copyWeights_;
  std::vector<SprClass> classes_;
  unsigned int imin_;
  unsigned int imax_;
  std::vector<double> median_;
};

inline double SprAbsFilter::atw(int i) const
{
  if( i>=0 && i<(int)copyWeights_.size() ) return copyWeights_[i];
  std::cerr << "Index out of range for weights " << i << " " 
	    << copyWeights_.size() << std::endl;
  return 0;
}

#endif

