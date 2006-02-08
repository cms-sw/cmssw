#ifndef Common_TriggerResults_h
#define Common_TriggerResults_h

/*
  Author: Jim Kowalkowski 13-01-06
  $Id: TriggerResults.h,v 1.3 2006/02/07 07:51:41 wmtan Exp $

  The trigger path results are maintained here as a sequence of bits,
  one per trigger path.  They are assigned in the order they appeared in
  the process-level pset.

  Implementation notes:

  1) cannot get bitset to store properly, so vector<bool> will be saved
  for now

  2) there is as of this writing, no place in the file to store parameter
  sets or run information.  The trigger bit descriptions need to be stored
  in these sections.  This object stores the parameter ID, which can be
  used to locate the parameter in the file when that option becomes available.
  For now, this object contains the trigger paths as a vector of strings.

 */

#include "DataFormats/Common/interface/ParameterSetID.h"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>
#include <bitset>
#include <iostream>

namespace edm
{
  class TriggerResults
  {
  public:
    typedef std::bitset<2048> BitMask;
    typedef std::vector<bool> BitVector;
    typedef std::vector<std::string> Strings; 
    
    TriggerResults(): insync_(false),bits_(),stored_(),id_(),saved_names_() { }
    
    TriggerResults(const BitMask& in,const Strings& trigger_names):
      insync_(true),bits_(in),stored_(trigger_names.size()),id_(),saved_names_(trigger_names)
    { toVector(); }

    // Cannot work until parameter set ID can be resolved
    // TriggerResults(const BitMask& in,edm::ParameterSetID id):
    //   insync_(true),bits_(in),stored_(),id_(),saved_names_()
    // { toVector(); }

    TriggerResults(const BitMask& in,edm::ParameterSetID id, 
		   const Strings& trigger_names):
      insync_(true),bits_(in),stored_(trigger_names.size()),id_(id),saved_names_(trigger_names)
    { toVector(); }

    bool fail() const { toBitset(); return bits_.none(); }
    bool pass() const { toBitset(); return bits_.any();  }

    int numBitsUsed() const { return saved_names_.size(); }

    // these are not efficient ways to implement these functions
    // and need to be improved

    bool isSet(const std::string& path_name) const
    {
      toBitset();
      Strings::const_iterator i = find(saved_names_.begin(),
				       saved_names_.end(),path_name);
      return i==saved_names_.end() ? false : 
	bits_[distance(saved_names_.begin(),i)];
    }

    bool isSet(int bit_pos) const
    {
      toBitset();
      return bits_[bit_pos];
    }

    bool isSet(const BitMask& compare) const
    {
      toBitset();
      return (bits_ & compare).any();
    }

    bool isSet(const BitVector& compare) const
    {
      toBitset();
      bool result = false;

      if(stored_.size() <= compare.size())
	result = std::inner_product(stored_.begin(),stored_.end(),
				    compare.begin(),false);
      else
	result = std::inner_product(compare.begin(),compare.end(),
				    stored_.begin(),false);

      return result;
    }

    bool applyMasks(const BitVector& accept,const BitVector& veto) const
    {
      toBitset();
      // not implemented
      return false;
    }

  private:
    void toBitset() const
    {
      if(insync_) return;
      for(unsigned int i=0;i<saved_names_.size();++i)
	{
	  bits_[i]=stored_[i];
	  vetobits_[i]=!stored_[i];
	}
      insync_=true;
    }

    void toVector()
    {
      // needed until bitset can be successfully stored/retrieved
      for(unsigned int i=0;i<saved_names_.size();++i) stored_[i]=bits_[i];
    }

    // current implementation is a simple one
    // could change in the future
    mutable bool insync_; // transient
    mutable BitMask bits_; // transient for now
    mutable BitMask vetobits_; // transient for now
    BitVector stored_;
    edm::ParameterSetID id_;
    Strings saved_names_;
    // job that generated this pset, used to locate path names
    // next is needed until the file metadata caches psets
  };

  inline std::ostream& operator<<(std::ostream& ost, const TriggerResults& tr)
  {
	int tot = tr.numBitsUsed();
	for(int i=0;i<tot;++i)
	{
		ost << tr.isSet(i)?"1":"0";
	}
    return ost;
  }

}

#endif
