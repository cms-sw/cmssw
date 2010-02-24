#ifndef HLTReco_HLTPrescaleTable_h
#define HLTReco_HLTPrescaleTable_h

/** \class trigger::HLTPrescaleTable
 *
 *  The single EDProduct containing the HLT Prescale Table
 *
 *  $Date: 2010/02/24 11:23:30 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include <map>
#include <string>
#include <vector>
#include <cassert>

namespace trigger
{

  using std::map;
  using std::string;
  using std::vector;

  /// The single EDProduct containing the HLT Prescale Table
  class HLTPrescaleTable {

  /// data members
  private:
    /// index number of default prescale set to use
    unsigned int set_;
    /// names of prescale sets
    vector<string> labels_;
    /// prescale sets keyed on trigger path name
    map<string,vector<unsigned int> > table_;
    /// consistency condition: all vectors must have the same length

  ///methods
  public:

    /// number of prescale sets available
    unsigned int size() const {return labels_.size();}

    /// high-level user access method: prescale for given trigger path
    unsigned int prescale(const string& trigger) const {
      return prescale(set_,trigger);
    }

    /// high-level user access method: prescale for given trigger path
    unsigned int prescale(unsigned int set, const string& trigger) const {
      const map<string,vector<unsigned int> >::const_iterator it(table_.find(trigger));
      if ((it==table_.end()) || (set>=it->second.size())) {
	return 1;
      } else {
	return it->second[set];
      }
    }

    /// default constructor
    HLTPrescaleTable(): set_(0), labels_(), table_() { }

    /// real constructor taking payload
    HLTPrescaleTable(unsigned int set, const vector<string>& labels, const map<string,vector<unsigned int> >& table):
     set_(set), labels_(labels), table_(table) {
      /// checking consistency
      const unsigned int n(labels_.size());
      assert (set_<n);
      const map<string,vector<unsigned int> >::const_iterator ib(table_.begin());
      const map<string,vector<unsigned int> >::const_iterator ie(table_.end());
      for (map<string,vector<unsigned int> >::const_iterator it=ib; it!=ie; ++it) {
	assert (it->second.size()==n);
      }
    }

    /// low-level const accessors for data members
    unsigned int set() const {return set_;}
    const vector<string>& labels() const {return labels_;}
    const map<string,vector<unsigned int> >& table() const {return table_;}

  };
}

#endif
