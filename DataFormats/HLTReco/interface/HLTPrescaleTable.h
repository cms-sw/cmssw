#ifndef HLTReco_HLTPrescaleTable_h
#define HLTReco_HLTPrescaleTable_h

/** \class trigger::HLTPrescaleTable
 *
 *  The single EDProduct containing the HLT Prescale Table
 *
 *  $Date: 2010/10/14 23:00:36 $
 *  $Revision: 1.7 $
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
  /// The single EDProduct containing the HLT Prescale Table
  class HLTPrescaleTable {

  /// data members
  private:
    /// index number of default prescale set to use
    unsigned int set_;
    /// names of prescale sets
    std::vector<std::string> labels_;
    /// prescale sets keyed on trigger path name
    std::map<std::string,std::vector<unsigned int> > table_;
    /// consistency condition: all vectors must have the same length

  ///methods
  public:

    /// number of prescale sets available
    unsigned int size() const {return labels_.size();}

    /// high-level user access method: prescale for given trigger path
    unsigned int prescale(const std::string& trigger) const {
      return prescale(set_,trigger);
    }

    /// high-level user access method: prescale for given trigger path
    unsigned int prescale(unsigned int set, const std::string& trigger) const {
      const std::map<std::string,std::vector<unsigned int> >::const_iterator it(table_.find(trigger));
      if ((it==table_.end()) || (set>=it->second.size())) {
	return 1;
      } else {
	return it->second[set];
      }
    }

    /// default constructor
    HLTPrescaleTable(): set_(0), labels_(), table_() { }

    /// real constructor taking payload
    HLTPrescaleTable(unsigned int set, const std::vector<std::string>& labels, const std::map<std::string,std::vector<unsigned int> >& table):
     set_(set), labels_(labels), table_(table) {
      /// checking consistency
      const unsigned int n(labels_.size());
      assert((((set_==0)&&(n==0)) || (set_<n)));
      const std::map<std::string,std::vector<unsigned int> >::const_iterator ib(table_.begin());
      const std::map<std::string,std::vector<unsigned int> >::const_iterator ie(table_.end());
      for (std::map<std::string,std::vector<unsigned int> >::const_iterator it=ib; it!=ie; ++it) {
	assert (it->second.size()==n);
      }
    }

    /// merge rule - just checking equality
    bool isProductEqual(const HLTPrescaleTable& that) const {
      return ((set()==that.set()) && (labels()==that.labels()) && (table()==that.table()));
    }

    /// low-level const accessors for data members
    unsigned int set() const {return set_;}
    const std::vector<std::string>& labels() const {return labels_;}
    const std::map<std::string,std::vector<unsigned int> >& table() const {return table_;}

  };
}

#endif
