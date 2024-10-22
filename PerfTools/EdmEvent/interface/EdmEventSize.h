#ifndef PerfTools_EdmEventSize_H
#define PerfTools_EdmEventSize_H

#include <string>
#include <vector>
#include <iosfwd>

namespace perftools {

  /** \class EdmEventSize
   *  Measure the size of each product in an edm::event
   *  Provides the output as an ascii table or root histograms

   *  Based on the original implementation by Luca Lista
   *
   *  Algorithm:
   *  Measure the size of each branch in a tree as the sum of the sizes of 
   *  all its baskets
   *  Estimate the "size in memory" multipling the actual branch size 
   *  by its compression factor
   *
   *  \author Vincenzo Innocente
   */
  class EdmEventSize {
  public:
    /// generic exception
    struct Error {
      Error(std::string const& idescr, int icode) : descr(idescr), code(icode) {}
      std::string descr;
      int code;
    };

    /// the information for each branch
    struct BranchRecord {
      BranchRecord() : compr_size(0.), uncompr_size(0.) {}
      BranchRecord(std::string const& iname, double compr, double uncompr)
          : fullName(iname), name(iname), compr_size(compr), uncompr_size(uncompr) {}
      std::string fullName;
      std::string name;
      double compr_size;
      double uncompr_size;
    };

    typedef std::vector<BranchRecord> Branches;

    /// Constructor
    EdmEventSize();
    /// Constructor and parse
    explicit EdmEventSize(std::string const& fileName, std::string const& treeName = "Events");

    /// read file, compute branch size, sort by size
    void parseFile(std::string const& fileName, std::string const& treeName = "Events");

    /// sort by name
    void sortAlpha();

    /// transform Branch names in "formatted" prodcut identifiers
    void formatNames();

    /// dump the ascii table on "co"
    void dump(std::ostream& co, bool header = true) const;

    /// produce histograms and optionally write them in "file" or as "plot"
    void produceHistos(std::string const& plot, std::string const& file, int top = 0) const;

  private:
    std::string m_fileName;
    int m_nEvents;
    Branches m_branches;
  };

}  // namespace perftools

#endif  // PerfTools_EdmEventSize_H
