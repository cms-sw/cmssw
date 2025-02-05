#ifndef PerfTools_EdmEventSize_H
#define PerfTools_EdmEventSize_H

#include <string>
#include <vector>
#include <iosfwd>

namespace perftools {

  /** \class EdmEventSize
   *  Measure the size of each product in an edm::event
   *  Provides the output as an ascii table, json file or root histograms

   *  Based on the original implementation by Luca Lista
   *
   *  Algorithm:
   *  Measure the size of each branch in a tree as the sum of the sizes of 
   *  all its baskets
   *  Estimate the "size in memory" multipling the actual branch size 
   *  by its compression factor
   *  Optionally, measure the size of each leaf in a branch and calculate
   *  the overhead of the branch size with respect to the sum of the leaf sizes
   *
   *  \author Vincenzo Innocente
   *  \author Simone Rossi Tisbeni
   */

  enum class EdmEventMode { Branches, Leaves };

  template <EdmEventMode M>
  class EdmEventSize {
  public:
    enum class Format { text, json };

    /// generic exception
    struct Error {
      Error(std::string const& idescr, int icode) : descr(idescr), code(icode) {}
      std::string descr;
      int code;
    };

    /// the information for each branch
    struct Record {
      Record() : compr_size(0.), uncompr_size(0.) {}
      Record(std::string const& iname, size_t inEvents, size_t compr, size_t uncompr)
          : name(iname), nEvents(inEvents), compr_size(compr), uncompr_size(uncompr) {
        if constexpr (M == EdmEventMode::Branches) {
          type = name;
        } else if constexpr (M == EdmEventMode::Leaves) {
          if (name.find('.') == std::string::npos) {
            type = name;
            label = "";
          } else {
            type = name.substr(0, name.find('.'));
            label = name.substr(name.find('.') + 1);
          }
        }
      }
      std::string name;
      std::string type;
      std::string label;
      size_t nEvents;
      size_t compr_size;
      size_t uncompr_size;
    };

    typedef std::vector<Record> Records;

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

    /// dump the json table on "co"
    void dumpJson(std::ostream& co) const;

    /// produce histograms and optionally write them in "file" or as "plot"
    void produceHistos(std::string const& plot, std::string const& file, int top = 0) const;

  private:
    std::string m_fileName;
    int m_nEvents;
    Records m_records;
  };

  template class perftools::EdmEventSize<perftools::EdmEventMode::Leaves>;

  template class perftools::EdmEventSize<perftools::EdmEventMode::Branches>;

}  // namespace perftools

#endif  // PerfTools_EdmEventSize_H
