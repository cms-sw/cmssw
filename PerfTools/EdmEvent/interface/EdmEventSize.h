#ifndef PerfTools_EdmEventSize_H
#define PerfTools_EdmEventSize_H

#include<string>
#include<vector>
#include<iosfwd>

namespace perftools {

  class EdmEventSize {
  public:
    struct Error {
      Error(std::string const & idescr, int icode) :
	descr(idescr), code(icode){}
      std::string descr;
      int code;
    };

    struct BranchRecord {
      BranchRecord() : 
	compr_size(0),  
	uncompr_size(0) {}
      std::string name;
      size_t compr_size;
      size_t uncompr_size;
    };

    typedef std::vector<BranchRecord> Branches;

    EdmEventSize();
    explicit EdmEventSize(std::string const & fileName);
    
    void parseFile(std::string const & fileName);

    void sortAlpha();

    void dump(std::ostream & co) const;

    void produceHistos(std::string const & plot, std::string const & file) const; 

  private:
    Branches m_branches;

  };

}

#endif // PerfTools_EdmEventSize_H
