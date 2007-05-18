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
	compr_size(0.),  
	uncompr_size(0.) {}
      BranchRecord(std::string const & iname,
		   double compr,  double uncompr) : 
	fullName(iname), name(iname), 
	compr_size(compr), uncompr_size(uncompr){}
      std::string fullName;
      std::string name;
      double compr_size;
      double uncompr_size;
    };

    typedef std::vector<BranchRecord> Branches;

    EdmEventSize();
    explicit EdmEventSize(std::string const & fileName);
    
    void parseFile(std::string const & fileName);

    void sortAlpha();

    void shortNames();

    void dump(std::ostream & co) const;

    void produceHistos(std::string const & plot, std::string const & file, int top=0) const; 

  private:
    Branches m_branches;

  };

}

#endif // PerfTools_EdmEventSize_H
