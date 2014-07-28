#ifndef GeneratorInterface_LHEInterface_LHERunInfo_h
#define GeneratorInterface_LHEInterface_LHERunInfo_h

#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"
#include "SimDataFormats/GeneratorProducts/interface/LHERunInfoProduct.h"

#ifndef XERCES_CPP_NAMESPACE_QUALIFIER
#	define UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
#	define XERCES_CPP_NAMESPACE_QUALIFIER dummy::
namespace dummy {
  class DOMNode;
  class DOMDocument;
}
#endif

namespace lhef {

  class LHERunInfo {
  public:
    LHERunInfo(std::istream &in);
    LHERunInfo(const HEPRUP &heprup);
    LHERunInfo(const HEPRUP &heprup,
	       const std::vector<LHERunInfoProduct::Header> &headers,
	       const std::vector<std::string> &comments);
    LHERunInfo(const LHERunInfoProduct &product);
    ~LHERunInfo();

    class Header : public LHERunInfoProduct::Header {
    public:
      Header();
      Header(const std::string &tag);
      Header(const Header &orig);
      Header(const LHERunInfoProduct::Header &orig);
      ~Header();

#ifndef UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
      const XERCES_CPP_NAMESPACE_QUALIFIER DOMNode
	*getXMLNode() const;
#endif

    private:
      mutable XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *xmlDoc;
    };

    const HEPRUP *getHEPRUP() const { return &heprup; } 

    bool operator == (const LHERunInfo &other) const;
    inline bool operator != (const LHERunInfo &other) const
    { return !(*this == other); }

    const std::vector<Header> &getHeaders() const { return headers; }
    const std::vector<std::string> &getComments() const { return comments; }

    std::vector<std::string> findHeader(const std::string &tag) const;

    void addHeader(const Header &header) { headers.push_back(header); }
    void addComment(const std::string &line) { comments.push_back(line); }

    enum CountMode {
      kTried = 0,
      kSelected,
      kKilled,
      kAccepted
    };

    struct XSec {
    public:
    XSec() : value_(0.0), error_(0.0) {}
    XSec(double v, double e): value_(v), error_(e){}
      double value(){return value_;}
      double error(){return error_;}
    private:
      double	value_;
      double	error_;
    };

    void count(int process, CountMode count, double eventWeight = 1.0,
	       double brWeight = 1.0, double matchWeight = 1.0);
    XSec xsec() const;
    void statistics() const;

    std::pair<int, int> pdfSetTranslation() const;

    struct Counter {
    public:
    Counter() : n_(0), sum_(0.0), sum2_(0.0) {}
    Counter(unsigned int n1, double sum1, double sum21) 
    :n_(n1), sum_(sum1), sum2_(sum21) {}
      inline void add(double weight)
      {
	n_++;
	sum_ += weight;
	sum2_ += weight * weight;
      }
      unsigned int n() const {return n_;}
      double sum() const {return sum_;}
      double sum2() const {return sum2_;}
    private: 
      unsigned int	n_;
      double		sum_;
      double		sum2_;
    };

    struct Process {
    public:
    Process(): process_(-1), heprupIndex_(-1){}
    Process(int id): process_(id), heprupIndex_(-1){}
      // accessors
      int process() const {return process_;} 
      unsigned int heprupIndex() const {return heprupIndex_;}
      Counter tried() const {return tried_;}
      Counter selected() const {return selected_;}
      Counter killed() const {return killed_;}
      Counter accepted() const {return accepted_;}
      Counter acceptedBr() const {return acceptedBr_;}	        

      // setters
      void setProcess(int id) {process_ = id;}
      void setHepRupIndex(int id) {heprupIndex_ = id;}
      void addTried(double w) {tried_.add(w);}
      void addSelected(double w) {selected_.add(w);}
      void addKilled(double w) {killed_.add(w);}
      void addAccepted(double w) {accepted_.add(w);}
      void addAcceptedBr(double w) {acceptedBr_.add(w);}	        

    private:
      int		process_;
      unsigned int	heprupIndex_;
      Counter		tried_;
      Counter		selected_;
      Counter		killed_;
      Counter		accepted_;
      Counter		acceptedBr_;
    };


  private:
    void init();

    HEPRUP				heprup;
    std::vector<Process>		processes;
    std::vector<Header>		headers;
    std::vector<std::string>	comments;

  public:

    struct ProcessLumi {
    public:
      ProcessLumi(){}
      ProcessLumi(int id):thisProcess_(id){}
      // accessors
      XSec  getHepXSec() const {return hepXSec_;}
      Process getProcess() const {return thisProcess_;}
      // setters
      void setHepXSec(double value, double error) {hepXSec_ = XSec(value,error);}
      void setProcess(int id) {thisProcess_.setProcess(id);}
      void setHepRupIndex(int id) {thisProcess_.setHepRupIndex(id);}
      void addTried(double w) {thisProcess_.addTried(w);}
      void addSelected(double w) {thisProcess_.addSelected(w);}
      void addKilled(double w) {thisProcess_.addKilled(w);}
      void addAccepted(double w) {thisProcess_.addAccepted(w);}
      void addAcceptedBr(double w) {thisProcess_.addAcceptedBr(w);}	        

    private:  
      Process thisProcess_;
      XSec    hepXSec_;
    };

    const std::vector<ProcessLumi>& getLumiProcesses() const {return processesLumi;}
    const int getHEPIDWTUP() const {return heprup.IDWTUP;}
    void initLumi();
    XSec xsecLumi() const;
  private:
    std::vector<ProcessLumi>       	processesLumi;

  };


} // namespace lhef

#ifdef UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
#	undef XERCES_CPP_NAMESPACE_QUALIFIER
#endif

#endif // GeneratorRunInfo_LHEInterface_LHERunInfo_h
