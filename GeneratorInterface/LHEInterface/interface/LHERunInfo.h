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
#define UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
#define XERCES_CPP_NAMESPACE_QUALIFIER dummy::
namespace dummy {
  class DOMNode;
  class DOMDocument;
}  // namespace dummy
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
      const XERCES_CPP_NAMESPACE_QUALIFIER DOMNode *getXMLNode() const;
#endif

    private:
      mutable XERCES_CPP_NAMESPACE_QUALIFIER DOMDocument *xmlDoc;
    };

    const HEPRUP *getHEPRUP() const { return &heprup; }

    bool operator==(const LHERunInfo &other) const;
    inline bool operator!=(const LHERunInfo &other) const { return !(*this == other); }

    const std::vector<Header> &getHeaders() const { return headers; }
    const std::vector<std::string> &getComments() const { return comments; }

    std::vector<std::string> findHeader(const std::string &tag) const;

    void addHeader(const Header &header) { headers.push_back(header); }
    void addComment(const std::string &line) { comments.push_back(line); }

    enum CountMode { kTried = 0, kSelected, kKilled, kAccepted };

    struct XSec {
    public:
      XSec() : value_(0.0), error_(0.0) {}
      XSec(double v, double e) : value_(v), error_(e) {}
      double value() { return value_; }
      double error() { return error_; }

    private:
      double value_;
      double error_;
    };

    void count(int process, CountMode count, double eventWeight = 1.0, double brWeight = 1.0, double matchWeight = 1.0);
    XSec xsec() const;
    void statistics() const;

    std::pair<int, int> pdfSetTranslation() const;

    struct Counter {
    public:
      Counter() : n_(0), sum_(0.0), sum2_(0.0) {}
      Counter(unsigned int n1, double sum1, double sum21) : n_(n1), sum_(sum1), sum2_(sum21) {}
      inline void add(double weight) {
        n_++;
        sum_ += weight;
        sum2_ += weight * weight;
      }
      unsigned int n() const { return n_; }
      double sum() const { return sum_; }
      double sum2() const { return sum2_; }

    private:
      unsigned int n_;
      double sum_;
      double sum2_;
    };

    struct Process {
    public:
      Process() : process_(-1), heprupIndex_(-1), nPassPos_(0), nPassNeg_(0), nTotalPos_(0), nTotalNeg_(0) {}
      Process(int id) : process_(id), heprupIndex_(-1), nPassPos_(0), nPassNeg_(0), nTotalPos_(0), nTotalNeg_(0) {}
      // accessors
      int process() const { return process_; }
      unsigned int heprupIndex() const { return heprupIndex_; }
      XSec getLHEXSec() const { return lheXSec_; }

      unsigned int nPassPos() const { return nPassPos_; }
      unsigned int nPassNeg() const { return nPassNeg_; }
      unsigned int nTotalPos() const { return nTotalPos_; }
      unsigned int nTotalNeg() const { return nTotalNeg_; }

      Counter tried() const { return tried_; }
      Counter selected() const { return selected_; }
      Counter killed() const { return killed_; }
      Counter accepted() const { return accepted_; }
      Counter acceptedBr() const { return acceptedBr_; }

      // setters
      void setProcess(int id) { process_ = id; }
      void setHepRupIndex(int id) { heprupIndex_ = id; }
      void setLHEXSec(double value, double error) { lheXSec_ = XSec(value, error); }

      void addNPassPos(unsigned int n = 1) { nPassPos_ += n; }
      void addNPassNeg(unsigned int n = 1) { nPassNeg_ += n; }
      void addNTotalPos(unsigned int n = 1) { nTotalPos_ += n; }
      void addNTotalNeg(unsigned int n = 1) { nTotalNeg_ += n; }

      void addTried(double w) { tried_.add(w); }
      void addSelected(double w) { selected_.add(w); }
      void addKilled(double w) { killed_.add(w); }
      void addAccepted(double w) { accepted_.add(w); }
      void addAcceptedBr(double w) { acceptedBr_.add(w); }

    private:
      int process_;
      XSec lheXSec_;
      unsigned int heprupIndex_;
      unsigned int nPassPos_;
      unsigned int nPassNeg_;
      unsigned int nTotalPos_;
      unsigned int nTotalNeg_;
      Counter tried_;
      Counter selected_;
      Counter killed_;
      Counter accepted_;
      Counter acceptedBr_;
    };

  private:
    void init();

    HEPRUP heprup;
    std::vector<Process> processes;
    std::vector<Header> headers;
    std::vector<std::string> comments;

  public:
    const std::vector<Process> &getLumiProcesses() const { return processesLumi; }
    const int getHEPIDWTUP() const { return heprup.IDWTUP; }
    void initLumi();

  private:
    std::vector<Process> processesLumi;
  };

}  // namespace lhef

#ifdef UNDEF_XERCES_CPP_NAMESPACE_QUALIFIER
#undef XERCES_CPP_NAMESPACE_QUALIFIER
#endif

#endif  // GeneratorRunInfo_LHEInterface_LHERunInfo_h
