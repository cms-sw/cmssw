#include <algorithm>
#include <iostream>
#include <iomanip>
#include <string>
#include <cctype>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>

#include <xercesc/dom/DOM.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/GeneratorProducts/interface/LesHouches.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"

#include "XMLUtils.h"

XERCES_CPP_NAMESPACE_USE

static int skipWhitespace(std::istream &in) {
  int ch;
  do {
    ch = in.get();
  } while (std::isspace(ch));
  if (ch != std::istream::traits_type::eof())
    in.putback(ch);
  return ch;
}

namespace lhef {

  LHERunInfo::LHERunInfo(std::istream &in) {
    in >> heprup.IDBMUP.first >> heprup.IDBMUP.second >> heprup.EBMUP.first >> heprup.EBMUP.second >>
        heprup.PDFGUP.first >> heprup.PDFGUP.second >> heprup.PDFSUP.first >> heprup.PDFSUP.second >> heprup.IDWTUP >>
        heprup.NPRUP;
    if (!in.good())
      throw cms::Exception("InvalidFormat") << "Les Houches file contained invalid"
                                               " header in init section."
                                            << std::endl;

    heprup.resize();

    for (int i = 0; i < heprup.NPRUP; i++) {
      in >> heprup.XSECUP[i] >> heprup.XERRUP[i] >> heprup.XMAXUP[i] >> heprup.LPRUP[i];
      if (!in.good())
        throw cms::Exception("InvalidFormat") << "Les Houches file contained invalid data"
                                                 " in header payload line "
                                              << (i + 1) << "." << std::endl;
    }

    while (skipWhitespace(in) == '#') {
      std::string line;
      std::getline(in, line);
      comments.push_back(line + "\n");
    }

    if (!in.eof())
      edm::LogInfo("Generator|LHEInterface")
          << "Les Houches file contained spurious"
             " content after the regular data (this is normal for LHEv3 files for now)."
          << std::endl;

    init();
  }

  LHERunInfo::LHERunInfo(const HEPRUP &heprup) : heprup(heprup) { init(); }

  LHERunInfo::LHERunInfo(const HEPRUP &heprup,
                         const std::vector<LHERunInfoProduct::Header> &headers,
                         const std::vector<std::string> &comments)
      : heprup(heprup) {
    std::copy(headers.begin(), headers.end(), std::back_inserter(this->headers));
    std::copy(comments.begin(), comments.end(), std::back_inserter(this->comments));

    init();
  }

  LHERunInfo::LHERunInfo(const LHERunInfoProduct &product) : heprup(product.heprup()) {
    std::copy(product.headers_begin(), product.headers_end(), std::back_inserter(headers));
    std::copy(product.comments_begin(), product.comments_end(), std::back_inserter(comments));

    init();
  }

  LHERunInfo::~LHERunInfo() {}

  void LHERunInfo::init() {
    for (int i = 0; i < heprup.NPRUP; i++) {
      Process proc;
      proc.setProcess(heprup.LPRUP[i]);
      proc.setHepRupIndex((unsigned int)i);
      processes.push_back(proc);
    }

    std::sort(processes.begin(), processes.end());
  }

  void LHERunInfo::initLumi() {
    processesLumi.clear();
    for (int i = 0; i < heprup.NPRUP; i++) {
      Process proc;
      proc.setProcess(heprup.LPRUP[i]);
      proc.setHepRupIndex((unsigned int)i);
      proc.setLHEXSec(heprup.XSECUP[i], heprup.XERRUP[i]);
      processesLumi.push_back(proc);
    }

    std::sort(processesLumi.begin(), processesLumi.end());
  }

  bool LHERunInfo::operator==(const LHERunInfo &other) const { return heprup == other.heprup; }

  void LHERunInfo::count(int process, CountMode mode, double eventWeight, double brWeight, double matchWeight) {
    std::vector<Process>::iterator proc = std::lower_bound(processes.begin(), processes.end(), process);
    if (proc == processes.end() || proc->process() != process)
      return;

    std::vector<Process>::iterator procLumi = std::lower_bound(processesLumi.begin(), processesLumi.end(), process);
    if (procLumi == processesLumi.end() || procLumi->process() != process)
      return;

    switch (mode) {
      case kAccepted:
        proc->addAcceptedBr(eventWeight * brWeight * matchWeight);
        proc->addAccepted(eventWeight * matchWeight);
        procLumi->addAcceptedBr(eventWeight * brWeight * matchWeight);
        procLumi->addAccepted(eventWeight * matchWeight);
        [[fallthrough]];
      case kKilled:
        proc->addKilled(eventWeight * matchWeight);
        procLumi->addKilled(eventWeight * matchWeight);
        if (eventWeight > 0) {
          proc->addNPassPos();
          procLumi->addNPassPos();
        } else {
          proc->addNPassNeg();
          procLumi->addNPassNeg();
        }
        [[fallthrough]];
      case kSelected:
        proc->addSelected(eventWeight);
        procLumi->addSelected(eventWeight);
        if (eventWeight > 0) {
          proc->addNTotalPos();
          procLumi->addNTotalPos();
        } else {
          proc->addNTotalNeg();
          procLumi->addNTotalNeg();
        }
        [[fallthrough]];
      case kTried:
        proc->addTried(eventWeight);
        procLumi->addTried(eventWeight);
    }
  }

  LHERunInfo::XSec LHERunInfo::xsec() const {
    double sigBrSum = 0.0;
    double errBr2Sum = 0.0;
    int idwtup = heprup.IDWTUP;
    for (std::vector<Process>::const_iterator proc = processes.begin(); proc != processes.end(); ++proc) {
      unsigned int idx = proc->heprupIndex();

      if (!proc->killed().n())
        continue;

      double sigma2Sum, sigma2Err;
      sigma2Sum = heprup.XSECUP[idx] * heprup.XSECUP[idx];
      sigma2Err = heprup.XERRUP[idx] * heprup.XERRUP[idx];

      double sigmaAvg = heprup.XSECUP[idx];

      double fracAcc = 0;
      double ntotal = proc->nTotalPos() - proc->nTotalNeg();
      double npass = proc->nPassPos() - proc->nPassNeg();
      switch (idwtup) {
        case 3:
        case -3:
          fracAcc = ntotal > 0 ? npass / ntotal : -1;
          break;
        default:
          fracAcc = proc->selected().sum() > 0 ? proc->killed().sum() / proc->selected().sum() : -1;
          break;
      }

      if (fracAcc <= 0)
        continue;

      double fracBr = proc->accepted().sum() > 0.0 ? proc->acceptedBr().sum() / proc->accepted().sum() : 1;
      double sigmaFin = sigmaAvg * fracAcc;
      double sigmaFinBr = sigmaFin * fracBr;

      double relErr = 1.0;

      double efferr2 = 0;
      switch (idwtup) {
        case 3:
        case -3: {
          double ntotal_pos = proc->nTotalPos();
          double effp = ntotal_pos > 0 ? (double)proc->nPassPos() / ntotal_pos : 0;
          double effp_err2 = ntotal_pos > 0 ? (1 - effp) * effp / ntotal_pos : 0;

          double ntotal_neg = proc->nTotalNeg();
          double effn = ntotal_neg > 0 ? (double)proc->nPassNeg() / ntotal_neg : 0;
          double effn_err2 = ntotal_neg > 0 ? (1 - effn) * effn / ntotal_neg : 0;

          efferr2 = ntotal > 0
                        ? (ntotal_pos * ntotal_pos * effp_err2 + ntotal_neg * ntotal_neg * effn_err2) / ntotal / ntotal
                        : 0;
          break;
        }
        default: {
          double denominator = pow(proc->selected().sum(), 4);
          double passw = proc->killed().sum();
          double passw2 = proc->killed().sum2();
          double failw = proc->selected().sum() - passw;
          double failw2 = proc->selected().sum2() - passw2;
          double numerator = (passw2 * failw * failw + failw2 * passw * passw);

          efferr2 = denominator > 0 ? numerator / denominator : 0;
          break;
        }
      }
      double delta2Veto = efferr2 / fracAcc / fracAcc;
      double delta2Sum = delta2Veto + sigma2Err / sigma2Sum;
      relErr = (delta2Sum > 0.0 ? std::sqrt(delta2Sum) : 0.0);

      double deltaFinBr = sigmaFinBr * relErr;

      sigBrSum += sigmaFinBr;
      errBr2Sum += deltaFinBr * deltaFinBr;
    }

    XSec result(sigBrSum, std::sqrt(errBr2Sum));

    return result;
  }

  void LHERunInfo::statistics() const {
    double sigSelSum = 0.0;
    double sigSum = 0.0;
    double sigBrSum = 0.0;
    double errSel2Sum = 0.0;
    double errBr2Sum = 0.0;
    double errMatch2Sum = 0.0;
    unsigned long nAccepted = 0;
    unsigned long nTried = 0;
    unsigned long nAccepted_pos = 0;
    unsigned long nTried_pos = 0;
    unsigned long nAccepted_neg = 0;
    unsigned long nTried_neg = 0;
    int idwtup = heprup.IDWTUP;

    LogDebug("LHERunInfo") << " statistics";
    LogDebug("LHERunInfo") << "Process and cross-section statistics";
    LogDebug("LHERunInfo") << "------------------------------------";
    LogDebug("LHERunInfo") << "Process\t\txsec_before [pb]\t\tpassed\tnposw\tnnegw\ttried\tnposw\tnnegw \txsec_match "
                              "[pb]\t\t\taccepted [%]\t event_eff [%]";

    for (std::vector<Process>::const_iterator proc = processes.begin(); proc != processes.end(); ++proc) {
      unsigned int idx = proc->heprupIndex();

      if (!proc->selected().n()) {
        LogDebug("LHERunInfo") << proc->process() << "\t0\t0\tn/a\t\t\tn/a";
        continue;
      }

      double sigma2Sum, sigma2Err;
      sigma2Sum = heprup.XSECUP[idx] * heprup.XSECUP[idx];
      sigma2Err = heprup.XERRUP[idx] * heprup.XERRUP[idx];

      double sigmaAvg = heprup.XSECUP[idx];

      double fracAcc = 0;
      double ntotal = proc->nTotalPos() - proc->nTotalNeg();
      double npass = proc->nPassPos() - proc->nPassNeg();
      switch (idwtup) {
        case 3:
        case -3:
          fracAcc = ntotal > 0 ? npass / ntotal : -1;
          break;
        default:
          fracAcc = proc->selected().sum() > 0 ? proc->killed().sum() / proc->selected().sum() : -1;
          break;
      }

      double fracBr = proc->accepted().sum() > 0.0 ? proc->acceptedBr().sum() / proc->accepted().sum() : 1;
      double sigmaFin = fracAcc > 0 ? sigmaAvg * fracAcc : 0;
      double sigmaFinBr = sigmaFin * fracBr;

      double relErr = 1.0;
      double relAccErr = 1.0;
      double efferr2 = 0;

      if (proc->killed().n() > 0 && fracAcc > 0) {
        switch (idwtup) {
          case 3:
          case -3: {
            double ntotal_pos = proc->nTotalPos();
            double effp = ntotal_pos > 0 ? (double)proc->nPassPos() / ntotal_pos : 0;
            double effp_err2 = ntotal_pos > 0 ? (1 - effp) * effp / ntotal_pos : 0;

            double ntotal_neg = proc->nTotalNeg();
            double effn = ntotal_neg > 0 ? (double)proc->nPassNeg() / ntotal_neg : 0;
            double effn_err2 = ntotal_neg > 0 ? (1 - effn) * effn / ntotal_neg : 0;

            efferr2 = ntotal > 0 ? (ntotal_pos * ntotal_pos * effp_err2 + ntotal_neg * ntotal_neg * effn_err2) /
                                       ntotal / ntotal
                                 : 0;
            break;
          }
          default: {
            double denominator = pow(proc->selected().sum(), 4);
            double passw = proc->killed().sum();
            double passw2 = proc->killed().sum2();
            double failw = proc->selected().sum() - passw;
            double failw2 = proc->selected().sum2() - passw2;
            double numerator = (passw2 * failw * failw + failw2 * passw * passw);

            efferr2 = denominator > 0 ? numerator / denominator : 0;
            break;
          }
        }
        double delta2Veto = efferr2 / fracAcc / fracAcc;
        double delta2Sum = delta2Veto + sigma2Err / sigma2Sum;
        relErr = (delta2Sum > 0.0 ? std::sqrt(delta2Sum) : 0.0);
        relAccErr = (delta2Veto > 0.0 ? std::sqrt(delta2Veto) : 0.0);
      }
      double deltaFinBr = sigmaFinBr * relErr;

      double ntotal_proc = proc->nTotalPos() + proc->nTotalNeg();
      double event_eff_proc = ntotal_proc > 0 ? (double)(proc->nPassPos() + proc->nPassNeg()) / ntotal_proc : -1;
      double event_eff_err_proc = ntotal_proc > 0 ? std::sqrt((1 - event_eff_proc) * event_eff_proc / ntotal_proc) : -1;

      LogDebug("LHERunInfo") << proc->process() << "\t\t" << std::scientific << std::setprecision(3)
                             << heprup.XSECUP[proc->heprupIndex()] << " +/- " << heprup.XERRUP[proc->heprupIndex()]
                             << "\t\t" << proc->accepted().n() << "\t" << proc->nPassPos() << "\t" << proc->nPassNeg()
                             << "\t" << proc->tried().n() << "\t" << proc->nTotalPos() << "\t" << proc->nTotalNeg()
                             << "\t" << std::scientific << std::setprecision(3) << sigmaFinBr << " +/- " << deltaFinBr
                             << "\t\t" << std::fixed << std::setprecision(1) << (fracAcc * 100) << " +/- "
                             << (std::sqrt(efferr2) * 100) << "\t" << std::fixed << std::setprecision(1)
                             << (event_eff_proc * 100) << " +/- " << (event_eff_err_proc * 100);

      nAccepted += proc->accepted().n();
      nTried += proc->tried().n();
      nAccepted_pos += proc->nPassPos();
      nTried_pos += proc->nTotalPos();
      nAccepted_neg += proc->nPassNeg();
      nTried_neg += proc->nTotalNeg();
      sigSelSum += sigmaAvg;
      sigSum += sigmaFin;
      sigBrSum += sigmaFinBr;
      errSel2Sum += sigma2Err;
      errBr2Sum += deltaFinBr * deltaFinBr;
      errMatch2Sum += sigmaFin * relAccErr * sigmaFin * relAccErr;
    }

    double ntotal_all = (nTried_pos + nTried_neg);
    double event_eff_all = ntotal_all > 0 ? (double)(nAccepted_pos + nAccepted_neg) / ntotal_all : -1;
    double event_eff_err_all = ntotal_all > 0 ? std::sqrt((1 - event_eff_all) * event_eff_all / ntotal_all) : -1;

    LogDebug("LHERunInfo") << "Total\t\t" << std::scientific << std::setprecision(3) << sigSelSum << " +/- "
                           << std::sqrt(errSel2Sum) << "\t\t" << nAccepted << "\t" << nAccepted_pos << "\t"
                           << nAccepted_neg << "\t" << nTried << "\t" << nTried_pos << "\t" << nTried_neg << "\t"
                           << std::scientific << std::setprecision(3) << sigBrSum << " +/- " << std::sqrt(errBr2Sum)
                           << "\t\t" << std::fixed << std::setprecision(1) << (sigSum / sigSelSum * 100) << " +/- "
                           << (std::sqrt(errMatch2Sum) / sigSelSum * 100) << "\t" << std::fixed << std::setprecision(1)
                           << (event_eff_all * 100) << " +/- " << (event_eff_err_all * 100);
  }

  LHERunInfo::Header::Header() : xmlDoc(nullptr) {}

  LHERunInfo::Header::Header(const std::string &tag) : LHERunInfoProduct::Header(tag), xmlDoc(nullptr) {}

  LHERunInfo::Header::Header(const Header &orig) : LHERunInfoProduct::Header(orig), xmlDoc(nullptr) {}

  LHERunInfo::Header::Header(const LHERunInfoProduct::Header &orig)
      : LHERunInfoProduct::Header(orig), xmlDoc(nullptr) {}

  LHERunInfo::Header::~Header() {
    if (xmlDoc)
      xmlDoc->release();
  }

  static void fillLines(std::vector<std::string> &lines, const char *data, int len = -1) {
    const char *end = len >= 0 ? (data + len) : nullptr;
    while (*data && (!end || data < end)) {
      std::size_t len = std::strcspn(data, "\r\n");
      if (end && data + len > end)
        len = end - data;
      if (data[len] == '\r' && data[len + 1] == '\n')
        len += 2;
      else if (data[len])
        len++;
      lines.push_back(std::string(data, len));
      data += len;
    }
  }

  static std::vector<std::string> domToLines(const DOMNode *node) {
    std::vector<std::string> result;
    DOMImplementation *impl = DOMImplementationRegistry::getDOMImplementation(XMLUniStr("Core"));
    std::unique_ptr<DOMLSSerializer> writer(((DOMImplementationLS *)(impl))->createLSSerializer());

    std::unique_ptr<DOMLSOutput> outputDesc(((DOMImplementationLS *)impl)->createLSOutput());
    assert(outputDesc.get());
    outputDesc->setEncoding(XMLUniStr("UTF-8"));

    XMLSimpleStr buffer(writer->writeToString(node));

    const char *p = std::strchr((const char *)buffer, '>') + 1;
    const char *q = std::strrchr(p, '<');
    fillLines(result, p, q - p);

    return result;
  }

  std::vector<std::string> LHERunInfo::findHeader(const std::string &tag) const {
    const LHERunInfo::Header *header = nullptr;
    for (std::vector<Header>::const_iterator iter = headers.begin(); iter != headers.end(); ++iter) {
      if (iter->tag() == tag)
        return std::vector<std::string>(iter->begin(), iter->end());
      if (iter->tag() == "header")
        header = &*iter;
    }

    if (!header)
      return std::vector<std::string>();

    const DOMNode *root = header->getXMLNode();
    if (!root)
      return std::vector<std::string>();

    for (const DOMNode *iter = root->getFirstChild(); iter; iter = iter->getNextSibling()) {
      if (iter->getNodeType() != DOMNode::ELEMENT_NODE)
        continue;
      if (tag == (const char *)XMLSimpleStr(iter->getNodeName()))
        return domToLines(iter);
    }

    return std::vector<std::string>();
  }

  namespace {
    class HeaderReader : public CBInputStream::Reader {
    public:
      HeaderReader(const LHERunInfo::Header *header) : header(header), mode(kHeader), iter(header->begin()) {}

      const std::string &data() override {
        switch (mode) {
          case kHeader:
            tmp = "<" + header->tag() + ">";
            mode = kBody;
            break;
          case kBody:
            if (iter != header->end())
              return *iter++;
            tmp = "</" + header->tag() + ">";
            mode = kFooter;
            break;
          case kFooter:
            tmp.clear();
        }

        return tmp;
      }

    private:
      enum Mode { kHeader, kBody, kFooter };

      const LHERunInfo::Header *header;
      Mode mode;
      LHERunInfo::Header::const_iterator iter;
      std::string tmp;
    };
  }  // anonymous namespace

  const DOMNode *LHERunInfo::Header::getXMLNode() const {
    if (tag().empty())
      return nullptr;

    if (!xmlDoc) {
      XercesDOMParser parser;
      parser.setValidationScheme(XercesDOMParser::Val_Auto);
      parser.setDoNamespaces(false);
      parser.setDoSchema(false);
      parser.setValidationSchemaFullChecking(false);

      HandlerBase errHandler;
      parser.setErrorHandler(&errHandler);
      parser.setCreateEntityReferenceNodes(false);

      try {
        std::unique_ptr<CBInputStream::Reader> reader(new HeaderReader(this));
        CBInputSource source(reader);
        parser.parse(source);
        xmlDoc = parser.adoptDocument();
      } catch (const XMLException &e) {
        throw cms::Exception("Generator|LHEInterface")
            << "XML parser reported DOM error no. " << (unsigned long)e.getCode() << ": "
            << XMLSimpleStr(e.getMessage()) << "." << std::endl;
      } catch (const SAXException &e) {
        throw cms::Exception("Generator|LHEInterface")
            << "XML parser reported: " << XMLSimpleStr(e.getMessage()) << "." << std::endl;
      }
    }

    return xmlDoc->getDocumentElement();
  }

  std::pair<int, int> LHERunInfo::pdfSetTranslation() const {
    int pdfA = -1, pdfB = -1;

    if (heprup.PDFGUP.first >= 0) {
      pdfA = heprup.PDFSUP.first;
    }

    if (heprup.PDFGUP.second >= 0) {
      pdfB = heprup.PDFSUP.second;
    }

    return std::make_pair(pdfA, pdfB);
  }

  const bool operator==(const LHERunInfo::Process &lhs, const LHERunInfo::Process &rhs) {
    return (lhs.process() == rhs.process());
  }

  const bool operator<(const LHERunInfo::Process &lhs, const LHERunInfo::Process &rhs) {
    return (lhs.process() < rhs.process());
  }

}  // namespace lhef
