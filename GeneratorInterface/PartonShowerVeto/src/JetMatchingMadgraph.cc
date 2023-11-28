#include <functional>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <cstring>
#include <string>
#include <cctype>
#include <map>
#include <set>

#include <boost/algorithm/string/trim.hpp>

// #include <HepMC/GenEvent.h>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "GeneratorInterface/LHEInterface/interface/LHERunInfo.h"
#include "GeneratorInterface/LHEInterface/interface/LHEEvent.h"
#include "GeneratorInterface/PartonShowerVeto/interface/JetMatchingMadgraph.h"

namespace gen {

  extern "C" {
#define PARAMLEN 20
  namespace {
    struct Param {
      Param(const std::string &str) {
        int len = std::min(PARAMLEN, (int)str.length());
        std::memcpy(value, str.c_str(), len);
        std::memset(value + len, ' ', PARAMLEN - len);
      }

      char value[PARAMLEN];
    };
  }  // namespace
  extern void mginit_(int *npara, Param *params, Param *values);
  extern void mgevnt_(void);
  extern void mgveto_(int *veto);

  extern struct UPPRIV {
    int lnhin, lnhout;
    int mscal, ievnt;
    int ickkw, iscale;
  } uppriv_;

  extern struct MEMAIN {
    double etcjet, rclmax, etaclmax, qcut, showerkt, clfact;
    int maxjets, minjets, iexcfile, ktsche;
    int mektsc, nexcres, excres[30];
    int nqmatch, excproc, iexcproc[1000], iexcval[1000];
    bool nosingrad, jetprocs;
  } memain_;

  extern struct OUTTREE {
    int flag;
  } outtree_;

  extern struct MEMAEV {
    double ptclus[20];
    int nljets, iexc, ifile;
  } memaev_;

  extern struct PYPART {
    int npart, npartd, ipart[1000];
    double ptpart[1000];
  } pypart_;
  }  // extern "C"

  template <typename T>
  T JetMatchingMadgraph::parseParameter(const std::string &value) {
    std::istringstream ss(value);
    T result;
    ss >> result;
    return result;
  }

  template <>
  std::string JetMatchingMadgraph::parseParameter(const std::string &value) {
    std::string result;
    if (!result.empty() && result[0] == '\'')
      result = result.substr(1);
    if (!result.empty() && result[result.length() - 1] == '\'')
      result.resize(result.length() - 1);
    return result;
  }

  template <>
  bool JetMatchingMadgraph::parseParameter(const std::string &value_) {
    std::string value(value_);
    std::transform(value.begin(), value.end(), value.begin(), (int (*)(int))std::toupper);
    return value == "T" || value == "Y" || value == "True" || value == "1" || value == ".TRUE.";
  }

  template <typename T>
  T JetMatchingMadgraph::getParameter(const std::map<std::string, std::string> &params,
                                      const std::string &var,
                                      const T &defValue) {
    std::map<std::string, std::string>::const_iterator pos = params.find(var);
    if (pos == params.end())
      return defValue;
    return parseParameter<T>(pos->second);
  }

  template <typename T>
  T JetMatchingMadgraph::getParameter(const std::string &var, const T &defValue) const {
    return getParameter(mgParams, var, defValue);
  }

  JetMatchingMadgraph::JetMatchingMadgraph(const edm::ParameterSet &params)
      : JetMatching(params), runInitialized(false) {
    std::string mode = params.getParameter<std::string>("mode");
    if (mode == "inclusive") {
      soup = false;
      exclusive = false;
    } else if (mode == "exclusive") {
      soup = false;
      exclusive = true;
    } else if (mode == "auto")
      soup = true;
    else
      throw cms::Exception("Generator|LHEInterface") << "Madgraph jet matching scheme requires \"mode\" "
                                                        "parameter to be set to either \"inclusive\", "
                                                        "\"exclusive\" or \"auto\"."
                                                     << std::endl;

    memain_.etcjet = 0.;
    memain_.rclmax = 0.0;
    memain_.clfact = 0.0;
    memain_.ktsche = 0.0;
    memain_.etaclmax = params.getParameter<double>("MEMAIN_etaclmax");
    memain_.qcut = params.getParameter<double>("MEMAIN_qcut");
    memain_.minjets = params.getParameter<int>("MEMAIN_minjets");
    memain_.maxjets = params.getParameter<int>("MEMAIN_maxjets");
    memain_.showerkt = params.getParameter<double>("MEMAIN_showerkt");
    memain_.nqmatch = params.getParameter<int>("MEMAIN_nqmatch");
    outtree_.flag = params.getParameter<int>("outTree_flag");
    std::string list_excres = params.getParameter<std::string>("MEMAIN_excres");
    std::vector<std::string> elems;
    std::stringstream ss(list_excres);
    std::string item;
    int index = 0;
    while (std::getline(ss, item, ',')) {
      elems.push_back(item);
      memain_.excres[index] = std::atoi(item.c_str());
      index++;
    }
    memain_.nexcres = index;
  }

  JetMatchingMadgraph::~JetMatchingMadgraph() {}

  double JetMatchingMadgraph::getJetEtaMax() const { return memain_.etaclmax; }

  std::set<std::string> JetMatchingMadgraph::capabilities() const {
    std::set<std::string> result;
    result.insert("psFinalState");
    result.insert("hepevt");
    result.insert("pythia6");
    return result;
  }

  static std::map<std::string, std::string> parseHeader(const std::vector<std::string> &header) {
    std::map<std::string, std::string> params;

    for (std::vector<std::string>::const_iterator iter = header.begin(); iter != header.end(); ++iter) {
      std::string line = *iter;
      if (line.empty() || line[0] == '#')
        continue;

      std::string::size_type pos = line.find('!');
      if (pos != std::string::npos)
        line.resize(pos);

      pos = line.find('=');
      if (pos == std::string::npos)
        continue;

      std::string var = boost::algorithm::trim_copy(line.substr(pos + 1));
      std::string value = boost::algorithm::trim_copy(line.substr(0, pos));

      params[var] = value;
    }

    return params;
  }

  template <typename T>
  void JetMatchingMadgraph::updateOrDie(const std::map<std::string, std::string> &params,
                                        T &param,
                                        const std::string &name) {
    if (param < 0) {
      param = getParameter(params, name, param);
    }
    if (param < 0)
      throw cms::Exception("Generator|PartonShowerVeto") << "The MGParamCMS header does not specify the jet "
                                                            "matching parameter \""
                                                         << name
                                                         << "\", but it "
                                                            "is requested by the CMSSW configuration."
                                                         << std::endl;
  }

  // implements the Madgraph method - use ME2pythia.f

  void JetMatchingMadgraph::init(const lhef::LHERunInfo *runInfo) {
    // read MadGraph run card

    std::map<std::string, std::string> parameters;

    std::vector<std::string> header = runInfo->findHeader("MGRunCard");
    if (header.empty())
      throw cms::Exception("Generator|PartonShowerVeto") << "In order to use MadGraph jet matching, "
                                                            "the input file has to contain the corresponding "
                                                            "MadGraph headers."
                                                         << std::endl;

    mgParams = parseHeader(header);

    // set variables in common block

    std::vector<Param> params;
    std::vector<Param> values;
    for (std::map<std::string, std::string>::const_iterator iter = mgParams.begin(); iter != mgParams.end(); ++iter) {
      params.push_back(" " + iter->first);
      values.push_back(iter->second);
    }

    // set MG matching parameters

    uppriv_.ickkw = getParameter<int>("ickkw", 0);
    memain_.mektsc = getParameter<int>("ktscheme", 0);

    header = runInfo->findHeader("MGParamCMS");

    std::map<std::string, std::string> mgInfoCMS = parseHeader(header);

    for (std::map<std::string, std::string>::const_iterator iter = mgInfoCMS.begin(); iter != mgInfoCMS.end(); ++iter) {
      std::cout << "mgInfoCMS: " << iter->first << " " << iter->second << std::endl;
    }

    updateOrDie(mgInfoCMS, memain_.etaclmax, "etaclmax");
    updateOrDie(mgInfoCMS, memain_.qcut, "qcut");
    updateOrDie(mgInfoCMS, memain_.minjets, "minjets");
    updateOrDie(mgInfoCMS, memain_.maxjets, "maxjets");
    updateOrDie(mgInfoCMS, memain_.showerkt, "showerkt");
    updateOrDie(mgInfoCMS, memain_.nqmatch, "nqmatch");

    // run Fortran initialization code

    int nparam = params.size();
    mginit_(&nparam, &params.front(), &values.front());
    runInitialized = true;
  }

  //void JetMatchingMadgraph::beforeHadronisation(
  //				const std::shared_ptr<lhef::LHEEvent> &event)

  void JetMatchingMadgraph::beforeHadronisation(const lhef::LHEEvent *event) {
    if (!runInitialized)
      throw cms::Exception("Generator|PartonShowerVeto") << "Run not initialized in JetMatchingMadgraph" << std::endl;

    if (uppriv_.ickkw) {
      std::vector<std::string> comments = event->getComments();
      if (comments.size() == 1) {
        std::istringstream ss(comments[0].substr(1));
        for (int i = 0; i < 1000; i++) {
          double pt;
          ss >> pt;
          if (!ss.good())
            break;
          pypart_.ptpart[i] = pt;
        }
      } else {
        edm::LogWarning("Generator|LHEInterface") << "Expected exactly one comment line per "
                                                     "event containing MadGraph parton scale "
                                                     "information."
                                                  << std::endl;

        const lhef::HEPEUP *hepeup = event->getHEPEUP();
        for (int i = 2; i < hepeup->NUP; i++) {
          double mt2 = hepeup->PUP[i][0] * hepeup->PUP[i][0] + hepeup->PUP[i][1] * hepeup->PUP[i][1] +
                       hepeup->PUP[i][4] * hepeup->PUP[i][4];
          pypart_.ptpart[i - 2] = std::sqrt(mt2);
        }
      }
    }

    // mgevnt_();
    eventInitialized = true;
  }

  void JetMatchingMadgraph::beforeHadronisationExec() {
    mgevnt_();
    eventInitialized = true;
    return;
  }

  /*
int JetMatchingMadgraph::match(const HepMC::GenEvent *partonLevel,
                                  const HepMC::GenEvent *finalState,
                                  bool showeredFinalState)
*/
  int JetMatchingMadgraph::match(const lhef::LHEEvent *partonLevel, const std::vector<fastjet::PseudoJet> *jetInput) {
    /*
	if (!showeredFinalState)
		throw cms::Exception("Generator|LHEInterface")
			<< "MadGraph matching expected parton shower "
			   "final state." << std::endl;
*/

    if (!runInitialized)
      throw cms::Exception("Generator|LHEInterface") << "Run not initialized in JetMatchingMadgraph" << std::endl;

    if (!eventInitialized)
      throw cms::Exception("Generator|LHEInterface") << "Event not initialized in JetMatchingMadgraph" << std::endl;

    if (soup)
      memaev_.iexc = (memaev_.nljets < memain_.maxjets);
    else
      memaev_.iexc = exclusive;

    int veto = 0;
    mgveto_(&veto);
    fMatchingStatus = true;
    eventInitialized = false;

    return veto;
  }

}  // end namespace gen
