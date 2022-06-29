#ifndef RecoLuminosity_LumiProducer_LumiCalculator_h
#define RecoLuminosity_LumiProducer_LumiCalculator_h
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Luminosity/interface/LumiSummaryRunHeader.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <boost/regex.hpp>
#include <iostream>
#include <map>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
struct hltPerPathInfo {
  hltPerPathInfo() : prescale(0) {}
  unsigned int prescale;
};
struct l1PerBitInfo {
  l1PerBitInfo() : prescale(0) {}
  unsigned int prescale;
};
struct MyPerLumiInfo {
  unsigned int lsnum;
  float livefraction;
  float intglumi;
  unsigned long long deadcount;
};

class LumiCalculator : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit LumiCalculator(edm::ParameterSet const& pset);
  ~LumiCalculator() override;

private:
  void beginJob() override;
  void beginRun(const edm::Run& run, const edm::EventSetup& c) override;
  void analyze(edm::Event const& e, edm::EventSetup const& c) override;
  void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override {}
  void endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void endJob() override;
  std::vector<std::string> splitpathstr(const std::string& strValue, const std::string separator);
  HLTConfigProvider hltConfig_;
  std::multimap<std::string, std::string> trgpathMmap_;  //key:hltpath,value:l1bit
  std::map<std::string, hltPerPathInfo> hltmap_;
  std::map<std::string, l1PerBitInfo> l1map_;  //
  std::vector<MyPerLumiInfo> perrunlumiinfo_;

private:
  edm::LogInfo* log_;
  bool showTrgInfo_;
  unsigned int currentlumi_;
};  //end class

// -----------------------------------------------------------------

LumiCalculator::LumiCalculator(edm::ParameterSet const& pset) : log_(new edm::LogInfo("LumiReport")), currentlumi_(0) {
  showTrgInfo_ = pset.getUntrackedParameter<bool>("showTriggerInfo", false);
  consumes<LumiSummary, edm::InLumi>(edm::InputTag("lumiProducer", ""));
  consumes<LumiSummaryRunHeader, edm::InRun>(edm::InputTag("lumiProducer", ""));
}

// -----------------------------------------------------------------

LumiCalculator::~LumiCalculator() {
  delete log_;
  log_ = nullptr;
}

// -----------------------------------------------------------------

void LumiCalculator::analyze(edm::Event const& e, edm::EventSetup const&) {}

// -----------------------------------------------------------------

void LumiCalculator::beginJob() {}

// -----------------------------------------------------------------

void LumiCalculator::beginRun(const edm::Run& run, const edm::EventSetup& c) {
  //std::cout<<"I'm in run number "<<run.run()<<std::endl;
  //if(!hltConfig_.init("HLT")){
  // throw cms::Exception("HLT process cannot be initialized");
  //}
  bool changed(true);
  const std::string processname("HLT");
  if (!hltConfig_.init(run, c, processname, changed)) {
    throw cms::Exception("HLT process cannot be initialized");
  }
  perrunlumiinfo_.clear();
  trgpathMmap_.clear();
  hltmap_.clear();
  l1map_.clear();
  //hltConfig_.dump("processName");
  //hltConfig_.dump("TableName");
  //hltConfig_.dump("Triggers");
  //hltConfig_.dump("Modules");
  if (showTrgInfo_) {
    *log_ << "======Trigger Configuration Overview======\n";
    *log_ << "Run " << run.run() << " Trigger Table : " << hltConfig_.tableName() << "\n";
  }
  unsigned int totaltrg = hltConfig_.size();
  for (unsigned int t = 0; t < totaltrg; ++t) {
    std::string hltname(hltConfig_.triggerName(t));
    std::vector<std::string> numpathmodules = hltConfig_.moduleLabels(hltname);
    if (showTrgInfo_) {
      *log_ << t << " HLT path\t" << hltname << "\n";
    }
    hltPerPathInfo hlt;
    hlt.prescale = 1;
    hltmap_.insert(std::make_pair(hltname, hlt));
    std::vector<std::string>::iterator hltpathBeg = numpathmodules.begin();
    std::vector<std::string>::iterator hltpathEnd = numpathmodules.end();
    unsigned int mycounter = 0;
    for (std::vector<std::string>::iterator numpathmodule = hltpathBeg; numpathmodule != hltpathEnd; ++numpathmodule) {
      if (hltConfig_.moduleType(*numpathmodule) != "HLTLevel1GTSeed") {
        continue;
      }
      ++mycounter;

      edm::ParameterSet l1GTPSet = hltConfig_.modulePSet(*numpathmodule);
      std::string l1pathname = l1GTPSet.getParameter<std::string>("L1SeedsLogicalExpression");
      //*log_<<"numpathmodule "<< *numpathmodule <<" : l1pathname "<<l1pathname<<"\n";
      if (mycounter > 1) {
        if (showTrgInfo_) {
          *log_ << "\tskip and erase previous seeds : multiple L1SeedsLogicalExpressions per hlt path\n";
        }
        //erase all previously calculated seeds for this path
        trgpathMmap_.erase(hltname);
        continue;
      }
      if (l1pathname.find('(') != std::string::npos) {
        if (showTrgInfo_) {
          *log_ << "  L1SeedsLogicalExpression(Complex)\t" << l1pathname << "\n";
          *log_ << "\tskip:contain complex logic\n";
        }
        continue;
      } else if (l1pathname.find("OR") != std::string::npos) {
        if (showTrgInfo_) {
          *log_ << "  L1SeedsLogicalExpression(ORed)\t" << l1pathname << "\n";
        }
        std::vector<std::string> seeds = splitpathstr(l1pathname, " OR ");
        if (seeds.size() > 2) {
          if (showTrgInfo_) {
            *log_ << "\tskip:contain >1 OR\n";
          }
          continue;
        } else {
          for (std::vector<std::string>::iterator i = seeds.begin(); i != seeds.end(); ++i) {
            if (!i->empty() && showTrgInfo_)
              *log_ << "\t\tseed: " << *i << "\n";
            if (i == seeds.begin()) {  //for now we take the first one from OR
              trgpathMmap_.insert(std::make_pair(hltname, *i));
            }
          }
        }
      } else if (l1pathname.find("AND") != std::string::npos) {
        if (showTrgInfo_) {
          *log_ << "  L1SeedsLogicalExpression(ANDed)\t" << l1pathname << "\n";
        }
        std::vector<std::string> seeds = splitpathstr(l1pathname, " AND ");
        if (seeds.size() > 2) {
          if (showTrgInfo_) {
            *log_ << "\tskip:contain >1 AND\n";
          }
          continue;
        } else {
          for (std::vector<std::string>::iterator i = seeds.begin(); i != seeds.end(); ++i) {
            if (!i->empty() && showTrgInfo_)
              *log_ << "\t\tseed: " << *i << "\n";
            if (i == seeds.begin()) {  //for now we take the first one
              trgpathMmap_.insert(std::make_pair(hltname, *i));
            }
          }
        }
      } else {
        if (showTrgInfo_) {
          *log_ << "  L1SeedsLogicalExpression(ONE)\t" << l1pathname << "\n";
        }
        if (splitpathstr(l1pathname, " NOT ").size() > 1) {
          if (showTrgInfo_) {
            *log_ << "\tskip:contain NOT\n";
          }
          continue;
        }
        trgpathMmap_.insert(std::make_pair(hltname, l1pathname));
      }
    }
  }
  if (showTrgInfo_) {
    *log_ << "================\n";
  }
}

// -----------------------------------------------------------------
void LumiCalculator::endLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c) {
  /**Integrated Luminosity per Lumi Section
     instantaneousLumi*93.244(sec)
  **/
  //std::cout<<"I'm in lumi block "<<lumiBlock.id()<<std::endl;

  edm::Handle<LumiSummary> lumiSummary;
  lumiBlock.getByLabel("lumiProducer", lumiSummary);

  edm::Handle<LumiSummaryRunHeader> lumiSummaryRH;
  lumiBlock.getRun().getByLabel("lumiProducer", lumiSummaryRH);

  MyPerLumiInfo l;
  l.lsnum = lumiBlock.id().luminosityBlock();

  //
  //collect lumi.
  //
  l.deadcount = lumiSummary->deadcount();
  l.intglumi = lumiSummary->avgInsDelLumi() * 93.244;
  l.livefraction = lumiSummary->liveFrac();

  *log_ << "====== Lumi Section " << lumiBlock.id().luminosityBlock() << " ======\n";
  *log_ << "\t Luminosity " << l.intglumi << "\n";
  *log_ << "\t Dead count " << l.deadcount << "\n";
  *log_ << "\t Deadtime corrected Luminosity " << l.intglumi * l.livefraction << "\n";

  //
  //print correlated hlt-l1 info, only if you ask
  //
  if (showTrgInfo_) {
    std::map<std::string, hltPerPathInfo>::iterator hltit;
    std::map<std::string, hltPerPathInfo>::iterator hltitBeg = hltmap_.begin();
    std::map<std::string, hltPerPathInfo>::iterator hltitEnd = hltmap_.end();

    typedef std::pair<std::multimap<std::string, std::string>::iterator,
                      std::multimap<std::string, std::string>::iterator>
        TRGMAPIT;
    unsigned int c = 0;
    for (hltit = hltitBeg; hltit != hltitEnd; ++hltit) {
      std::string hltname = hltit->first;
      *log_ << c << " HLT path  " << hltname << " , prescale : " << hltit->second.prescale << "\n";
      TRGMAPIT ppp;
      ppp = trgpathMmap_.equal_range(hltname);
      if (ppp.first == ppp.second) {
        *log_ << "    no L1\n";
      }
      for (std::multimap<std::string, std::string>::iterator mit = ppp.first; mit != ppp.second; ++mit) {
        std::string l1name = mit->second;
        *log_ << "    L1 name : " << l1name;
        LumiSummary::L1 l1result = lumiSummary->l1info(lumiSummaryRH->getL1Index(l1name));
        *log_ << " prescale : " << l1result.prescale << "\n";
        *log_ << "\n";
      }
      ++c;
    }
  }
  //
  //accumulate hlt counts. Absent for now
  //
  /**
     for(hltit=hltitBeg;hltit!=hltitEnd;++hltit){
     }
  **/

  //
  //accumulate l1 counts
  //
  size_t n = lumiSummary->nTriggerLine();
  for (size_t i = 0; i < n; ++i) {
    std::string l1bitname = lumiSummaryRH->getL1Name(lumiSummary->l1info(i).triggernameidx);
    l1PerBitInfo t;
    if (currentlumi_ == 0) {
      t.prescale = lumiSummary->l1info(i).prescale;
      l1map_.insert(std::make_pair(l1bitname, t));
    }
  }

  perrunlumiinfo_.push_back(l);

  ++currentlumi_;
}

// -----------------------------------------------------------------
void LumiCalculator::endRun(edm::Run const& run, edm::EventSetup const& c) {
  /**Notes on calculation:
     
  1. CMS recorded Luminosity per run :
     sum over HLX&&HF certified LS 
     lumiSummary->avgInsDelLumi()*93.244*livefraction()

     2. Effective Luminosity per run per trigger line:
     For the moment, we take only the first L1 seed in case of 'OR' or 'AND'
     relationship between HLT and L1 seeds
     
     avgInsDelLumi()*93.244*livefraction()/(HLTprescale*L1prescale)
     for now HLTprescale=1
     
     3. LHC delivered:
     there is no point in calculating delivered when data do not contain all LS
  **/
  //std::cout<<"valid trigger lines "<<trgpathMmap_.size()<<std::endl;
  //std::cout<<"total lumi lines "<<perrunlumiinfo_.size()<<std::endl;
  std::vector<MyPerLumiInfo>::const_iterator lumiIt;
  std::vector<MyPerLumiInfo>::const_iterator lumiItBeg = perrunlumiinfo_.begin();
  std::vector<MyPerLumiInfo>::const_iterator lumiItEnd = perrunlumiinfo_.end();
  float recorded = 0.0;

  *log_ << "================ Run Summary " << run.run() << "================\n";
  for (lumiIt = lumiItBeg; lumiIt != lumiItEnd; ++lumiIt) {  //loop over LS
    recorded += lumiIt->intglumi * lumiIt->livefraction;
  }
  *log_ << "  CMS Recorded Lumi (e+27cm^-2) : " << recorded << "\n";
  *log_ << "  Effective Lumi (e+27cm^-2) per trigger path: "
        << "\n\n";
  std::multimap<std::string, std::string>::iterator it;
  std::multimap<std::string, std::string>::iterator itBeg = trgpathMmap_.begin();
  std::multimap<std::string, std::string>::iterator itEnd = trgpathMmap_.end();
  unsigned int cc = 0;
  for (it = itBeg; it != itEnd; ++it) {
    *log_ << "  " << cc << "  " << it->first << " - " << it->second << " : ";
    ++cc;
    std::map<std::string, hltPerPathInfo>::const_iterator hltIt = hltmap_.find(it->first);
    if (hltIt == hltmap_.end()) {
      std::cout << "HLT path " << it->first << " not found" << std::endl;
      *log_ << "\n";
      continue;
    }
    std::map<std::string, l1PerBitInfo>::const_iterator l1It = l1map_.find(it->second);
    if (l1It == l1map_.end()) {
      std::cout << "L1 bit " << it->second << " not found" << std::endl;
      *log_ << "\n";
      continue;
    }
    unsigned int hltprescale = hltIt->second.prescale;
    unsigned int l1prescale = l1It->second.prescale;
    if (hltprescale != 0 && l1prescale != 0) {
      float effectiveLumi = recorded / (hltprescale * l1prescale);
      *log_ << effectiveLumi << "\n";
    } else {
      *log_ << "0 prescale exception\n";
      continue;
    }
    *log_ << "\n";
  }
}

// -----------------------------------------------------------------
void LumiCalculator::endJob() {}

std::vector<std::string> LumiCalculator::splitpathstr(const std::string& strValue, const std::string separator) {
  std::vector<std::string> vecstrResult;
  boost::regex re(separator);
  boost::sregex_token_iterator p(strValue.begin(), strValue.end(), re, -1);
  boost::sregex_token_iterator end;
  while (p != end) {
    vecstrResult.push_back(*p++);
  }
  return vecstrResult;
}

DEFINE_FWK_MODULE(LumiCalculator);
#endif
