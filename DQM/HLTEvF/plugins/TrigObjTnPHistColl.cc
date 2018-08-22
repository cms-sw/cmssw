#include "TrigObjTnPHistColl.h"

#include "FWCore/Common/interface/TriggerNames.h"

namespace{
  std::vector<float> convertToFloat(const std::vector<double>& vecD){
    return std::vector<float>(vecD.begin(),vecD.end());
  }
}

TrigObjTnPHistColl::TrigObjTnPHistColl(const edm::ParameterSet& config):
  tagCuts_(config.getParameter<std::vector<edm::ParameterSet>>("tagCuts")),
  probeCuts_(config.getParameter<std::vector<edm::ParameterSet>>("probeCuts")),
  tagFilters_(config.getParameter<edm::ParameterSet>("tagFilters")),
  collName_(config.getParameter<std::string>("collName")),
  folderName_(config.getParameter<std::string>("folderName")),
  histDefs_(config.getParameter<edm::ParameterSet>("histDefs")),
  evtTrigSel_(config.getParameter<edm::ParameterSet>("evtTrigSel"))
  
{
  auto probeFilters = config.getParameter<std::vector<std::string> >("probeFilters");
  for(auto& probeFilter : probeFilters){
    probeHists_.emplace_back(ProbeData(std::move(probeFilter)));
  }
}

edm::ParameterSetDescription TrigObjTnPHistColl::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.addVPSet("tagCuts",VarRangeCut<trigger::TriggerObject>::makePSetDescription(),std::vector<edm::ParameterSet>());
  desc.addVPSet("probeCuts",VarRangeCut<trigger::TriggerObject>::makePSetDescription(),std::vector<edm::ParameterSet>());
  desc.add<edm::ParameterSetDescription>("tagFilters",FilterSelector::makePSetDescription());
  desc.add<std::string>("collName","stdTag");
  desc.add<std::string>("folderName","HLT/EGM/TrigObjTnP");
  desc.add<edm::ParameterSetDescription>("histDefs",HistDefs::makePSetDescription());
  desc.add<std::vector<std::string>>("probeFilters",std::vector<std::string>());  
  desc.add<edm::ParameterSetDescription>("evtTrigSel",PathSelector::makePSetDescription());  
  return desc;
}

void TrigObjTnPHistColl::bookHists(DQMStore::ConcurrentBooker& iBooker)
{
  iBooker.setCurrentFolder(folderName_);
  for(auto& probe : probeHists_){
    probe.bookHists(collName_,iBooker,histDefs_);
  }
}

void TrigObjTnPHistColl::fill(const trigger::TriggerEvent& trigEvt,
			      const edm::TriggerResults& trigResults,
			      const edm::TriggerNames& trigNames)const
{
  if(evtTrigSel_(trigResults,trigNames)==false) return;

  auto tagTrigKeys = tagFilters_.getPassingKeys(trigEvt);
  for(auto& tagKey : tagTrigKeys){
    const trigger::TriggerObject& tagObj = trigEvt.getObjects()[tagKey];
    if(tagCuts_(tagObj)){
      for(auto& probeColl : probeHists_) probeColl.fill(tagKey,trigEvt,probeCuts_);
    }
  }
}

//trigger::Keys is likely a vector containing 0-3 short ints (probably 10 max),
// passing by value makes this much  easier code wise (otherwise would have to 
//create a dummy empty vector) and shouldnt be too much of a performance hit
const trigger::Keys TrigObjTnPHistColl::getKeys(const trigger::TriggerEvent& trigEvt,const std::string& filterName)
{
  edm::InputTag filterTag(filterName,"",trigEvt.usedProcessName());
  trigger::size_type filterIndex = trigEvt.filterIndex(filterTag); 
  if(filterIndex<trigEvt.sizeFilters()) return trigEvt.filterKeys(filterIndex);
  else return trigger::Keys();
}

TrigObjTnPHistColl::FilterSelector::FilterSelector(const edm::ParameterSet& config):
  isAND_(config.getParameter<bool>("isAND"))
{
  auto filterSetConfigs = config.getParameter<std::vector<edm::ParameterSet>>("filterSets");
  for(auto& filterSetConfig : filterSetConfigs) filterSets_.emplace_back(FilterSet(filterSetConfig));
}

edm::ParameterSetDescription TrigObjTnPHistColl::FilterSelector::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.addVPSet("filterSets",FilterSet::makePSetDescription(),std::vector<edm::ParameterSet>());
  desc.add<bool>("isAND",false);
  return desc;
}

const trigger::Keys TrigObjTnPHistColl::FilterSelector::getPassingKeys(const trigger::TriggerEvent& trigEvt)const
{
  trigger::Keys passingKeys;
  bool isFirstFilterSet = true;
  for(const auto& filterSet : filterSets_){
    auto keysOfFilterSet = filterSet.getPassingKeys(trigEvt);
    if(isFirstFilterSet) passingKeys = keysOfFilterSet;
    else mergeTrigKeys(passingKeys,keysOfFilterSet,isAND_);
    isFirstFilterSet = false;
  }
  cleanTrigKeys(passingKeys);
  return passingKeys;
}

void TrigObjTnPHistColl::FilterSelector::mergeTrigKeys(trigger::Keys& keys,const trigger::Keys& keysToMerge,bool isAND)
{
  if(isAND){
    for(auto& key : keys) {
      if(std::count(keysToMerge.begin(),keysToMerge.end(),key)==0){
	key=std::numeric_limits<trigger::size_type>::max();
      }
    }
  }else{
    for(const auto key : keysToMerge){
      keys.push_back(key);
    }
  }  
}

void TrigObjTnPHistColl::FilterSelector::cleanTrigKeys(trigger::Keys& keys)
{
  std::sort(keys.begin(),keys.end());
  std::unique(keys.begin(),keys.end());
  while(!keys.empty() && keys.back()==std::numeric_limits<trigger::size_type>::max()){
    keys.pop_back();
  }
}

TrigObjTnPHistColl::FilterSelector::FilterSet::FilterSet(const edm::ParameterSet& config):
  filters_(config.getParameter<std::vector<std::string>>("filters")),
  isAND_(config.getParameter<bool>("isAND"))
{
  
}

TrigObjTnPHistColl::PathSelector::PathSelector(const edm::ParameterSet& config):
  selectionStr_(config.getParameter<std::string>("selectionStr")),
  isANDForExpandedPaths_(config.getParameter<bool>("isANDForExpandedPaths")),
  verbose_(config.getParameter<int>("verbose")),
  isInited_(false)
{

}

edm::ParameterSetDescription TrigObjTnPHistColl::PathSelector::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.add<std::string>("selectionStr",std::string(""));
  desc.add<bool>("isANDForExpandedPaths",false);
  desc.add<int>("verbose",1);
  return desc;
}

void TrigObjTnPHistColl::PathSelector::init(const HLTConfigProvider& hltConfig)
{
  expandedSelStr_ = expandSelectionStr(selectionStr_,hltConfig,isANDForExpandedPaths_,verbose_);
  isInited_ = true;
  if(verbose_>1){
    edm::LogInfo("TrigObjTnPHistColl::PathSelector" ) << "trigger selection string: \"" << expandedSelStr_ << "\"";
  }
}

bool TrigObjTnPHistColl::PathSelector::operator()(const edm::TriggerResults& trigResults,const edm::TriggerNames& trigNames)const
{
  if(selectionStr_.empty()) return true; //didnt specify any selection, default to pass
  else if(!isInited_){
    edm::LogError("TrigObjTnPHistColl") <<" error, TrigObjTnPHistColl::PathSelector is not initalised, returning false ";
    return false; 
  }else if(expandedSelStr_.empty()){
    //there was a problem parsing the expression, it was logged at the start, no need to do each run
    return false;
  }else{
    //as of 20/08/18, there is a bug in L1GtLogicParser, it must take a non-const std::string
    //as input because it overloads the constructor between const and non-const std::string
    //for like no reason. And the const version is broken, you have to use non-const
    //hence we make a non-const copy of the selection string
    std::string selStr = expandedSelStr_;
    L1GtLogicParser logicParser(selStr);
    for(auto& token : logicParser.operandTokenVector()){
      const std::string&  pathName = token.tokenName;
      auto pathIndex = trigNames.triggerIndex(pathName);
      bool accept = pathIndex < trigNames.size() ? trigResults.accept(pathIndex) : false;
      token.tokenResult = accept;      
    } 
    return logicParser.expressionResult();
  }
} 

//a port of https://github.com/cms-sw/cmssw/blob/51eb73f59e2016d54618e2a8e19abab84fe33b47/CommonTools/TriggerUtils/src/GenericTriggerEventFlag.cc#L225-L238
std::string TrigObjTnPHistColl::PathSelector::expandSelectionStr(const std::string& selStr, const HLTConfigProvider& hltConfig, bool isAND,int verbose)
{
  std::string expandedSelStr(selStr);
  //it is very important to pass in as a non-const std::string, see comments else where
  L1GtLogicParser logicParser(expandedSelStr);
  for(const auto& token : logicParser.operandTokenVector()){
    const std::string&  pathName = token.tokenName;
    if ( pathName.find('*') != std::string::npos ) {
      std::string pathPatternExpanded =  expandPath(pathName, hltConfig, isAND, verbose);
      expandedSelStr.replace( expandedSelStr.find( pathName ), pathName.size(), pathPatternExpanded);
    }
  }
  return expandedSelStr;
}

//a port of GenericTriggerEventFlag::expandLogicalExpression 
//https://github.com/cms-sw/cmssw/blob/51eb73f59e2016d54618e2a8e19abab84fe33b47/CommonTools/TriggerUtils/src/GenericTriggerEventFlag.cc#L600-L632
std::string TrigObjTnPHistColl::PathSelector::expandPath(const std::string& pathPattern, const HLTConfigProvider& hltConfig, bool isAND,int verbose)
{
   // Find matching entries in the menu
  const std::vector<std::string>& trigNames = hltConfig.triggerNames();
  std::vector<std::string> matched;
  const std::string versionWildcard("_v*");
  if(pathPattern.substr(pathPattern.size() - versionWildcard.size()) == versionWildcard) {
    const std::string pathPatternBase(pathPattern.substr( 0, pathPattern.size() - versionWildcard.size()));
    matched = hltConfig.restoreVersion(trigNames, pathPatternBase);
  } else {
    matched = hltConfig.matched(trigNames, pathPattern);
  }
  
  if( matched.empty() ) {
    if(verbose>=1 ) edm::LogWarning("TrigObjTnPHistColl::PathSelector") << "pattern: \"" << pathPattern << "\" could not be resolved, please check your triggers are spelt correctly and present in the data you are running over";
    return "";
  }
  
  // Compose logical expression
  std::string expanded( "(" );
   for( unsigned iVers = 0; iVers < matched.size(); ++iVers ) {
     if( iVers > 0 ) expanded.append( isAND ? " AND " : " OR " );
     expanded.append( matched.at( iVers ) );
   }
   expanded.append( ")" );
   if(verbose>1 ) {
     edm::LogInfo("TrigObjTnPHistColl::PathSelector" ) << "Logical expression : \"" << pathPattern     << "\"\n"
						       << "        expanded to:  \"" << expanded << "\"";
   }
   return expanded;
}



edm::ParameterSetDescription TrigObjTnPHistColl::FilterSelector::FilterSet::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string> >("filters",std::vector<std::string>());
  desc.add<bool>("isAND",true);
  return desc;
}

const trigger::Keys TrigObjTnPHistColl::FilterSelector::FilterSet::getPassingKeys(const trigger::TriggerEvent& trigEvt)const
{
  trigger::Keys passingKeys;
  bool firstFilter = true;
  for(const auto& filterName : filters_){
    const trigger::Keys& trigKeys = getKeys(trigEvt,filterName);
    if(firstFilter) {
      passingKeys = trigKeys;
      firstFilter = false;
    }else mergeTrigKeys(passingKeys,trigKeys,isAND_);
  }
  cleanTrigKeys(passingKeys);
  
  return passingKeys;
}

TrigObjTnPHistColl::TrigObjVarF::TrigObjVarF(std::string varName):isAbs_(false)
{
  //first look for "Abs" at the end of the string
  auto absPos = varName.rfind("Abs");
  if(absPos != std::string::npos && absPos+3 == varName.size() ) {
    isAbs_ = true;
    varName = varName.erase(absPos);
  }
  if(varName=="pt") varFunc_ = &trigger::TriggerObject::pt;
  else if(varName=="eta") varFunc_ = &trigger::TriggerObject::eta;
  else if(varName=="phi") varFunc_ = &trigger::TriggerObject::phi;
  else{
    std::ostringstream msg;
    msg<<"var "<<varName<<" not recognised (use pt or p rather than et or e for speed!) ";
    if(isAbs_) msg<<" varName was \"Abs\" suffex cleaned where it tried to remove \"Abs\" at the end of the variable name ";
    msg <<__FILE__<<","<<__LINE__<<std::endl;
    throw cms::Exception("ConfigError") <<msg.str();
  }
}

TrigObjTnPHistColl::HistFiller::HistFiller(const edm::ParameterSet& config):
  localCuts_(config.getParameter<std::vector<edm::ParameterSet> >("localCuts")),
  var_(config.getParameter<std::string>("var"))
{
  
}

edm::ParameterSetDescription TrigObjTnPHistColl::HistFiller::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.addVPSet("localCuts",VarRangeCut<trigger::TriggerObject>::makePSetDescription());
  desc.add<std::string>("var","pt");
  return desc;
}

void TrigObjTnPHistColl::HistFiller::operator()(const trigger::TriggerObject& probe,float mass,
						const ConcurrentMonitorElement& hist)const
{
  if(localCuts_(probe)) hist.fill(var_(probe),mass);
}



TrigObjTnPHistColl::HistDefs::HistDefs(const edm::ParameterSet& config):
  massBins_(convertToFloat(config.getParameter<std::vector<double> >("massBins")))
{
  const auto histConfigs = config.getParameter<std::vector<edm::ParameterSet> >("configs");
  for(const auto& histConfig : histConfigs){
    histData_.emplace_back(Data(histConfig));
  }
}

edm::ParameterSetDescription TrigObjTnPHistColl::HistDefs::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.addVPSet("configs",Data::makePSetDescription(),std::vector<edm::ParameterSet>()); 
  std::vector<double> massBins;
  for(float mass = 60;mass<=120;mass+=1) massBins.push_back(mass);
  desc.add<std::vector<double>>("massBins",massBins);
  return desc;
}

std::vector<std::pair<TrigObjTnPHistColl::HistFiller,ConcurrentMonitorElement> > TrigObjTnPHistColl::HistDefs::bookHists(DQMStore::ConcurrentBooker& iBooker,const std::string& name,const std::string& title)const
{
  std::vector<std::pair<HistFiller,ConcurrentMonitorElement> > hists;
  for(const auto& data : histData_){
    hists.push_back({data.filler(),data.book(iBooker,name,title,massBins_)});
  }
  return hists;
}

TrigObjTnPHistColl::HistDefs::Data::Data(const edm::ParameterSet& config):
  histFiller_(config.getParameter<edm::ParameterSet>("filler")),
  bins_(convertToFloat(config.getParameter<std::vector<double> >("bins"))),
  nameSuffex_(config.getParameter<std::string>("nameSuffex")),
  titleSuffex_(config.getParameter<std::string>("titleSuffex"))
{ 

}

edm::ParameterSetDescription TrigObjTnPHistColl::HistDefs::Data::makePSetDescription()
{
  edm::ParameterSetDescription desc;
  desc.add<edm::ParameterSetDescription>("filler",TrigObjTnPHistColl::HistFiller::makePSetDescription());
  desc.add<std::vector<double> >("bins",{-2.5,-1.5,0,1.5,2.5});
  desc.add<std::string>("nameSuffex","_eta");
  desc.add<std::string>("titleSuffex",";#eta;mass [GeV]");
  return desc;
}

ConcurrentMonitorElement TrigObjTnPHistColl::HistDefs::Data::book(DQMStore::ConcurrentBooker& iBooker,
					      const std::string& name,const std::string& title,
					      const std::vector<float>& massBins)const
{
  return iBooker.book2D((name+nameSuffex_).c_str(),(title+titleSuffex_).c_str(),
			bins_.size()-1,bins_.data(),massBins.size()-1,massBins.data());
}

void TrigObjTnPHistColl::HistColl::bookHists(DQMStore::ConcurrentBooker& iBooker,
					     const std::string& name,const std::string& title,
					     const HistDefs& histDefs)
{
  hists_ = histDefs.bookHists(iBooker,name,title);
}

void TrigObjTnPHistColl::HistColl::fill(const trigger::TriggerObject& probe,float mass)const
{
  for(auto& hist : hists_){
    hist.first(probe,mass,hist.second);
  }
}

void TrigObjTnPHistColl::ProbeData::bookHists(const std::string& tagName,
					      DQMStore::ConcurrentBooker& iBooker,
					      const HistDefs& histDefs)
{
  hists_.bookHists(iBooker,tagName+"_"+probeFilter_,tagName+"_"+probeFilter_,histDefs);
}

void TrigObjTnPHistColl::ProbeData::fill(const trigger::size_type tagKey,const trigger::TriggerEvent& trigEvt,const VarRangeCutColl<trigger::TriggerObject>& probeCuts)const
{
  auto probeKeys = getKeys(trigEvt,probeFilter_);
  for(auto probeKey : probeKeys){
    const trigger::TriggerObject& probe = trigEvt.getObjects()[probeKey];
    if(tagKey != probeKey && probeCuts(probe) ){
      const trigger::TriggerObject& tag = trigEvt.getObjects()[tagKey];
      auto massFunc = [](float pt1,float eta1,float phi1,float pt2,float eta2,float phi2){
	return std::sqrt( 2*pt1*pt2*( std::cosh(eta1-eta2) - std::cos(phi1-phi2) ) );
      };
      float mass = massFunc(tag.pt(),tag.eta(),tag.phi(),probe.pt(),probe.eta(),probe.phi());
      hists_.fill(probe,mass);
    }
  }
}

