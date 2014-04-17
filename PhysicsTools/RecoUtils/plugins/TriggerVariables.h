#include "PhysicsTools/UtilAlgos/interface/CachingVariable.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

class L1BitComputer : public VariableComputer {
 public:
  L1BitComputer(CachingVariable::CachingVariableFactoryArg arg, edm::ConsumesCollector& iC):VariableComputer(arg, iC){
    src_=iC.consumes<L1GlobalTriggerReadoutRecord>(edm::Service<InputTagDistributorService>()->retrieve("src",arg.iConfig));
    for(int i = 0;i!=128;i++){
      std::stringstream ss;
      ss<<i;
      declare(ss.str(), iC);
    }

    for(int i = 0;i!=64;i++){
      std::stringstream ss;
      ss<<"TechTrig_";
      ss<<i;
      declare(ss.str(), iC);
    }


  }
    ~L1BitComputer(){};

    void compute(const edm::Event & iEvent) const{
      edm::Handle<L1GlobalTriggerReadoutRecord> l1Handle;
      iEvent.getByToken(src_, l1Handle);
      if (l1Handle.failedToGet()) doesNotCompute();
      const DecisionWord & dWord = l1Handle->decisionWord();
      for(int i = 0;i!=128;i++){
	std::stringstream ss;
	ss<<i;
	double r=dWord.at(i);
	assign(ss.str(),r);
      }

      const TechnicalTriggerWord & tTWord = l1Handle->technicalTriggerWord();
      for(int i = 0;i!=64;i++){
	std::stringstream ss;
        ss<<"TechTrig_";
        ss<<i;
        double r=tTWord.at(i);
        assign(ss.str(),r);
      }



    }
 private:
    edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> src_;
};

#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

class HLTBitComputer : public VariableComputer {
 public:
  HLTBitComputer(CachingVariable::CachingVariableFactoryArg arg, edm::ConsumesCollector& iC):VariableComputer(arg, iC){
    src_=iC.consumes<edm::TriggerResults>(edm::Service<InputTagDistributorService>()->retrieve("src",arg.iConfig));
    HLTConfigProvider provider;
    //the function will not work anymore until a major redesign is performed. so long for HLT variables.
    //provider.init(src_.process());
    validTriggerNames_ =  provider.triggerNames();
    for (unsigned int iT=0;iT!=validTriggerNames_.size();++iT){
      TString tname(validTriggerNames_[iT]);
      tname.ReplaceAll("HLT_","");//remove the "HLT_" prefix
      declare(std::string(tname), iC);
    }
  }
    ~HLTBitComputer(){}
    void compute(const edm::Event & iEvent) const{
      edm::Handle<edm::TriggerResults> trh;
      iEvent.getByToken(src_,trh);
      if (!trh.isValid()) doesNotCompute();
      const edm::TriggerNames & triggerNames = iEvent.triggerNames(*trh);
      for (unsigned int iT=0;iT!=validTriggerNames_.size();++iT){

	TString tname(validTriggerNames_[iT]);
	tname.ReplaceAll("HLT_","");
	double r=trh->accept(triggerNames.triggerIndex(validTriggerNames_[iT]));
	assign(std::string(tname),r);
      }

    }
 private:
    edm::EDGetTokenT<edm::TriggerResults> src_;
    std::vector<std::string> validTriggerNames_;
};

class HLTBitVariable : public CachingVariable {
 public:
  HLTBitVariable(CachingVariableFactoryArg arg, edm::ConsumesCollector& iC ) :
    CachingVariable("HLTBitVariable",arg.n,arg.iConfig,iC),
    src_(iC.consumes<edm::TriggerResults>(edm::Service<InputTagDistributorService>()->retrieve("src",arg.iConfig))),
    bitName_(arg.n){ arg.m[arg.n]=this;}
    CachingVariable::evalType eval(const edm::Event & iEvent) const{
      edm::Handle<edm::TriggerResults> hltHandle;
      iEvent.getByToken(src_, hltHandle);
      if (hltHandle.failedToGet()) return std::make_pair(false,0);
      const edm::TriggerNames & trgNames = iEvent.triggerNames(*hltHandle);
      unsigned int index = trgNames.triggerIndex(bitName_);
      if ( index==trgNames.size() ) return std::make_pair(false,0);
      else return std::make_pair(true, hltHandle->accept(index));

    }
 private:
  edm::EDGetTokenT<edm::TriggerResults> src_;
  std::string bitName_;
};
