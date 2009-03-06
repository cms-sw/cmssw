#ifndef ConfigurableAnalysis_CachingVariable_H
#define ConfigurableAnalysis_CachingVariable_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Services/interface/UpdaterService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "PhysicsTools/Utilities/interface/StringObjectFunction.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/UtilAlgos/interface/InputTagDistributor.h"

namespace edm {
  class EventSetup;
}
#include <vector>
#include "TString.h"

class Description {
 public:
  Description(){}
  Description(std::vector<std::string> & d) : d_(d){}
  std::string text(){
    std::string text;
    for (uint i=0;i!=d_.size();++i) 
      text+=d_[i]+"\n";
    return text;
  }
  const std::vector<std::string> lines(){return d_;}
  void addLine(const std::string & l){d_.push_back(l);}
 private :
  std::vector<std::string> d_;
};

class CachingVariable {
 public:
  //every variable return double values
  typedef double valueType;
  typedef std::pair<bool, valueType> evalType;

  CachingVariable(std::string m, std::string n, const edm::ParameterSet & iConfig) :
    cache_(std::make_pair(false,0)),method_(m),
    name_(n),conf_(iConfig) {}

  virtual ~CachingVariable() {}

  //does the variable computes
  bool compute(const edm::Event & iEvent) const {    return baseEval(iEvent).first;  }

  //accessor to the computed/cached value
  valueType operator()(const edm::Event & iEvent) const  {    return baseEval(iEvent).second;  }

  const std::string & name() const {return name_;}
  const std::string & method() const { return method_;}
  const Description & description()const { return d_;}
  void addDescriptionLine(const std::string & s){ d_.addLine(s);}
  const std::string & holderName() const { return holderName_;}
  void setHolder(std::string hn) const { holderName_=hn;}

 protected:

  //  mutable valueType cache_;
  mutable evalType cache_;

  std::string method_;
  std::string name_;
  mutable std::string holderName_;
  evalType & baseEval(const edm::Event & iEvent) const {
    if (edm::Service<UpdaterService>()->checkOnce(name_+":"+holderName_)){
      LogDebug("CachingVariable")<<name_+":"+holderName_<<" is checking once";
      cache_=eval(iEvent);
    }
    return cache_;
  }
  virtual evalType eval(const edm::Event & iEvent) const =0;

  Description d_;
  edm::ParameterSet conf_;
};



class Splitter : public CachingVariable {
 public:
  Splitter(std::string method, std::string n, const edm::ParameterSet & iConfig) :
    CachingVariable(method,n,iConfig) {}

  //purely virtual here 
  virtual CachingVariable::evalType eval(const edm::Event & iEvent) const =0;

  uint maxIndex() const { return maxSlots()-1;}

  //maximum NUMBER of slots: counting over/under flows
  virtual uint maxSlots() const { return labels_.size();}

  const std::string shortLabel(uint i) const{ 
    if (i>=short_labels_.size()){
      edm::LogError("Splitter")<<"trying to access slots short_label at index: "<<i<<"while of size: "<<short_labels_.size()<<"\n"<<conf_.dump();
      return short_labels_.back(); }
    else  return short_labels_[i];}
  
  const std::string & label(uint i) const{ 
    if (i>=labels_.size()){
      edm::LogError("Splitter")<<"trying to access slots label at index: "<<i<<"while of size: "<<labels_.size()<<"\n"<<conf_.dump();
      return labels_.back(); }
    else return labels_[i];}
  
 protected:
  std::vector<std::string> labels_;
  std::vector<std::string> short_labels_;
};


class VarSplitter : public Splitter{ 
 public:
  VarSplitter(std::string n, const edm::ParameterSet & iConfig) :
    Splitter("VarSplitter",n,iConfig) {
    var_=iConfig.getParameter<std::string>("var");
    useUnderFlow_=iConfig.getParameter<bool>("useUnderFlow");
    useOverFlow_=iConfig.getParameter<bool>("useOverFlow");
    slots_=iConfig.getParameter<std::vector<double> >("slots");
    if (useUnderFlow_){
      labels_.push_back("underflow");
      short_labels_.push_back("_"+n+"_underflow");}
    std::vector<std::string> confLabels;
    if (iConfig.exists("labels")){
      confLabels=iConfig.getParameter<std::vector<std::string> >("labels");
    }
    else{
      std::string labelFormat = iConfig.getParameter<std::string>("labelsFormat");
      for (uint is=0;is!=slots_.size()-1;++is){
	std::string l(Form(labelFormat.c_str(),slots_[is],slots_[is+1]));
	//---	std::cout<<"forming: "<<labelFormat<<" "<<slots_[is]<<" "<<slots_[is+1]<<" : "<<l<<std::endl;
	confLabels.push_back(l);
      }
    }
    for (uint i=0;i!=confLabels.size();++i){
      labels_.push_back(confLabels[i]);
      std::stringstream ss;
      ss<<"_"<<n<<"_"<<i;
      short_labels_.push_back(ss.str());
    }
    if (useOverFlow_)
      { labels_.push_back("overFlow");
	short_labels_.push_back("_"+n+"_overFlow");}
    
    //check consistency
    if (labels_.size()!=maxSlots())
      edm::LogError("Splitter")<<"splitter with name: "<<name()<<" has inconsistent configuration\n"<<conf_.dump();
  }

  CachingVariable::evalType eval(const edm::Event & iEvent) const;

  //redefine the maximum number of slots
  uint maxSlots() const{
    uint s=slots_.size()-1;
    if (useUnderFlow_) s++;
    if (useOverFlow_) s++;
    return s;}

 protected:
  std::string var_;
  bool useUnderFlow_;
  bool useOverFlow_;
  std::vector<double> slots_;
};


template <typename Object, const char * label> 
class ExpressionVariable : public CachingVariable {
 public:
  ExpressionVariable(std::string n, const edm::ParameterSet & iConfig) :
    CachingVariable(std::string(label)+"ExpressionVariable",n,iConfig) {
    src_=InputTagDistributor::retrieve("src",iConfig);
    std::string expr=iConfig.getParameter<std::string>("expr");
    index_=iConfig.getParameter<uint>("index");
    f_ = new StringObjectFunction<Object>(expr);
  }

  CachingVariable::evalType eval(const edm::Event & iEvent) const {
    edm::Handle<edm::View<Object> > oH;
    iEvent.getByLabel(src_,oH);
    if (index_>=oH->size()){
      LogDebug(method())<<"fail to get object at index: "<<index_<<" in collection: "<<src_;
      return std::make_pair(false,0);
    }
    const Object & o = (*oH)[index_];
    return std::make_pair(true,(*f_)(o));
  }

 private:
  edm::InputTag src_;
  uint index_;
  StringObjectFunction<Object> * f_;
};


template< typename LHS,const char * lLHS, typename RHS,const char * lRHS, typename Calculator>
class TwoObjectVariable : public CachingVariable {
public:
  TwoObjectVariable(std::string n, const edm::ParameterSet& iConfig) :
    //    CachingVariable(std::string(lLHS)+std::string(lRHS)+Calculator::calculationType,n,iConfig),
    CachingVariable(std::string(lLHS)+std::string(lRHS),n,iConfig),
    srcLhs_(InputTagDistributor::retrieve("srcLhs",iConfig)),
    indexLhs_(iConfig.getParameter<uint>("indexLhs")),
    srcRhs_(InputTagDistributor::retrieve("srcRhs",iConfig)),
    indexRhs_(iConfig.getParameter<uint>("indexRhs"))
      {
	std::stringstream ss;
	addDescriptionLine("cos delta phi variable: ");
	ss<<"Cos(DeltaPhi( Obj1, Oj2 )) ";
	addDescriptionLine(ss.str());	ss.str("");
	ss<<"with Obj1 at index: "<<indexLhs_<<" of: "<<srcLhs_;
	addDescriptionLine(ss.str());	ss.str("");
	ss<<"with Obj2 at index: "<<indexRhs_<<" of: "<<srcRhs_;
	addDescriptionLine(ss.str());	ss.str("");
      }

    class getObject{
    public:
      getObject() : test(false),lhs(0),rhs(0){}
      bool test;
      const LHS * lhs;
      const RHS * rhs;
    };
    getObject objects(const edm::Event & iEvent) const {
      getObject failed;
      edm::Handle<std::vector<LHS> > lhsH;
      iEvent.getByLabel(srcLhs_, lhsH);
      if (lhsH.failedToGet()){
	LogDebug("TwoObjectVariable")<<name()<<" could not get a collection with label: "<<srcLhs_;
	return failed;}
      if (indexLhs_>=lhsH->size()){
	LogDebug("TwoObjectVariable")<<name()<<" tries to access index: "<<indexLhs_<<" of: "<<srcLhs_<<" with: "<<lhsH->size()<<" entries.";
      return failed;}
      const LHS & lhs = (*lhsH)[indexLhs_];
      
      edm::Handle<std::vector<RHS> > rhsH;
      iEvent.getByLabel(srcRhs_, rhsH);
      if (rhsH.failedToGet()){
	LogDebug("TwoObjectVariable")<<name()<<" could not get a collection with label: "<<srcLhs_;
	return failed;}
      
      if (indexRhs_>=rhsH->size()){
	LogDebug("TwoObjectVariable")<<name()<<" tries to access index: "<<indexRhs_<<" of: "<<srcRhs_<<" with: "<<rhsH->size()<<" entries.";
	return failed;}
      const RHS & rhs = (*rhsH)[indexRhs_];
      
      failed.test=true;
      failed.lhs=&lhs;
      failed.rhs=&rhs;
      return failed;
    }

    //to be overloaded by the user
    virtual CachingVariable::valueType calculate(getObject & o) const {
      Calculator calc;
      return calc(*o.lhs,*o.rhs);
    }
    CachingVariable::evalType eval(const edm::Event & iEvent) const {
      getObject o=objects(iEvent);
      if (!o.test) return std::make_pair(false,0);
      return std::make_pair(true,calculate(o));
    }
private:
  edm::InputTag srcLhs_;
  uint indexLhs_;
  edm::InputTag srcRhs_;
  uint indexRhs_;
};


class Power : public CachingVariable {
 public:
  Power(std::string & n,const edm::ParameterSet & iConfig) :
    CachingVariable("Power",n,iConfig){
    power_=iConfig.getParameter<double>("power");
    var_=iConfig.getParameter<std::string>("var");
    std::stringstream ss("Calculare X^Y, with X=");
    ss<<var_<<" and Y="<<power_;
    addDescriptionLine(ss.str());
  }
  ~Power(){}
 
 //concrete calculation of the variable
  CachingVariable::evalType eval(const edm::Event & iEvent) const;

 private:
  double power_;
  std::string var_;
};



#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

typedef edmplugin::PluginFactory< CachingVariable* (std::string , edm::ParameterSet&) > CachingVariableFactory;



#endif
