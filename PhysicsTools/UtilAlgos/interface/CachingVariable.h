#ifndef ConfigurableAnalysis_CachingVariable_H
#define ConfigurableAnalysis_CachingVariable_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
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
  std::string text() const {
    std::string text;
    for (unsigned int i=0;i!=d_.size();++i)
      text+=d_[i]+"\n";
    return text;
  }
  const std::vector<std::string> lines(){return d_;}
  void addLine(const std::string & l){d_.push_back(l);}
 private :
  std::vector<std::string> d_;
};

class VariableComputer;

class CachingVariable {
 public:
  //every variable return double values
  typedef double valueType;
  typedef std::pair<bool, valueType> evalType;
  typedef std::map<std::string, const CachingVariable*> vMap;
  struct CachingVariableFactoryArg {
    CachingVariableFactoryArg( const CachingVariableFactoryArg & copy) : n(copy.n),m(copy.m),iConfig(copy.iConfig){}
    CachingVariableFactoryArg(std::string & N,CachingVariable::vMap & M,edm::ParameterSet & P) : n(N),m(M),iConfig(P){}
    std::string & n;
    CachingVariable::vMap & m;
    edm::ParameterSet & iConfig;
  };

  CachingVariable(std::string m, std::string n, const edm::ParameterSet & iConfig, edm::ConsumesCollector& iC) :
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

  void print() const {
    edm::LogVerbatim("CachingVariable")<<name()
				       <<"\n"<<description().text();
  }
 protected:

  mutable evalType cache_;
  mutable edm::Event::CacheIdentifier_t eventCacheID_=0;

  std::string method_;
  std::string name_;
  mutable std::string holderName_;
  void setCache(valueType & v) const {
    eventCacheID_ = std::numeric_limits<edm::Event::CacheIdentifier_t>::max();
    cache_.first=true; cache_.second = v;}
  void setNotCompute() const {
    eventCacheID_ = std::numeric_limits<edm::Event::CacheIdentifier_t>::max();
    cache_.first=false; cache_.second = 0;}
  evalType & baseEval(const edm::Event & iEvent) const {
    if(notSeenThisEventAlready(iEvent)) {
      LogDebug("CachingVariable")<<name_+":"+holderName_<<" is checking once";
      cache_=eval(iEvent);
    }
    return cache_;
  }
  bool notSeenThisEventAlready(const edm::Event& iEvent) const {
    bool retValue = (std::numeric_limits<edm::Event::CacheIdentifier_t>::max() != eventCacheID_ and
		     eventCacheID_ != iEvent.cacheIdentifier());
    if(retValue) {
      eventCacheID_=iEvent.cacheIdentifier();
    }
    return retValue;
  }

  //cannot be made purely virtual otherwise one cannot have purely CachingVariableObjects
  virtual evalType eval(const edm::Event & iEvent) const {return std::make_pair(false,0);};

  Description d_;
  edm::ParameterSet conf_;
  friend class VariableComputer;
};


class ComputedVariable;
class VariableComputer{
 public:
  VariableComputer(const CachingVariable::CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC);
  virtual ~VariableComputer(){}

  virtual void compute(const edm::Event & iEvent) const = 0;
  const std::string & name() const { return name_;}
  void declare(std::string var, edm::ConsumesCollector& iC);
  void assign(std::string  var, double & value) const;
  void doesNotCompute() const;
  void doesNotCompute(std::string var) const;

  bool notSeenThisEventAlready(const edm::Event& iEvent) const {
    bool retValue = eventCacheID_ != iEvent.cacheIdentifier();
    if(retValue) {
      eventCacheID_=iEvent.cacheIdentifier();
    }
    return retValue;
  }

 protected:
  const CachingVariable::CachingVariableFactoryArg & arg_;
  std::string name_;
  std::string method_;
  mutable std::map<std::string ,const ComputedVariable *> iCompute_;
  std::string separator_;

  mutable edm::Event::CacheIdentifier_t eventCacheID_=0;

};


#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

typedef edmplugin::PluginFactory< CachingVariable* (CachingVariable::CachingVariableFactoryArg, edm::ConsumesCollector& iC) > CachingVariableFactory;
typedef edmplugin::PluginFactory< VariableComputer* (CachingVariable::CachingVariableFactoryArg, edm::ConsumesCollector& iC) > VariableComputerFactory;

class ComputedVariable : public CachingVariable {
 public:
  ComputedVariable(const CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC );
  ComputedVariable(const std::string & M, std::string & N, edm::ParameterSet & P, const VariableComputer * c, edm::ConsumesCollector& iC) :
    CachingVariable(M,N,P,iC), myComputer(c){}
  virtual ~ComputedVariable(){};

  virtual evalType eval(const edm::Event & iEvent) const {
    if (myComputer->notSeenThisEventAlready(iEvent))
      myComputer->compute(iEvent);
    return cache_;
  }
 private:
  const VariableComputer * myComputer;
};

class VariableComputerTest : public VariableComputer {
 public:
  VariableComputerTest(const CachingVariable::CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) ;
  ~VariableComputerTest(){};

  void compute(const edm::Event & iEvent) const;
};

class Splitter : public CachingVariable {
 public:
  Splitter(std::string method, std::string n, const edm::ParameterSet & iConfig, edm::ConsumesCollector& iC) :
    CachingVariable(method,n,iConfig,iC) {}

  //purely virtual here
  virtual CachingVariable::evalType eval(const edm::Event & iEvent) const =0;

  unsigned int maxIndex() const { return maxSlots()-1;}

  //maximum NUMBER of slots: counting over/under flows
  virtual unsigned int maxSlots() const { return labels_.size();}

  const std::string shortLabel(unsigned int i) const{
    if (i>=short_labels_.size()){
      edm::LogError("Splitter")<<"trying to access slots short_label at index: "<<i<<"while of size: "<<short_labels_.size()<<"\n"<<conf_.dump();
      return short_labels_.back(); }
    else  return short_labels_[i];}

  const std::string & label(unsigned int i) const{
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
  VarSplitter(const CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC ) :
    Splitter("VarSplitter",arg.n,arg.iConfig,iC) {
    var_=arg.iConfig.getParameter<std::string>("var");
    useUnderFlow_=arg.iConfig.getParameter<bool>("useUnderFlow");
    useOverFlow_=arg.iConfig.getParameter<bool>("useOverFlow");
    slots_=arg.iConfig.getParameter<std::vector<double> >("slots");
    if (useUnderFlow_){
      labels_.push_back("underflow");
      short_labels_.push_back("_"+arg.n+"_underflow");}
    std::vector<std::string> confLabels;
    if (arg.iConfig.exists("labels")){
      confLabels=arg.iConfig.getParameter<std::vector<std::string> >("labels");
    }
    else{
      std::string labelFormat = arg.iConfig.getParameter<std::string>("labelsFormat");
      for (unsigned int is=0;is!=slots_.size()-1;++is){
	std::string l(Form(labelFormat.c_str(),slots_[is],slots_[is+1]));
	confLabels.push_back(l);
      }
    }
    for (unsigned int i=0;i!=confLabels.size();++i){
      labels_.push_back(confLabels[i]);
      std::stringstream ss;
      ss<<"_"<<arg.n<<"_"<<i;
      short_labels_.push_back(ss.str());
    }
    if (useOverFlow_)
      { labels_.push_back("overFlow");
	short_labels_.push_back("_"+arg.n+"_overFlow");}

    //check consistency
    if (labels_.size()!=maxSlots())
      edm::LogError("Splitter")<<"splitter with name: "<<name()<<" has inconsistent configuration\n"<<conf_.dump();

    arg.m[arg.n]=this;
  }

  CachingVariable::evalType eval(const edm::Event & iEvent) const;

  //redefine the maximum number of slots
  unsigned int maxSlots() const{
    unsigned int s=slots_.size()-1;
    if (useUnderFlow_) s++;
    if (useOverFlow_) s++;
    return s;}

 protected:
  std::string var_;
  bool useUnderFlow_;
  bool useOverFlow_;
  std::vector<double> slots_;
};

template <typename Object> class sortByStringFunction  {
 public:
  sortByStringFunction(StringObjectFunction<Object> * f) : f_(f){}
  ~sortByStringFunction(){}

  bool operator() (const Object * o1, const Object * o2) {
    return (*f_)(*o1) > (*f_)(*o2);
  }
 private:
  StringObjectFunction<Object> * f_;
};

template <typename Object, const char * label>
class ExpressionVariable : public CachingVariable {
 public:
  ExpressionVariable(const CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) :
    CachingVariable(std::string(label)+"ExpressionVariable",arg.n,arg.iConfig,iC) , f_(0), forder_(0) {
    srcTag_=edm::Service<InputTagDistributorService>()->retrieve("src",arg.iConfig);
    src_=iC.consumes<edm::View<Object> >(srcTag_);
    //old style constructor
    if (arg.iConfig.exists("expr") && arg.iConfig.exists("index")){
      std::string expr=arg.iConfig.getParameter<std::string>("expr");
      index_=arg.iConfig.getParameter<unsigned int>("index");
      f_ = new StringObjectFunction<Object>(expr);
      addDescriptionLine("calculating: "+expr);
      std::stringstream ss;
      ss<<"on object at index: "<<index_<<" of: "<<srcTag_;

      if (arg.iConfig.exists("order")){
	std::string order=arg.iConfig.getParameter<std::string>("order");
	forder_ = new StringObjectFunction<Object>(order);
	ss<<" after sorting according to: "<<order;
      }else forder_ =0;

      if (arg.iConfig.exists("selection")){
	std::string selection=arg.iConfig.getParameter<std::string>("selection");
	selector_ = new StringCutObjectSelector<Object>(selection);
	ss<<" and selecting only: "<<selection;
      }else selector_=0;



      addDescriptionLine(ss.str());	ss.str("");
      arg.m[arg.n] = this;
    }
    else{
      //multiple instance constructor
      std::map<std::string, edm::Entry> indexEntry;
      if (arg.n.find("_N")!=std::string::npos){
	//will have to loop over indexes
	std::vector<unsigned int> indexes = arg.iConfig.getParameter<std::vector<unsigned int> >("indexes");
	for (unsigned int iI=0;iI!=indexes.size();++iI){
	  edm::ParameterSet toUse = arg.iConfig;
	  edm::Entry e("unsigned int",indexes[iI],true);
	  std::stringstream ss;
	  //add +1 0->1, 1->2, ... in the variable label
	  ss<<indexes[iI]+1;
	  indexEntry.insert(std::make_pair(ss.str(),e));
	}
      }//contains "_N"

      std::map< std::string, edm::Entry> varEntry;
      if (arg.n.find("_V")!=std::string::npos){
	//do something fancy for multiple variable from one PSet
	std::vector<std::string> vars = arg.iConfig.getParameter<std::vector<std::string> >("vars");
	for (unsigned int v=0;v!=vars.size();++v){
	  unsigned int sep=vars[v].find(":");
	  std::string name=vars[v].substr(0,sep);
	  std::string expr=vars[v].substr(sep+1);

	  edm::Entry e("string",expr,true);
	  varEntry.insert(std::make_pair(name,e));
	}
      }//contains "_V"

      std::string radical = arg.n;
      //remove the "_V";
      if (!varEntry.empty())
	radical = radical.substr(0,radical.size()-2);
      //remove the "_N";
      if (!indexEntry.empty())
	radical = radical.substr(0,radical.size()-2);

      if(varEntry.empty()){
	//loop only the indexes
	for(std::map< std::string, edm::Entry>::iterator iIt=indexEntry.begin();iIt!=indexEntry.end();++iIt){
	  edm::ParameterSet toUse = arg.iConfig;
	  toUse.insert(true,"index",iIt->second);
	  std::string newVname = radical+iIt->first;
	  //	  std::cout<<"in the loop, creating variable with name: "<<newVname<<std::endl;
	  // the constructor auto log the new variable in the map
	  new ExpressionVariable(CachingVariable::CachingVariableFactoryArg(newVname,arg.m,toUse),iC);
	}
      }else{
	for (std::map< std::string, edm::Entry>::iterator vIt=varEntry.begin();vIt!=varEntry.end();++vIt){
	  if (indexEntry.empty()){
	    edm::ParameterSet toUse = arg.iConfig;
	    toUse.insert(true,"expr",vIt->second);
	    std::string newVname = radical+vIt->first;
	    //	    std::cout<<"in the loop, creating variable with name: "<<newVname<<std::endl;
	    // the constructor auto log the new variable in the map
	    new ExpressionVariable(CachingVariable::CachingVariableFactoryArg(newVname,arg.m,toUse),iC);
	  }else{
	    for(std::map< std::string, edm::Entry>::iterator iIt=indexEntry.begin();iIt!=indexEntry.end();++iIt){
	      edm::ParameterSet toUse = arg.iConfig;
	      toUse.insert(true,"expr",vIt->second);
	      toUse.insert(true,"index",iIt->second);
	      std::string newVname = radical+iIt->first+vIt->first;
	      //	      std::cout<<"in the loop, creating variable with name: "<<newVname<<std::endl;
	      // the constructor auto log the new variable in the map
	      new ExpressionVariable(CachingVariable::CachingVariableFactoryArg(newVname,arg.m,toUse),iC);
	    }}
	}
      }
      //there is a memory leak here, because the object we are in is not logged in the arg.m, the variable is not valid
      // anyways, but reside in memory with no ways of de-allocating it.
      // since the caching variables are actually "global" objects, it does not matter.
      // we cannot add it to the map, otherwise, it would be considered for eventV ntupler
    }
  }
  ~ExpressionVariable(){
    if (f_) delete f_;
    if (forder_) delete forder_;
    if (selector_) delete selector_;
  }

  CachingVariable::evalType eval(const edm::Event & iEvent) const {
    if (!f_) {
      edm::LogError(method())<<" no parser attached.";
      return std::make_pair(false,0);
    }
    edm::Handle<edm::View<Object> > oH;
    iEvent.getByToken(src_,oH);
    if (index_>=oH->size()){
      LogDebug(method())<<"fail to get object at index: "<<index_<<" in collection: "<<src_;
      return std::make_pair(false,0);
    }

    //get the ordering right first. if required
    if (selector_ || forder_){
      std::vector<const Object*> copyToSort(0);
      copyToSort.reserve(oH->size());
      for (unsigned int i=0;i!=oH->size();++i){
        if (selector_ && !((*selector_)((*oH)[i]))) continue;
        copyToSort.push_back(&(*oH)[i]);
      }
      if (index_ >= copyToSort.size()) return std::make_pair(false,0);
      if (forder_) std::sort(copyToSort.begin(), copyToSort.end(), sortByStringFunction<Object>(forder_));

      const Object * o = copyToSort[index_];
      return std::make_pair(true,(*f_)(*o));
    }
    else{
      const Object & o = (*oH)[index_];
      return std::make_pair(true,(*f_)(o));
    }
  }

 private:
  edm::InputTag srcTag_;
  edm::EDGetTokenT<edm::View<Object> > src_;
  unsigned int index_;
  StringObjectFunction<Object> * f_;
  StringObjectFunction<Object> * forder_;
  StringCutObjectSelector<Object> * selector_;
};


template< typename LHS,const char * lLHS, typename RHS,const char * lRHS, typename Calculator>
class TwoObjectVariable : public CachingVariable {
public:
  TwoObjectVariable(const CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) :
    CachingVariable(Calculator::calculationType()+std::string(lLHS)+std::string(lRHS),arg.n,arg.iConfig,iC),
    srcLhsTag_(edm::Service<InputTagDistributorService>()->retrieve("srcLhs",arg.iConfig)),
    srcLhs_(iC.consumes<std::vector<LHS> >(srcLhsTag_)),
    indexLhs_(arg.iConfig.getParameter<unsigned int>("indexLhs")),
    srcRhsTag_(edm::Service<InputTagDistributorService>()->retrieve("srcRhs",arg.iConfig)),
    srcRhs_(iC.consumes<std::vector<RHS> >(srcRhsTag_)),
    indexRhs_(arg.iConfig.getParameter<unsigned int>("indexRhs"))
      {
	std::stringstream ss;
	addDescriptionLine(Calculator::description());
	ss<<"with Obj1 at index: "<<indexLhs_<<" of: "<<srcLhs_;
	addDescriptionLine(ss.str());	ss.str("");
	ss<<"with Obj2 at index: "<<indexRhs_<<" of: "<<srcRhs_;
	addDescriptionLine(ss.str());	ss.str("");
	arg.m[arg.n]=this;
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
      iEvent.getByToken(srcLhs_, lhsH);
      if (lhsH.failedToGet()){
	LogDebug("TwoObjectVariable")<<name()<<" could not get a collection with label: "<<srcLhsTag_;
	return failed;}
      if (indexLhs_>=lhsH->size()){
	LogDebug("TwoObjectVariable")<<name()<<" tries to access index: "<<indexLhs_<<" of: "<<srcLhsTag_<<" with: "<<lhsH->size()<<" entries.";
      return failed;}
      const LHS & lhs = (*lhsH)[indexLhs_];

      edm::Handle<std::vector<RHS> > rhsH;
      iEvent.getByToken(srcRhs_, rhsH);
      if (rhsH.failedToGet()){
	LogDebug("TwoObjectVariable")<<name()<<" could not get a collection with label: "<<srcLhsTag_;
	return failed;}

      if (indexRhs_>=rhsH->size()){
	LogDebug("TwoObjectVariable")<<name()<<" tries to access index: "<<indexRhs_<<" of: "<<srcRhsTag_<<" with: "<<rhsH->size()<<" entries.";
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
  edm::InputTag srcLhsTag_;
  edm::EDGetTokenT<std::vector<LHS> > srcLhs_;
  unsigned int indexLhs_;
  edm::InputTag srcRhsTag_;
  edm::EDGetTokenT<std::vector<RHS> > srcRhs_;
  unsigned int indexRhs_;
};


class VariablePower : public CachingVariable {
 public:
  VariablePower(const CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) :
    CachingVariable("Power",arg.n,arg.iConfig,iC){
    power_=arg.iConfig.getParameter<double>("power");
    var_=arg.iConfig.getParameter<std::string>("var");
    std::stringstream ss("Calculare X^Y, with X=");
    ss<<var_<<" and Y="<<power_;
    addDescriptionLine(ss.str());
    arg.m[arg.n]=this;
  }
  ~VariablePower(){}

 //concrete calculation of the variable
  CachingVariable::evalType eval(const edm::Event & iEvent) const;

 private:
  double power_;
  std::string var_;
};


template <typename TYPE>
class SimpleValueVariable : public CachingVariable {
 public:
  SimpleValueVariable(const CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) :
    CachingVariable("SimpleValueVariable",arg.n,arg.iConfig,iC),
    src_(iC.consumes<TYPE>(edm::Service<InputTagDistributorService>()->retrieve("src",arg.iConfig))) { arg.m[arg.n]=this;}
  CachingVariable::evalType eval(const edm::Event & iEvent) const{
    edm::Handle<TYPE> value;
    try{    iEvent.getByToken(src_,value);   }
    catch(...){ return std::make_pair(false,0); }
    if (value.failedToGet() || !value.isValid()) return std::make_pair(false,0);
    else return std::make_pair(true, *value);
  }
 private:
  edm::EDGetTokenT<TYPE> src_;
};

template <typename TYPE>
class SimpleValueVectorVariable : public CachingVariable {
 public:
  SimpleValueVectorVariable(const CachingVariableFactoryArg& arg, edm::ConsumesCollector& iC) :
    CachingVariable("SimpleValueVectorVariable",arg.n,arg.iConfig,iC),
    src_(iC.consumes<TYPE>(edm::Service<InputTagDistributorService>()->retrieve("src",arg.iConfig))),
    index_(arg.iConfig.getParameter<unsigned int>("index")) { arg.m[arg.n]=this;}
  CachingVariable::evalType eval(const edm::Event & iEvent) const{
    edm::Handle<std::vector<TYPE> > values;
    try { iEvent.getByToken(src_,values);}
    catch(...){ return std::make_pair(false,0); }
    if (values.failedToGet() || !values.isValid()) return std::make_pair(false,0);
    else if (index_>=values->size()) return std::make_pair(false,0);
    else return std::make_pair(true, (*values)[index_]);
  }

 private:
  edm::EDGetTokenT<TYPE> src_;
  unsigned int index_;
};







#endif
