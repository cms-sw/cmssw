//These objects allow an arbitary parameristisation to be used
//design:
// 1) the function and parameterisation can be changing without having to change the interface
//    or breaking backwards compatibiltiy with existing configs.
//    This is vital for the HLT and the main driving force behind the design which otherwise 
//    could have been a lot simplier

// 2) An intermediate object, "AbsEtaNrClus" is used to pass in the eta and nr cluster variables
//    as it allows the function to accept either a reco::ElectronSeed or an eta/nrClus pair.
//    The former simplies the accessing of the eta/nrClus from the seed object and the later
//    simplifies unit testing as one can generate all possible values of eta/nrClus easily

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"

#include "TF1.h"

namespace egPM {
  
  struct GausPlusConstFunc {
    float constTerm_,norm_,mean_,sigma_;
    constexpr static float kSqrt2Pi=std::sqrt(std::asin(1)*4);
    GausPlusConstFunc(const edm::ParameterSet& config):
      constTerm_(config.getParameter<double>("constTerm")),norm_(config.getParameter<double>("norm")),
      mean_(config.getParameter<double>("mean")),sigma_(config.getParameter<double>("sigma")){}
    float operator()(float x)const{return constTerm_+norm_*std::exp(-.5*(x-mean_)*(x-mean_)/(sigma_*sigma_))/(kSqrt2Pi*sigma_);}
  };
  
  template<size_t order>
  struct PolyFunc {
    std::array<float,order+1> para_;
    PolyFunc(const edm::ParameterSet& config){
      for(size_t paraNr=0;paraNr<para_.size();paraNr++){
	std::ostringstream paraName;
	paraName<<"p"<<paraNr;
	para_[paraNr]=config.getParameter<double>(paraName.str());
      }
    }
    float operator()(float x)const{
      float retVal=0.,xVal=1.;
      for(size_t termOrder=0;termOrder<=order;termOrder++){
	retVal+=xVal*para_[termOrder];
	xVal*=x;
      }
      return retVal;
    }
  };

  struct AbsEtaNrClus{
    float absEta;
    size_t nrClus;
    
    AbsEtaNrClus(const reco::ElectronSeed& seed){
      reco::SuperClusterRef scRef = seed.caloCluster().castTo<reco::SuperClusterRef>();
      absEta = std::abs(scRef->eta());
      nrClus = scRef->clustersSize();
    }
    AbsEtaNrClus(float iEta,size_t iNrClus): //iEta= input eta
      absEta(std::abs(iEta)),nrClus(iNrClus){}
  };
  
  template<typename ParamType>
  class ParamBin {
  public:
    ParamBin(){}
    virtual ~ParamBin(){}
    virtual bool pass(const ParamType&)const=0; 
    virtual float operator()(const ParamType&)const=0;
  protected:
    //right now only TF1 is supported so short cut the function
    //the FUNCTYPE::funcExpr is designed for future extensions
    static std::string stripFuncId(const std::string& inStr){
      if(inStr.substr(0,5)=="TF1:=") return inStr.substr(5);
      else return std::string();
    }
  };
   

  class AbsEtaClusParamBin : public ParamBin<egPM::AbsEtaNrClus>  {
    size_t minNrClus_; //inclusive
    size_t maxNrClus_; //inclusive
    float minEta_; //inclusive
    float maxEta_;//exclusive
    //std::function<float(float)> etaFunc_; 
    TF1 etaFunc_;
  public:
    
    AbsEtaClusParamBin(const edm::ParameterSet& config):
      minNrClus_(config.getParameter<int>("minNrClus")),
      maxNrClus_(config.getParameter<int>("maxNrClus")),
      minEta_(config.getParameter<double>("minEta")),
      maxEta_(config.getParameter<double>("maxEta")),
      etaFunc_("func",stripFuncId(config.getParameter<std::string>("funcType")).c_str(),0,3.)
    {
      const std::vector<double> params = config.getParameter<std::vector<double>>("funcParams");
      for(size_t paraNr=0;paraNr<params.size();paraNr++){
	etaFunc_.SetParameter(paraNr,params[paraNr]);
      }
    }

    bool pass(const egPM::AbsEtaNrClus& seed)const override {
      return seed.absEta>=minEta_ && seed.absEta<maxEta_ &&
	seed.nrClus>=minNrClus_ && seed.nrClus<=maxNrClus_;
    }
    
    float operator()(const egPM::AbsEtaNrClus& seed)const override{
      if(!pass(seed)) return 0;
      else return etaFunc_.Eval(seed.absEta);	
    }
  };

  class AbsEtaClusWithClusCorrParamBin : public AbsEtaClusParamBin  {
    float corrPerClus_;
    int clusNrOffset_;
  public:
    AbsEtaClusWithClusCorrParamBin(const edm::ParameterSet& config): 
      AbsEtaClusParamBin(config),
      corrPerClus_(config.getParameter<double>("corrPerClus")),
      clusNrOffset_(config.getParameter<int>("clusNrOffset")){}
    float operator()(const egPM::AbsEtaNrClus& seed)const override{
      return corrPerClus_*std::max(static_cast<int>(seed.nrClus)-clusNrOffset_,0)+AbsEtaClusParamBin::operator()(seed);
    }
  };


  
  template<typename ParamType>
  class Param {
    std::vector<std::unique_ptr<ParamBin<ParamType> > > bins_;
  public:
    Param(const edm::ParameterSet& config){
      std::vector<edm::ParameterSet> binConfigs = config.getParameter<std::vector<edm::ParameterSet> >("bins");
      for(auto& binConfig : binConfigs) bins_.emplace_back(createParamBin_(binConfig));
      std::cout<<" "<<std::endl;
      for(auto& binConfig : binConfigs) std::cout <<"paraSets.push_back(edm::ParameterSet(\""<<binConfig.toString()<<"\"));"<<std::endl;
    }
    float operator()(const ParamType& seed)const{
      for(auto& bin : bins_){
	if(bin->pass(seed)) return  (*bin)(seed);
      }
      return -1; //didnt find a suitable bin, just return -1 for now
    } 
	
  private:
    std::unique_ptr<ParamBin<ParamType> > createParamBin_(const edm::ParameterSet& config){
      std::string type = config.getParameter<std::string>("binType");
      if(type=="AbsEtaClusParamBin") return std::make_unique<AbsEtaClusParamBin>(config);
      else if(type=="AbsEtaClusWithClusCorrParamBin") return std::make_unique<AbsEtaClusWithClusCorrParamBin>(config);
      else throw cms::Exception("InvalidConfig") << " type "<<type<<" is not recognised, configuration is invalid and needs to be fixed"<<std::endl;
    }
  };
}
