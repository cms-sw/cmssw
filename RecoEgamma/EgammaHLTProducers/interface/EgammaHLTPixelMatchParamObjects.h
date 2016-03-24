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
#include "TF2.h"
#include "TF3.h"

namespace egPM {
  template<typename T>
  struct ConfigType{
    typedef T type;
  };
  template<>
  struct ConfigType<size_t>{
    typedef int type;
  };
  template<>
  struct ConfigType<float>{
    typedef double type;
  };

  struct AbsEtaNrClus{
    float x;
    size_t y;
    
    AbsEtaNrClus(const reco::ElectronSeed& seed){
      reco::SuperClusterRef scRef = seed.caloCluster().castTo<reco::SuperClusterRef>();
      x = std::abs(scRef->eta());
      y = scRef->clustersSize();
    }
    AbsEtaNrClus(float iEta,size_t iNrClus): //iEta= input eta
      x(std::abs(iEta)),y(iNrClus){}

    bool pass(float absEtaMin,float absEtaMax,size_t nrClusMin,size_t nrClusMax)const{
      return x>=absEtaMin && x<absEtaMax && y>=nrClusMin && y<=nrClusMax;
    }
  };
  struct AbsEtaNrClusPhi{
    float x;
    size_t y;
    float z;

    AbsEtaNrClusPhi(const reco::ElectronSeed& seed){
      reco::SuperClusterRef scRef = seed.caloCluster().castTo<reco::SuperClusterRef>();
      x = std::abs(scRef->eta());
      y = scRef->clustersSize();
      z = scRef->phi();
    }
    AbsEtaNrClusPhi(float iEta,size_t iNrClus,float iPhi): //iEta= input eta
      x(std::abs(iEta)),y(iNrClus),z(iPhi){}

    bool pass(float absEtaMin,float absEtaMax,size_t nrClusMin,size_t nrClusMax,
	      float phiMin,float phiMax)const{
      return x>=absEtaMin && x<absEtaMax && y>=nrClusMin && y<=nrClusMax 
	&& z>=phiMin && z < phiMax;
    }
  };

  struct AbsEtaNrClusEt {
    float x;
    size_t y;
    float z;

    AbsEtaNrClusEt(const reco::ElectronSeed& seed){
      reco::SuperClusterRef scRef = seed.caloCluster().castTo<reco::SuperClusterRef>();
      x = std::abs(scRef->eta());
      y = scRef->clustersSize();
      z = scRef->energy()*sin(scRef->position().Theta());
    }
    AbsEtaNrClusEt(float iEta,size_t iNrClus,float iEt): //iEta= input eta
      x(std::abs(iEta)),y(iNrClus),z(iEt){}

    bool pass(float absEtaMin,float absEtaMax,size_t nrClusMin,size_t nrClusMax,
	      float etMin,float etMax)const{
      return x>=absEtaMin && x<absEtaMax && y>=nrClusMin && y<=nrClusMax 
	&& z>=etMin && z < etMax;
    }
  };
  
  template<typename ParamType,bool=true>
  struct TF1Wrap{
  private:
    TF1 func_;
  public:
    TF1Wrap(const std::string& funcExpr,const std::vector<double>& params):
      func_("func",funcExpr.c_str()){
      for(size_t paraNr=0;paraNr<params.size();paraNr++){
	func_.SetParameter(paraNr,params[paraNr]);
      }
    }
    float operator()(const ParamType& obj){
      return func_.Eval(obj.x);
    };
  };
  template<typename ParamType>
  class TF1Wrap<ParamType,false>{
  public:
    TF1Wrap(const std::string& funcExpr,const std::vector<double>& params){}
    float operator()(const ParamType& obj){return 1.;};
  };
    
  template<typename ParamType,bool=true>
  struct TF2Wrap{
  private:
    TF2 func_;
  public:
    TF2Wrap(const std::string& funcExpr,const std::vector<double>& params):
      func_("func",funcExpr.c_str()){
      for(size_t paraNr=0;paraNr<params.size();paraNr++){
	func_.SetParameter(paraNr,params[paraNr]);
      }
    }      
    float operator()(const ParamType& obj){
      return func_.Eval(obj.x,obj.y);
    };
  };
  template<typename ParamType>
  class TF2Wrap<ParamType,false>{
  public:
    TF2Wrap(const std::string& funcExpr,const std::vector<double>& params){}
    float operator()(const ParamType& obj){return 1.;};
  };

  template<typename ParamType,bool=true>
  struct TF3Wrap{
  private:
    TF3 func_;
  public:
    TF3Wrap(const std::string& funcExpr,const std::vector<double>& params):
      func_("func",funcExpr.c_str()){
      for(size_t paraNr=0;paraNr<params.size();paraNr++){
	func_.SetParameter(paraNr,params[paraNr]);
      }
    }      
    float operator()(const ParamType& obj){
      return func_.Eval(obj.x,obj.y,obj.z);
    };
  };
  template<typename ParamType>
  class TF3Wrap<ParamType,false>{
  public:
    TF3Wrap(const std::string& funcExpr,const std::vector<double>& params){}
    float operator()(const ParamType& obj){return 1.;};
  };
      


  template<typename T>
  constexpr auto has1D(int) -> decltype(T::x,bool()){return true;}
  template<typename T>
  constexpr bool has1D(...){return false;}
  template<typename T>
  constexpr auto has2D(int) -> decltype(T::y,bool()){return true;}
  template<typename T>
  constexpr bool has2D(...){return false;}
  template<typename T>
  constexpr auto has3D(int) -> decltype(T::z,bool()){return true;}
  template<typename T>
  constexpr bool has3D(...){return false;}

  template<typename ParamType>
  class ParamBin {
  public:
    ParamBin(){}
    virtual ~ParamBin(){}
    virtual bool pass(const ParamType&)const=0; 
    virtual float operator()(const ParamType&)const=0;
  protected:
    //the FUNCTYPE:=funcExpr is designed for future extensions
    static std::pair<std::string,std::string> readFuncStr(const std::string& inStr){
      std::cout <<"str "<<inStr<<" "<<inStr.substr(5)<<std::endl;
      size_t pos=inStr.find(":=");
      if(pos!=std::string::npos) return std::make_pair(inStr.substr(0,pos),inStr.substr(pos+2));
      else return std::make_pair(inStr,std::string(""));
    }
    std::function<float(const ParamType&)> makeFunc(const edm::ParameterSet& config){
      auto funcType = readFuncStr(config.getParameter<std::string>("funcType"));
      auto funcParams = config.getParameter<std::vector<double> >("funcParams");
      if(funcType.first=="TF1" && has1D<ParamType>(0)) return TF1Wrap<ParamType,has1D<ParamType>(0)>(funcType.second,funcParams);
      else if(funcType.first=="TF2" && has2D<ParamType>(0)) return TF2Wrap<ParamType,has2D<ParamType>(0)>(funcType.second,funcParams);
      else if(funcType.first=="TF3" && has3D<ParamType>(0)) return TF3Wrap<ParamType,has3D<ParamType>(0)>(funcType.second,funcParams);
      else throw cms::Exception("InvalidConfig") << " type "<<funcType.first<<" is not recognised, configuration is invalid and needs to be fixed"<<std::endl;
    }
  };

  template<typename ParamType>
  class ParamBin1D : ParamBin<ParamType> {
  private:
    using XType = decltype(ParamType::x);
    XType xMin_,xMax_;
    std::function<float(const ParamType&)> func_;
  public: 
    ParamBin1D(const edm::ParameterSet& config):
      xMin_(config.getParameter<typename ConfigType<XType>::type >("xMin")),
      xMax_(config.getParameter<typename ConfigType<XType>::type >("xMax")),
      func_(ParamBin<ParamType>::makeFunc(config))
    {
    }
    bool pass(const ParamType& seed)const override {
      return seed.pass(xMin_,xMax_);
    }
    float operator()(const ParamType& seed)const override{
      if(!pass(seed)) return 0;
      else return func_(seed);
    }
  };
    
    
    
  template<typename ParamType>
  class ParamBin2D : public ParamBin<ParamType> {
  private:
    using XType = decltype(ParamType::x);
    using YType = decltype(ParamType::y);
    XType xMin_,xMax_;
    YType yMin_,yMax_;
    std::function<float(const ParamType&)> func_;
  public:
    ParamBin2D(const edm::ParameterSet& config):
      xMin_(config.getParameter<typename ConfigType<XType>::type >("xMin")),
      xMax_(config.getParameter<typename ConfigType<XType>::type >("xMax")),
      yMin_(config.getParameter<typename ConfigType<YType>::type >("yMin")),
      yMax_(config.getParameter<typename ConfigType<YType>::type >("yMax")),
      func_(ParamBin<ParamType>::makeFunc(config))
    {
    }

    bool pass(const ParamType& seed)const override {
      return seed.pass(xMin_,xMax_,yMin_,yMax_);
    }  
    float operator()(const ParamType& seed)const override{
      if(!pass(seed)) return 0;
      else return func_(seed);
    }
  };
  

  template<typename ParamType>
  class ParamBin3D : ParamBin<ParamType> {
    using XType = decltype(ParamType::x);
    using YType = decltype(ParamType::y);
    using ZType = decltype(ParamType::z);

    XType xMin_,xMax_;
    YType yMin_,yMax_;
    ZType zMin_,zMax_;
    std::function<float(const ParamType&)> func_;
  public:
    ParamBin3D(const edm::ParameterSet& config):
      xMin_(config.getParameter<typename ConfigType<XType>::type >("xMin")),
      xMax_(config.getParameter<typename ConfigType<XType>::type >("xMax")),      
      yMin_(config.getParameter<typename ConfigType<YType>::type >("yMin")),
      yMax_(config.getParameter<typename ConfigType<YType>::type >("yMax")),
      zMin_(config.getParameter<typename ConfigType<ZType>::type >("zMin")),
      zMax_(config.getParameter<typename ConfigType<ZType>::type >("zMax")),
      func_(ParamBin<ParamType>::makeFunc(config))
    {
    }

    bool pass(const ParamType& seed)const override {
      return seed.pass(xMin_,xMax_,yMin_,yMax_,zMin_,zMax_);
    }  
    float operator()(const ParamType& seed)const override{
      if(!pass(seed)) return 0;
      else return func_(seed);
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
      if(type=="AbsEtaClus") return std::make_unique<ParamBin2D<AbsEtaNrClus>>(config);
      else throw cms::Exception("InvalidConfig") << " type "<<type<<" is not recognised, configuration is invalid and needs to be fixed"<<std::endl;
    }
  };
}
