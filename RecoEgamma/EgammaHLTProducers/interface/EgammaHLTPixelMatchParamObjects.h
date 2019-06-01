//These objects allow an arbitary parameristisation to be used
//design:
//    the function and parameterisation can be changing without having to change the interface
//    or breaking backwards compatibiltiy with existing configs.
//    This is vital for the HLT and the main driving force behind the design which otherwise
//    could have been a lot simplier
//
//usage:
//    The variables used are defined by an intermediate object which defines the
//    variables x,y, and z. 1D objects only need to define x, 2D objects x,y, etc.
//    Example is "AbsEtaNrClus" which defines x as |supercluster eta| and y as #subclusters of the supercluster.
//    The object also defines a pass function where it determines if x (and y,z) is
//    within the specificed xmin and xmax range. This is mainly done as individual
//    objects can decide whether this means min<=x<max or min<=x<=max
//
//    These objects are used by the binning objects. The bins can currently be 1D, 2D,
//    or 3D based on the intermediate object. Each bin has a function with it.
//    The function takes a ParmaType as an argument and returns a float.
//    The function is defined by a string of format FUNCID:=ExtraConfigInfo
//    FUNCID = function type which allows different functions to be selected
//    while ExtraConfigInfo is any extra information needed to configure that func
//    currently implimented types are TF1, TF2, TF3
//    so to get a TF1 which is a pol3 just do TF1:=pol3
//
//future plans:
//    Seperate out intermediate wrapper objects such as AbsEtaNrClus
//    and put them in their own file. However some mechanism is needed to register them
//    so for now this isnt done. Note we might move to the standard CMSSW of dealing
//    with this

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TF1.h"
#include "TF2.h"
#include "TF3.h"

namespace egPM {

  struct AbsEtaNrClus {
    float x;
    size_t y;

    AbsEtaNrClus(const reco::ElectronSeed& seed) {
      reco::SuperClusterRef scRef = seed.caloCluster().castTo<reco::SuperClusterRef>();
      x = std::abs(scRef->eta());
      y = scRef->clustersSize();
    }
    bool pass(float absEtaMin, float absEtaMax, size_t nrClusMin, size_t nrClusMax) const {
      return x >= absEtaMin && x < absEtaMax && y >= nrClusMin && y <= nrClusMax;
    }
  };
  struct AbsEtaNrClusPhi {
    float x;
    size_t y;
    float z;

    AbsEtaNrClusPhi(const reco::ElectronSeed& seed) {
      reco::SuperClusterRef scRef = seed.caloCluster().castTo<reco::SuperClusterRef>();
      x = std::abs(scRef->eta());
      y = scRef->clustersSize();
      z = scRef->phi();
    }
    bool pass(float absEtaMin, float absEtaMax, size_t nrClusMin, size_t nrClusMax, float phiMin, float phiMax) const {
      return x >= absEtaMin && x < absEtaMax && y >= nrClusMin && y <= nrClusMax && z >= phiMin && z < phiMax;
    }
  };

  struct AbsEtaNrClusEt {
    float x;
    size_t y;
    float z;

    AbsEtaNrClusEt(const reco::ElectronSeed& seed) {
      reco::SuperClusterRef scRef = seed.caloCluster().castTo<reco::SuperClusterRef>();
      x = std::abs(scRef->eta());
      y = scRef->clustersSize();
      z = scRef->energy() * sin(scRef->position().Theta());
    }
    bool pass(float absEtaMin, float absEtaMax, size_t nrClusMin, size_t nrClusMax, float etMin, float etMax) const {
      return x >= absEtaMin && x < absEtaMax && y >= nrClusMin && y <= nrClusMax && z >= etMin && z < etMax;
    }
  };

  //these structs wrap the TF1 object
  //also if the ParamType doesnt have a high enough dimension
  //(ie using only one with x,y for a TF3), then the second
  //template parameter disables the function
  template <typename ParamType, bool = true>
  struct TF1Wrap {
  private:
    TF1 func_;

  public:
    TF1Wrap(const std::string& funcExpr, const std::vector<double>& params) : func_("func", funcExpr.c_str()) {
      for (size_t paraNr = 0; paraNr < params.size(); paraNr++) {
        func_.SetParameter(paraNr, params[paraNr]);
      }
    }
    float operator()(const ParamType& obj) { return func_.Eval(obj.x); };
  };
  template <typename ParamType>
  class TF1Wrap<ParamType, false> {
  public:
    TF1Wrap(const std::string& funcExpr, const std::vector<double>& params) {}
    float operator()(const ParamType& obj) { return 1.; };
  };

  template <typename ParamType, bool = true>
  struct TF2Wrap {
  private:
    TF2 func_;

  public:
    TF2Wrap(const std::string& funcExpr, const std::vector<double>& params) : func_("func", funcExpr.c_str()) {
      for (size_t paraNr = 0; paraNr < params.size(); paraNr++) {
        func_.SetParameter(paraNr, params[paraNr]);
      }
    }
    float operator()(const ParamType& obj) { return func_.Eval(obj.x, obj.y); };
  };
  template <typename ParamType>
  class TF2Wrap<ParamType, false> {
  public:
    TF2Wrap(const std::string& funcExpr, const std::vector<double>& params) {}
    float operator()(const ParamType& obj) { return 1.; };
  };

  template <typename ParamType, bool = true>
  struct TF3Wrap {
  private:
    TF3 func_;

  public:
    TF3Wrap(const std::string& funcExpr, const std::vector<double>& params) : func_("func", funcExpr.c_str()) {
      for (size_t paraNr = 0; paraNr < params.size(); paraNr++) {
        func_.SetParameter(paraNr, params[paraNr]);
      }
    }
    float operator()(const ParamType& obj) { return func_.Eval(obj.x, obj.y, obj.z); };
  };
  template <typename ParamType>
  class TF3Wrap<ParamType, false> {
  public:
    TF3Wrap(const std::string& funcExpr, const std::vector<double>& params) {}
    float operator()(const ParamType& obj) { return 1.; };
  };

  //the following functions allow for the fact that the type in the CMSSW PSet does not
  //have floats do when it sees a float parameter, it retrieves a double
  template <typename T>
  struct ConfigType {
    typedef T type;
  };
  template <>
  struct ConfigType<float> {
    typedef double type;
  };
  template <>
  struct ConfigType<size_t> {
    typedef int type;
  };

  //helper functions to figure out what dimension the ParamType is
  //and not generate functions which require a higher dimension
  template <typename T>
  constexpr auto has1D(int) -> decltype(T::x, bool()) {
    return true;
  }
  template <typename T>
  constexpr bool has1D(...) {
    return false;
  }
  template <typename T>
  constexpr auto has2D(int) -> decltype(T::y, bool()) {
    return true;
  }
  template <typename T>
  constexpr bool has2D(...) {
    return false;
  }
  template <typename T>
  constexpr auto has3D(int) -> decltype(T::z, bool()) {
    return true;
  }
  template <typename T>
  constexpr bool has3D(...) {
    return false;
  }

  template <typename InputType>
  class ParamBin {
  public:
    ParamBin() {}
    virtual ~ParamBin() {}
    virtual bool pass(const InputType&) const = 0;
    virtual float operator()(const InputType&) const = 0;

  protected:
    //the FUNCTYPE:=funcExpr is designed for future extensions
    static std::pair<std::string, std::string> readFuncStr(const std::string& inStr) {
      size_t pos = inStr.find(":=");
      if (pos != std::string::npos)
        return std::make_pair(inStr.substr(0, pos), inStr.substr(pos + 2));
      else
        return std::make_pair(inStr, std::string(""));
    }
    template <typename ParamType>
    static std::function<float(const ParamType&)> makeFunc(const edm::ParameterSet& config) {
      auto funcType = readFuncStr(config.getParameter<std::string>("funcType"));
      auto funcParams = config.getParameter<std::vector<double>>("funcParams");
      if (funcType.first == "TF1" && has1D<ParamType>(0))
        return TF1Wrap<ParamType, has1D<ParamType>(0)>(funcType.second, funcParams);
      else if (funcType.first == "TF2" && has2D<ParamType>(0))
        return TF2Wrap<ParamType, has2D<ParamType>(0)>(funcType.second, funcParams);
      else if (funcType.first == "TF3" && has3D<ParamType>(0))
        return TF3Wrap<ParamType, has3D<ParamType>(0)>(funcType.second, funcParams);
      else
        throw cms::Exception("InvalidConfig") << " type " << funcType.first
                                              << " is not recognised or is imcompatable with the ParamType, "
                                                 "configuration is invalid and needs to be fixed"
                                              << std::endl;
    }
  };

  template <typename InputType, typename ParamType>
  class ParamBin1D : public ParamBin<InputType> {
  private:
    using XType = decltype(ParamType::x);
    XType xMin_, xMax_;
    std::function<float(const ParamType&)> func_;

  public:
    ParamBin1D(const edm::ParameterSet& config)
        : xMin_(config.getParameter<typename ConfigType<XType>::type>("xMin")),
          xMax_(config.getParameter<typename ConfigType<XType>::type>("xMax")),
          func_(ParamBin<InputType>::template makeFunc<ParamType>(config)) {}
    bool pass(const InputType& input) const override { return ParamType(input).pass(xMin_, xMax_); }
    float operator()(const InputType& input) const override {
      if (!pass(input))
        return 0;
      else
        return func_(ParamType(input));
    }
  };

  template <typename InputType, typename ParamType>
  class ParamBin2D : public ParamBin<InputType> {
  private:
    using XType = decltype(ParamType::x);
    using YType = decltype(ParamType::y);
    XType xMin_, xMax_;
    YType yMin_, yMax_;
    std::function<float(const ParamType&)> func_;

  public:
    ParamBin2D(const edm::ParameterSet& config)
        : xMin_(config.getParameter<typename ConfigType<XType>::type>("xMin")),
          xMax_(config.getParameter<typename ConfigType<XType>::type>("xMax")),
          yMin_(config.getParameter<typename ConfigType<YType>::type>("yMin")),
          yMax_(config.getParameter<typename ConfigType<YType>::type>("yMax")),
          func_(ParamBin<InputType>::template makeFunc<ParamType>(config)) {}

    bool pass(const InputType& input) const override { return ParamType(input).pass(xMin_, xMax_, yMin_, yMax_); }
    float operator()(const InputType& input) const override {
      if (!pass(input))
        return 0;
      else
        return func_(ParamType(input));
    }
  };

  template <typename InputType, typename ParamType>
  class ParamBin3D : public ParamBin<InputType> {
    using XType = decltype(ParamType::x);
    using YType = decltype(ParamType::y);
    using ZType = decltype(ParamType::z);

    XType xMin_, xMax_;
    YType yMin_, yMax_;
    ZType zMin_, zMax_;
    std::function<float(const ParamType&)> func_;

  public:
    ParamBin3D(const edm::ParameterSet& config)
        : xMin_(config.getParameter<typename ConfigType<XType>::type>("xMin")),
          xMax_(config.getParameter<typename ConfigType<XType>::type>("xMax")),
          yMin_(config.getParameter<typename ConfigType<YType>::type>("yMin")),
          yMax_(config.getParameter<typename ConfigType<YType>::type>("yMax")),
          zMin_(config.getParameter<typename ConfigType<ZType>::type>("zMin")),
          zMax_(config.getParameter<typename ConfigType<ZType>::type>("zMax")),
          func_(ParamBin<InputType>::template makeFunc<ParamType>(config)) {}

    bool pass(const InputType& input) const override {
      return ParamType(input).pass(xMin_, xMax_, yMin_, yMax_, zMin_, zMax_);
    }
    float operator()(const InputType& input) const override {
      if (!pass(input))
        return 0;
      else
        return func_(ParamType(input));
    }
  };

  template <typename InputType>
  class Param {
    std::vector<std::unique_ptr<ParamBin<InputType>>> bins_;

  public:
    Param(const edm::ParameterSet& config) {
      std::vector<edm::ParameterSet> binConfigs = config.getParameter<std::vector<edm::ParameterSet>>("bins");
      for (auto& binConfig : binConfigs)
        bins_.emplace_back(createParamBin_(binConfig));
    }
    float operator()(const InputType& input) const {
      for (auto& bin : bins_) {
        if (bin->pass(input))
          return (*bin)(input);
      }
      return -1;  //didnt find a suitable bin, just return -1 for now
    }

  private:
    std::unique_ptr<ParamBin<InputType>> createParamBin_(const edm::ParameterSet& config) {
      std::string type = config.getParameter<std::string>("binType");
      if (type == "AbsEtaClus")
        return std::make_unique<ParamBin2D<InputType, AbsEtaNrClus>>(config);
      else if (type == "AbsEtaClusPhi")
        return std::make_unique<ParamBin3D<InputType, AbsEtaNrClusPhi>>(config);
      else if (type == "AbsEtaClusEt")
        return std::make_unique<ParamBin3D<InputType, AbsEtaNrClusEt>>(config);
      else
        throw cms::Exception("InvalidConfig")
            << " type " << type << " is not recognised, configuration is invalid and needs to be fixed" << std::endl;
    }
  };
}  // namespace egPM
