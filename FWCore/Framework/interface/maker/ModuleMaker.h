#ifndef FWCore_Framework_ModuleMaker_h
#define FWCore_Framework_ModuleMaker_h

#include <cassert>
#include <memory>
#include <string>

#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/maker/MakeModuleParams.h"
#include "FWCore/Framework/interface/maker/ModuleHolder.h"
#include "FWCore/Framework/interface/maker/MakeModuleHelper.h"

#include "FWCore/Utilities/interface/Signal.h"

namespace edm {
  class ConfigurationDescriptions;
  class ModuleDescription;
  class ParameterSet;
  class ExceptionToActionTable;

  class ModuleMakerBase {
  public:
    virtual ~ModuleMakerBase();
    std::shared_ptr<maker::ModuleHolder> makeModule(MakeModuleParams const&,
                                                    signalslot::Signal<void(ModuleDescription const&)>& iPre,
                                                    signalslot::Signal<void(ModuleDescription const&)>& iPost) const;

    std::shared_ptr<maker::ModuleHolder> makeReplacementModule(edm::ParameterSet const& p) const {
      return makeModule(p);
    }

  protected:
    ModuleDescription createModuleDescription(MakeModuleParams const& p) const;

    void throwConfigurationException(ModuleDescription const& md, cms::Exception& iException) const;

    void throwValidationException(MakeModuleParams const& p, cms::Exception& iException) const;

    void validateEDMType(std::string const& edmType, MakeModuleParams const& p) const;

  private:
    virtual void fillDescriptions(ConfigurationDescriptions& iDesc) const = 0;
    virtual std::shared_ptr<maker::ModuleHolder> makeModule(edm::ParameterSet const& p) const = 0;
    virtual const std::string& baseType() const = 0;
  };

  template <class T>
  class ModuleMaker : public ModuleMakerBase {
  public:
    //typedef T worker_type;
    explicit ModuleMaker();

  private:
    void fillDescriptions(ConfigurationDescriptions& iDesc) const override;
    std::shared_ptr<maker::ModuleHolder> makeModule(edm::ParameterSet const& p) const override;
    const std::string& baseType() const override;
  };

  template <class T>
  ModuleMaker<T>::ModuleMaker() {}

  template <class T>
  void ModuleMaker<T>::fillDescriptions(ConfigurationDescriptions& iDesc) const {
    T::fillDescriptions(iDesc);
    T::prevalidate(iDesc);
  }

  template <class T>
  std::shared_ptr<maker::ModuleHolder> ModuleMaker<T>::makeModule(edm::ParameterSet const& p) const {
    typedef T UserType;
    typedef typename UserType::ModuleType ModuleType;
    typedef MakeModuleHelper<ModuleType> MakerHelperType;

    return std::shared_ptr<maker::ModuleHolder>(
        std::make_shared<maker::ModuleHolderT<ModuleType> >(MakerHelperType::template makeModule<UserType>(p)));
  }

  template <class T>
  const std::string& ModuleMaker<T>::baseType() const {
    return T::baseType();
  }

}  // namespace edm

#endif
