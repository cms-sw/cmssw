#ifndef FWCore_ParameterSet_PluginDescription_h
#define FWCore_ParameterSet_PluginDescription_h
// -*- C++ -*-
//
// Package:     FWCore/ParameterSet
// Class  :     PluginDescription
// 
/**\class PluginDescription PluginDescription.h "PluginDescription.h"

 Description: Use to describe how to validate a plugin that will be loaded

 Usage:
 User defined plugins which are constructed by passing a edm::ParameterSet can
 have their parameters validated using the \c edm::PluginDescription. For this to
 work, one must
 
 1) Plugin configuration
 All the parameters used by the Plugin must be contained within one PSet, this
 also includes an cms.string parameter containing the name of the actual plugin
 type to be loaded. E.g. if a module \c FooProd loaded a plugin of specific type
 \c BarHelper and \c BarHelper uses the parameter named \c value :
 \code{.py}
 foo = cms.EDProducer("FooProd",
                      pluginDescription = cms.PSet(type = cms.string("BarHelper"),
                                                   value = cms.int32(5) ) )
 \endcode

2) Plugin fillPSetDescription
Each user plugin must define a static member function:
 \code{.cpp}
 static void fillPSetDescription(edm::ParameterSetDescription const&);
 \endcode

 The function fills the \c edm::ParameterSetDescription with the descriptions of
 the parameters necessary for that particular plugin. Note that the parameter
 used by the module to find the type of the plugin should NOT be declared in
 the \c edm::ParameterSetDescription since it will be added by the
 \c edm::PluginDescription itself. E.g.
 
 \code{.cpp}
 void BarHelper::fillPSetDescription(edm::ParameterSetDescription& iPSD) {
    iPSD.add<int>("value", 5);
 }
 \endcode
 
 3) Module's fillDescriptions
 The module which uses the plugins must use the \c edm::PluginDescription within its
 \c fillDescriptions static member function.
 The \c edm::PluginDescription object is attached to the \c edm::ParameterSetDescription
 object which is being used to represent the PSet for the plugin. In turn, that
 \c edm::ParamterSetDescription is attached to the module's top level \edm::ParameterSetDescription.
 This exactly mimics what would be done if the plugin's parameters were directly known to
 the module.
 
 \code{.cpp}
 void FooProd::fillDescriptions(ConfigurationDescriptions& oDesc) {
   //specify how to get the plugin validation
   ParameterSetDescription pluginDesc;
   //'type' is the label to the string containing which plugin to load
   pluginDesc.addNode(edm::PluginDescription<HelperFactory>("type") );
 
   ParameterSetDescription desc;
   desc.add<ParameterSetDescription>("pluginDescription", pluginDesc);
 
   oDesc.addDefault(desc);
 }
 \endcode
 
 4) Factory registration
 Use \c EDM_REGISTER_VALIDATED_PLUGINFACTORY rather than
 \c EDM_REGISTER_PLUGINFACTORY. This new macro can be found in
 FWCore/ParameterSet/interface/ValidatedPluginFactoryMacros.h
 
 5) Plugin registration
 Use \c DEFINE_EDM_VALIDATED_PLUGIN rather than \c DEFINE_EDM_PLUGIN.
 This new macro can be found in FWCore/Framework/interface/ValidatedPluginMacros.h
 
 */
//
// Original Author:  Chris Jones
//         Created:  Wed, 19 Sep 2018 19:23:27 GMT
//

// system include files
#include "FWCore/ParameterSet/interface/ParameterDescriptionNode.h"
#include "FWCore/ParameterSet/interface/PluginDescriptionAdaptorBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

// user include files
#include <string>

// forward declarations
namespace edm {
template< typename T>
  class PluginDescription : public ParameterDescriptionNode
{
public:
  /**Constructor without a default for typeLabel
   @param[in] typeLabel the label for the std::string parameter which holds the plugin type to be loaded
   @param[in] typeLabelIsTracked 'true' if the parameter `typeLabel` is tracked, else should be false
   */
  PluginDescription(std::string typeLabel, bool typeLabelIsTracked):
  typeLabel_{std::move(typeLabel)},
  typeLabelIsTracked_{typeLabelIsTracked} {}

  /**Constructor with a default for typeLabel
   @param[in] typeLabel the label for the std::string parameter which holds the plugin type to be loaded
   @param[in] defaultType the default plugin type that should be loaded if no type is given
   @param[in] typeLabelIsTracked 'true' if the parameter `typeLabel` is tracked, else should be false
   */
  PluginDescription(std::string typeLabel, std::string defaultType, bool typeLabelIsTracked):
  typeLabel_{std::move(typeLabel)},
  defaultType_{std::move(defaultType)},
  typeLabelIsTracked_{typeLabelIsTracked} {}

  // ---------- const member functions ---------------------
  ParameterDescriptionNode* clone() const final {
    return new PluginDescription<T>(*this);
  }

protected:
  
  void checkAndGetLabelsAndTypes_(std::set<std::string>& usedLabels,
                                  std::set<ParameterTypes>& parameterTypes,
                                  std::set<ParameterTypes>& wildcardTypes) const final {
    
  }
  
  
  
  void validate_(ParameterSet& pset,
                 std::set<std::string>& validatedLabels,
                 bool optional) const final {
    loadPlugin(findType(pset));
    cache_->validate(pset);
    //all names are good
    auto n =pset.getParameterNames();
    validatedLabels.insert(n.begin(),n.end());
  }
  
  void writeCfi_(std::ostream& os,
                 bool& startWithComma,
                 int indentation,
                 bool& wroteSomething) const final {
    if(not defaultType_.empty()) {
      if (!edmplugin::PluginManager::isAvailable()) {
        auto conf = edmplugin::standard::config();
        conf.allowNoCache();
        edmplugin::PluginManager::configure(conf);
      }

      loadPlugin(defaultType_);
      
      cache_->writeCfi(os,startWithComma,indentation);
      wroteSomething = true;
    }
  }
  
  bool hasNestedContent_() const final {
    return true;
  }
  
  void printNestedContent_(std::ostream& os,
                           bool /*optional*/,
                           DocFormatHelper& dfh) const final {
    int indentation = dfh.indentation();

    using CreatedType = PluginDescriptionAdaptorBase<typename T::CreatedType>;
    using Factory = edmplugin::PluginFactory<CreatedType*()>;

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    printSpaces(os, indentation);
    os << "Section " << newSection
    << " " << Factory::get()->category() << " Plugins description:\n";
    if(!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.init();
    new_dfh.setSection(newSection);

    //loop over all possible plugins
    unsigned int pluginCount = 0;
    std::string previousName;
    for(auto const& info: edmplugin::PluginManager::get()->categoryToInfos().find(Factory::get()->category())->second) {

      // We only want to print the first instance of each plugin name
      if (previousName == info.name_) {
        continue;
      }

      std::stringstream ss;
      ss << dfh.section() << "." << dfh.counter();
      std::string newSection = ss.str();
      printSpaces(os, indentation);
      os << "Section " << newSection <<"."<< ++pluginCount
      << " " << info.name_ << " Plugin description:\n";
      if(!dfh.brief()) os << "\n";

      DocFormatHelper new_dfh(dfh);
      new_dfh.init();
      new_dfh.setSection(newSection);

      loadDescription(info.name_)->print(os,new_dfh);

      previousName = info.name_;
    }
  }
  
  bool exists_(ParameterSet const& pset) const final {
    return pset.existsAs<std::string>(typeLabel_, typeLabelIsTracked_);
  }
  
  bool partiallyExists_(ParameterSet const& pset) const final {
    return exists_(pset);
  }
  
  int howManyXORSubNodesExist_(ParameterSet const& pset) const final {
    return exists(pset) ? 1 : 0;
  }

private:

  std::string findType(edm::ParameterSet const& iPSet) const {
    if(typeLabelIsTracked_) {
      if(iPSet.existsAs<std::string>(typeLabel_) || defaultType_.empty()) {
        return iPSet.getParameter<std::string>(typeLabel_);
      } else {
        return defaultType_;
      }
    }
    if(defaultType_.empty()) {
      return iPSet.getUntrackedParameter<std::string>(typeLabel_);
    }
    return iPSet.getUntrackedParameter<std::string>(typeLabel_,defaultType_);
  }
  
  void loadPlugin(std::string const& iName) const {
    if(not cache_) {
      cache_ = loadDescription(iName);
    }
  }
  
  std::shared_ptr<ParameterSetDescription>
  loadDescription(std::string const& iName) const {
    using CreatedType = PluginDescriptionAdaptorBase<typename T::CreatedType>;
    std::unique_ptr<CreatedType> a(edmplugin::PluginFactory<CreatedType*()>::get()->create(iName));
    
    std::shared_ptr<ParameterSetDescription> desc = std::make_shared<ParameterSetDescription>(a->description());
    
    //There is no way to check to see if a node already wants a label
    if(typeLabelIsTracked_) {
      if(defaultType_.empty()) {
        desc->add<std::string>(typeLabel_);
      } else {
        desc->add<std::string>(typeLabel_,defaultType_);
      }
    } else {
      if(defaultType_.empty()) {
        desc->addUntracked<std::string>(typeLabel_);
      } else {
        desc->addUntracked<std::string>(typeLabel_,defaultType_);
      }
    }
    return desc;
  }

  // ---------- member data --------------------------------
  mutable std::shared_ptr<ParameterSetDescription> cache_;
  std::string typeLabel_;
  std::string defaultType_;
  bool typeLabelIsTracked_;
  
};
}

#endif
