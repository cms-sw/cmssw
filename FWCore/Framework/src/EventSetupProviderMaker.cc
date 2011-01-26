// -*- C++ -*-
//

// user include files
#include "FWCore/Framework/interface/EventSetupProviderMaker.h"

#include "FWCore/Framework/interface/CommonParams.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <string>

namespace edm {
  namespace eventsetup {
  // ---------------------------------------------------------------
    std::auto_ptr<EventSetupProvider>
    makeEventSetupProvider(ParameterSet const& params) {
      std::vector<std::string> prefers =
        params.getParameter<std::vector<std::string> >("@all_esprefers");

      if(prefers.empty()) {
        return std::auto_ptr<EventSetupProvider>(new EventSetupProvider());
      }

      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;

      //recordToData.insert(std::make_pair(std::string("DummyRecord"),
      //      std::make_pair(std::string("DummyData"), std::string())));
      //preferInfo[ComponentDescription("DummyProxyProvider", "", false)]=
      //      recordToData;

      for(std::vector<std::string>::iterator itName = prefers.begin(), itNameEnd = prefers.end();
          itName != itNameEnd;
          ++itName) {
        recordToData.clear();
        ParameterSet const& preferPSet = params.getParameterSet(*itName);
        std::vector<std::string> recordNames = preferPSet.getParameterNames();
        for(std::vector<std::string>::iterator itRecordName = recordNames.begin(),
            itRecordNameEnd = recordNames.end();
            itRecordName != itRecordNameEnd;
            ++itRecordName) {

          if((*itRecordName)[0] == '@') {
            //this is a 'hidden parameter' so skip it
            continue;
          }

          //this should be a record name with its info
          try {
            std::vector<std::string> dataInfo =
              preferPSet.getParameter<std::vector<std::string> >(*itRecordName);

            if(dataInfo.empty()) {
              //FUTURE: empty should just mean all data
              throw Exception(errors::Configuration)
                << "The record named "
                << *itRecordName << " specifies no data items";
            }
            //FUTURE: 'any' should be a special name
            for(std::vector<std::string>::iterator itDatum = dataInfo.begin(),
                itDatumEnd = dataInfo.end();
                itDatum != itDatumEnd;
                ++itDatum){
              std::string datumName(*itDatum, 0, itDatum->find_first_of("/"));
              std::string labelName;

              if(itDatum->size() != datumName.size()) {
                labelName = std::string(*itDatum, datumName.size() + 1);
              }
              recordToData.insert(std::make_pair(std::string(*itRecordName),
                                                 std::make_pair(datumName,
                                                                labelName)));
            }
          } catch(cms::Exception const& iException) {
            cms::Exception theError("ESPreferConfigurationError");
            theError << "While parsing the es_prefer statement for type="
                     << preferPSet.getParameter<std::string>("@module_type")
                     << " label=\""
                     << preferPSet.getParameter<std::string>("@module_label")
                     << "\" an error occurred.";
            theError.append(iException);
            throw theError;
          }
        }
        preferInfo[ComponentDescription(preferPSet.getParameter<std::string>("@module_type"),
                                        preferPSet.getParameter<std::string>("@module_label"),
                                        false)] = recordToData;
      }
      return std::auto_ptr<EventSetupProvider>(new EventSetupProvider(&preferInfo));
    }

    // ---------------------------------------------------------------
    void
    fillEventSetupProvider(EventSetupProvider& cp,
                           ParameterSet& params,
                           CommonParams const& common) {
      std::vector<std::string> providers =
        params.getParameter<std::vector<std::string> >("@all_esmodules");

      for(std::vector<std::string>::iterator itName = providers.begin(), itNameEnd = providers.end();
          itName != itNameEnd;
          ++itName) {
        ParameterSet* providerPSet = params.getPSetForUpdate(*itName);
        validateEventSetupParameters(*providerPSet);
        providerPSet->registerIt();
        ModuleFactory::get()->addTo(cp,
                                    *providerPSet,
                                    common.processName_,
                                    common.releaseVersion_,
                                    common.passID_);
      }

      std::vector<std::string> sources =
        params.getParameter<std::vector<std::string> >("@all_essources");

      for(std::vector<std::string>::iterator itName = sources.begin(), itNameEnd = sources.end();
          itName != itNameEnd;
          ++itName) {
        ParameterSet* providerPSet = params.getPSetForUpdate(*itName);
        validateEventSetupParameters(*providerPSet);
        providerPSet->registerIt();
        SourceFactory::get()->addTo(cp,
                                    *providerPSet,
                                    common.processName_,
                                    common.releaseVersion_,
                                    common.passID_);
      }
    }

    // ---------------------------------------------------------------
    void validateEventSetupParameters(ParameterSet & pset) {
      std::string modtype;
      std::string moduleLabel;
      try {
        modtype = pset.getParameter<std::string>("@module_type");
        moduleLabel = pset.getParameter<std::string>("@module_label");
        // Check for the "unlabeled" case
        // This is an artifact left over from the old configuration language
	// we were using before switching to the python configuration
        // This is handled in the validation code and python configuration
        // files by using a label equal to the module typename.
        if (moduleLabel == std::string("")) {
          moduleLabel = modtype;
        }

        std::auto_ptr<ParameterSetDescriptionFillerBase> filler(
          ParameterSetDescriptionFillerPluginFactory::get()->create(modtype));
        ConfigurationDescriptions descriptions(filler->baseType());
        filler->fill(descriptions);
        descriptions.validate(pset, moduleLabel);
      }
      catch (cms::Exception& iException) {
        Exception toThrow(errors::Configuration, "Failed validating configuration of ESProducer or ESSource.");
        toThrow << "\nThe plugin name is \"" << modtype << "\"\n";
        toThrow << "The module label is \"" << moduleLabel << "\"\n";
        toThrow.append(iException);
        throw toThrow;
      }
    }
  }
}
