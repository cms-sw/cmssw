// -*- C++ -*-
//

// user include files
#include "FWCore/Framework/src/EventSetupProviderMaker.h"

#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescriptionFillerPluginFactory.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <exception>
#include <string>

namespace edm {
  namespace eventsetup {
    // ---------------------------------------------------------------
    std::unique_ptr<EventSetupProvider> makeEventSetupProvider(ParameterSet const& params,
                                                               unsigned subProcessIndex,
                                                               ActivityRegistry* activityRegistry) {
      std::vector<std::string> prefers = params.getParameter<std::vector<std::string> >("@all_esprefers");

      if (prefers.empty()) {
        return std::make_unique<EventSetupProvider>(activityRegistry, subProcessIndex);
      }

      EventSetupProvider::PreferredProviderInfo preferInfo;
      EventSetupProvider::RecordToDataMap recordToData;

      //recordToData.insert(std::make_pair(std::string("DummyRecord"),
      //      std::make_pair(std::string("DummyData"), std::string())));
      //preferInfo[ComponentDescription("DummyESProductResolverProvider", "", false)]=
      //      recordToData;

      for (std::vector<std::string>::iterator itName = prefers.begin(), itNameEnd = prefers.end(); itName != itNameEnd;
           ++itName) {
        recordToData.clear();
        ParameterSet const& preferPSet = params.getParameterSet(*itName);
        std::vector<std::string> recordNames = preferPSet.getParameterNames();
        for (std::vector<std::string>::iterator itRecordName = recordNames.begin(), itRecordNameEnd = recordNames.end();
             itRecordName != itRecordNameEnd;
             ++itRecordName) {
          if ((*itRecordName)[0] == '@') {
            //this is a 'hidden parameter' so skip it
            continue;
          }

          //this should be a record name with its info
          try {
            std::vector<std::string> dataInfo = preferPSet.getParameter<std::vector<std::string> >(*itRecordName);

            if (dataInfo.empty()) {
              //FUTURE: empty should just mean all data
              throw Exception(errors::Configuration)
                  << "The record named " << *itRecordName << " specifies no data items";
            }
            //FUTURE: 'any' should be a special name
            for (std::vector<std::string>::iterator itDatum = dataInfo.begin(), itDatumEnd = dataInfo.end();
                 itDatum != itDatumEnd;
                 ++itDatum) {
              std::string datumName(*itDatum, 0, itDatum->find_first_of("/"));
              std::string labelName;

              if (itDatum->size() != datumName.size()) {
                labelName = std::string(*itDatum, datumName.size() + 1);
              }
              recordToData.insert(std::make_pair(std::string(*itRecordName), std::make_pair(datumName, labelName)));
            }
          } catch (cms::Exception const& iException) {
            cms::Exception theError("ESPreferConfigurationError");
            theError << "While parsing the es_prefer statement for type="
                     << preferPSet.getParameter<std::string>("@module_type") << " label=\""
                     << preferPSet.getParameter<std::string>("@module_label") << "\" an error occurred.";
            theError.append(iException);
            throw theError;
          }
        }
        preferInfo[ComponentDescription(preferPSet.getParameter<std::string>("@module_type"),
                                        preferPSet.getParameter<std::string>("@module_label"),
                                        ComponentDescription::unknownID(),
                                        false)] = recordToData;
      }
      return std::make_unique<EventSetupProvider>(activityRegistry, subProcessIndex, &preferInfo);
    }

    // ---------------------------------------------------------------
    void fillEventSetupProvider(ModuleTypeResolverMaker const* resolverMaker,
                                EventSetupsController& esController,
                                EventSetupProvider& cp,
                                ParameterSet& params) {
      std::vector<std::string> providers = params.getParameter<std::vector<std::string> >("@all_esmodules");

      for (std::vector<std::string>::iterator itName = providers.begin(), itNameEnd = providers.end();
           itName != itNameEnd;
           ++itName) {
        ParameterSet* providerPSet = params.getPSetForUpdate(*itName);
        ModuleFactory::get()->addTo(esController, cp, *providerPSet, resolverMaker);
      }

      std::vector<std::string> sources = params.getParameter<std::vector<std::string> >("@all_essources");

      for (std::vector<std::string>::iterator itName = sources.begin(), itNameEnd = sources.end(); itName != itNameEnd;
           ++itName) {
        ParameterSet* providerPSet = params.getPSetForUpdate(*itName);
        SourceFactory::get()->addTo(esController, cp, *providerPSet, resolverMaker);
      }
    }
  }  // namespace eventsetup
}  // namespace edm
