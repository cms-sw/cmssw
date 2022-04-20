#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "processEDAliases.h"

#include <map>

namespace edm {
  namespace {
    void checkAndInsertAlias(std::string const& friendlyClassName,
                             std::string const& moduleLabel,
                             std::string const& productInstanceName,
                             std::string const& processName,
                             std::string const& alias,
                             std::string const& instanceAlias,
                             ProductRegistry const& preg,
                             std::multimap<BranchKey, BranchKey>& aliasMap,
                             std::map<BranchKey, BranchKey>& aliasKeys) {
      std::string const star("*");

      BranchKey key(friendlyClassName, moduleLabel, productInstanceName, processName);
      if (preg.productList().find(key) == preg.productList().end()) {
        // No product was found matching the alias.
        // We throw an exception only if a module with the specified module label was created in this process.
        for (auto const& product : preg.productList()) {
          if (moduleLabel == product.first.moduleLabel() && processName == product.first.processName()) {
            throw Exception(errors::Configuration, "EDAlias does not match data\n")
                << "There are no products of type '" << friendlyClassName << "'\n"
                << "with module label '" << moduleLabel << "' and instance name '" << productInstanceName << "'.\n";
          }
        }
      }

      if (auto iter = aliasMap.find(key); iter != aliasMap.end()) {
        // If the same EDAlias defines multiple products pointing to the same product, throw
        if (iter->second.moduleLabel() == alias) {
          throw Exception(errors::Configuration, "EDAlias conflict\n")
              << "The module label alias '" << alias << "' is used for multiple products of type '" << friendlyClassName
              << "' with module label '" << moduleLabel << "' and instance name '" << productInstanceName
              << "'. One alias has the instance name '" << iter->first.productInstanceName()
              << "' and the other has the instance name '" << instanceAlias << "'.";
        }
      }

      std::string const& theInstanceAlias(instanceAlias == star ? productInstanceName : instanceAlias);
      BranchKey aliasKey(friendlyClassName, alias, theInstanceAlias, processName);
      if (auto it = preg.productList().find(aliasKey); it != preg.productList().end()) {
        // We might have already inserted an alias that was a chosen case of a SwitchProducer
        if (not it->second.isAlias()) {
          throw Exception(errors::Configuration, "EDAlias conflicts with data\n")
              << "A product of type '" << friendlyClassName << "'\n"
              << "with module label '" << alias << "' and instance name '" << theInstanceAlias << "'\n"
              << "already exists.\n";
        }
        return;
      }
      auto iter = aliasKeys.find(aliasKey);
      if (iter != aliasKeys.end()) {
        // The alias matches a previous one.  If the same alias is used for different product, throw.
        if (iter->second != key) {
          throw Exception(errors::Configuration, "EDAlias conflict\n")
              << "The module label alias '" << alias << "' and product instance alias '" << theInstanceAlias << "'\n"
              << "are used for multiple products of type '" << friendlyClassName << "'\n"
              << "One has module label '" << moduleLabel << "' and product instance name '" << productInstanceName
              << "',\n"
              << "the other has module label '" << iter->second.moduleLabel() << "' and product instance name '"
              << iter->second.productInstanceName() << "'.\n";
        }
      } else {
        auto prodIter = preg.productList().find(key);
        if (prodIter != preg.productList().end()) {
          if (!prodIter->second.produced()) {
            throw Exception(errors::Configuration, "EDAlias\n")
                << "The module label alias '" << alias << "' and product instance alias '" << theInstanceAlias << "'\n"
                << "are used for a product of type '" << friendlyClassName << "'\n"
                << "with module label '" << moduleLabel << "' and product instance name '" << productInstanceName
                << "',\n"
                << "An EDAlias can only be used for products produced in the current process. This one is not.\n";
          }
          aliasMap.insert(std::make_pair(key, aliasKey));
          aliasKeys.insert(std::make_pair(aliasKey, key));
        }
      }
    }
  }  // namespace

  namespace detail {
    void processEDAliases(std::vector<std::string> const& aliasNamesToProcess,
                          std::unordered_set<std::string> const& aliasModulesToProcess,
                          ParameterSet const& proc_pset,
                          std::string const& processName,
                          ProductRegistry& preg) {
      if (aliasNamesToProcess.empty()) {
        return;
      }
      std::string const star("*");
      std::string const empty("");
      ParameterSetDescription desc;
      desc.add<std::string>("type");
      desc.add<std::string>("fromProductInstance", star);
      desc.add<std::string>("toProductInstance", star);

      std::multimap<BranchKey, BranchKey> aliasMap;

      std::map<BranchKey, BranchKey> aliasKeys;  // Used to search for duplicates or clashes.

      // Auxiliary search structure to support wildcard for friendlyClassName
      std::multimap<std::string, BranchKey> moduleLabelToBranches;
      for (auto const& prod : preg.productList()) {
        if (processName == prod.second.processName()) {
          moduleLabelToBranches.emplace(prod.first.moduleLabel(), prod.first);
        }
      }

      // Now, loop over the alias information and store it in aliasMap.
      for (std::string const& alias : aliasNamesToProcess) {
        ParameterSet const& aliasPSet = proc_pset.getParameterSet(alias);
        std::vector<std::string> vPSetNames = aliasPSet.getParameterNamesForType<VParameterSet>();
        for (std::string const& moduleLabel : vPSetNames) {
          if (not aliasModulesToProcess.empty() and
              aliasModulesToProcess.find(moduleLabel) == aliasModulesToProcess.end()) {
            continue;
          }

          VParameterSet vPSet = aliasPSet.getParameter<VParameterSet>(moduleLabel);
          for (ParameterSet& pset : vPSet) {
            desc.validate(pset);
            std::string friendlyClassName = pset.getParameter<std::string>("type");
            std::string productInstanceName = pset.getParameter<std::string>("fromProductInstance");
            std::string instanceAlias = pset.getParameter<std::string>("toProductInstance");

            if (friendlyClassName == star) {
              bool processHasLabel = false;
              bool match = false;
              for (auto it = moduleLabelToBranches.lower_bound(moduleLabel);
                   it != moduleLabelToBranches.end() && it->first == moduleLabel;
                   ++it) {
                processHasLabel = true;
                if (productInstanceName != star and productInstanceName != it->second.productInstanceName()) {
                  continue;
                }
                match = true;

                checkAndInsertAlias(it->second.friendlyClassName(),
                                    moduleLabel,
                                    it->second.productInstanceName(),
                                    processName,
                                    alias,
                                    instanceAlias,
                                    preg,
                                    aliasMap,
                                    aliasKeys);
              }
              if (not match and processHasLabel) {
                // No product was found matching the alias.
                // We throw an exception only if a module with the specified module label was created in this process.
                // Note that if that condition is ever relatex, it  might be best to throw an exception with different
                // message (omitting productInstanceName) in case 'productInstanceName == start'
                throw Exception(errors::Configuration, "EDAlias parameter set mismatch\n")
                    << "There are no products with module label '" << moduleLabel << "' and product instance name '"
                    << productInstanceName << "'.\n";
              }
            } else if (productInstanceName == star) {
              bool match = false;
              BranchKey lowerBound(friendlyClassName, moduleLabel, empty, empty);
              for (ProductRegistry::ProductList::const_iterator it = preg.productList().lower_bound(lowerBound);
                   it != preg.productList().end() && it->first.friendlyClassName() == friendlyClassName &&
                   it->first.moduleLabel() == moduleLabel;
                   ++it) {
                if (it->first.processName() != processName) {
                  continue;
                }
                match = true;

                checkAndInsertAlias(friendlyClassName,
                                    moduleLabel,
                                    it->first.productInstanceName(),
                                    processName,
                                    alias,
                                    instanceAlias,
                                    preg,
                                    aliasMap,
                                    aliasKeys);
              }
              if (!match) {
                // No product was found matching the alias.
                // We throw an exception only if a module with the specified module label was created in this process.
                for (auto const& product : preg.productList()) {
                  if (moduleLabel == product.first.moduleLabel() && processName == product.first.processName()) {
                    throw Exception(errors::Configuration, "EDAlias parameter set mismatch\n")
                        << "There are no products of type '" << friendlyClassName << "'\n"
                        << "with module label '" << moduleLabel << "'.\n";
                  }
                }
              }
            } else {
              checkAndInsertAlias(friendlyClassName,
                                  moduleLabel,
                                  productInstanceName,
                                  processName,
                                  alias,
                                  instanceAlias,
                                  preg,
                                  aliasMap,
                                  aliasKeys);
            }
          }
        }
      }

      // Now add the new alias entries to the product registry.
      for (auto const& aliasEntry : aliasMap) {
        // Then check that the alias-for product exists
        ProductRegistry::ProductList::const_iterator it = preg.productList().find(aliasEntry.first);
        assert(it != preg.productList().end());
        preg.addLabelAlias(it->second, aliasEntry.second.moduleLabel(), aliasEntry.second.productInstanceName());
      }
    }
  }  // namespace detail
}  // namespace edm
