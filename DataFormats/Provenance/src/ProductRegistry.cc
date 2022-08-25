/**
   \file
   class impl

   \Original author Stefano ARGIRO
   \Current author Bill Tanenbaum
   \date 19 Jul 2005
*/

#include "DataFormats/Provenance/interface/ProductRegistry.h"

#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"

#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "TDictAttributeMap.h"

#include <cassert>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <ostream>

namespace edm {

  ProductRegistry::ProductRegistry() : productList_(), transient_() {}

  ProductRegistry::Transients::Transients()
      : frozen_(false),
        productProduced_(),
        anyProductProduced_(false),
        productLookups_{{std::make_unique<ProductResolverIndexHelper>(),
                         std::make_unique<ProductResolverIndexHelper>(),
                         std::make_unique<ProductResolverIndexHelper>(),
                         std::make_unique<ProductResolverIndexHelper>()}},
        nextIndexValues_(),
        branchIDToIndex_() {
    for (bool& isProduced : productProduced_)
      isProduced = false;
  }

  void ProductRegistry::Transients::reset() {
    frozen_ = false;
    for (bool& isProduced : productProduced_)
      isProduced = false;
    anyProductProduced_ = false;

    // propagate_const<T> has no reset() function
    for (auto& iterProductLookup : productLookups_) {
      iterProductLookup = std::make_unique<ProductResolverIndexHelper>();
    }
    nextIndexValues_.fill(0);

    branchIDToIndex_.clear();
  }

  ProductRegistry::ProductRegistry(ProductList const& productList, bool toBeFrozen)
      : productList_(productList), transient_() {
    freezeIt(toBeFrozen);
  }

  void ProductRegistry::addProduct(BranchDescription const& productDesc, bool fromListener) {
    assert(productDesc.produced());
    throwIfFrozen();
    std::pair<ProductList::iterator, bool> ret =
        productList_.insert(std::make_pair(BranchKey(productDesc), productDesc));
    if (!ret.second) {
      auto const& previous = *productList_.find(BranchKey(productDesc));
      if (previous.second.produced()) {
        // Duplicate registration in current process
        throw Exception(errors::LogicError, "Duplicate Product Identifier")
            << "\nThe Framework requires a unique branch name for each product\n"
            << "which consists of four parts: a friendly class name, module label,\n"
            << "product instance name, and process name. A product has been\n"
            << "registered with a duplicate branch name. The most common way\n"
            << "to fix this error is to modify the product instance name in\n"
            << "one of the offending 'produces' function calls. Another fix\n"
            << "would be to delete one of them if they are for the same product.\n\n"
            << "    friendly class name = " << previous.second.friendlyClassName() << "\n"
            << "    module label = " << previous.second.moduleLabel() << "\n"
            << "    product instance name = " << previous.second.productInstanceName() << "\n"
            << "    process name = " << previous.second.processName() << "\n\n"
            << "The following additional information is not used as part of\n"
            << "the unique branch identifier.\n\n"
            << "    branch types = " << previous.second.branchType() << "  " << productDesc.branchType() << "\n"
            << "    class name = " << previous.second.fullClassName() << "\n\n"
            << "Note that if the four parts of the branch name are the same,\n"
            << "then this error will occur even if the branch types differ!\n\n";
      } else {
        // Duplicate registration in previous process
        throw Exception(errors::Configuration, "Duplicate Process Name.\n")
            << "The process name " << productDesc.processName() << " was previously used for products in the input.\n"
            << "This has caused branch name conflicts between input products and new products.\n"
            << "Please modify the configuration file to use a distinct process name.\n"
            << "Alternately, drop all input products using that process name and the\n"
            << "descendants of those products.\n";
      }
    }
    addCalled(productDesc, fromListener);
  }

  void ProductRegistry::addLabelAlias(BranchDescription const& productDesc,
                                      std::string const& labelAlias,
                                      std::string const& instanceAlias) {
    assert(productDesc.produced());
    assert(productDesc.branchID().isValid());
    throwIfFrozen();
    BranchDescription bd(productDesc, labelAlias, instanceAlias);
    std::pair<ProductList::iterator, bool> ret = productList_.insert(std::make_pair(BranchKey(bd), bd));
    assert(ret.second);
    transient_.aliasToOriginal_.emplace_back(
        PRODUCT_TYPE, productDesc.unwrappedTypeID(), labelAlias, instanceAlias, productDesc.moduleLabel());
    addCalled(bd, false);
  }

  void ProductRegistry::copyProduct(BranchDescription const& productDesc) {
    assert(!productDesc.produced());
    throwIfFrozen();
    BranchKey k = BranchKey(productDesc);
    ProductList::iterator iter = productList_.find(k);
    if (iter == productList_.end()) {
      productList_.insert(std::make_pair(k, productDesc));
    } else {
      assert(combinable(iter->second, productDesc));
      iter->second.merge(productDesc);
    }
  }

  bool ProductRegistry::anyProducts(BranchType brType) const {
    throwIfNotFrozen();
    for (ProductList::const_iterator it = productList_.begin(), itEnd = productList_.end(); it != itEnd; ++it) {
      if (it->second.branchType() == brType) {
        return true;
      }
    }
    return false;
  }

  std::shared_ptr<ProductResolverIndexHelper const> ProductRegistry::productLookup(BranchType branchType) const {
    return get_underlying_safe(transient_.productLookups_[branchType]);
  }

  std::shared_ptr<ProductResolverIndexHelper> ProductRegistry::productLookup(BranchType branchType) {
    return get_underlying_safe(transient_.productLookups_[branchType]);
  }

  void ProductRegistry::setFrozen(bool initializeLookupInfo) {
    if (frozen())
      return;
    freezeIt();
    if (initializeLookupInfo) {
      initializeLookupTables(nullptr, nullptr, nullptr);
    }
    sort_all(transient_.aliasToOriginal_);
  }

  void ProductRegistry::setFrozen(std::set<TypeID> const& productTypesConsumed,
                                  std::set<TypeID> const& elementTypesConsumed,
                                  std::string const& processName) {
    if (frozen())
      return;
    freezeIt();
    initializeLookupTables(&productTypesConsumed, &elementTypesConsumed, &processName);
    sort_all(transient_.aliasToOriginal_);
  }

  void ProductRegistry::throwIfFrozen() const {
    if (frozen()) {
      throw cms::Exception("ProductRegistry", "throwIfFrozen")
          << "cannot modify the ProductRegistry because it is frozen\n";
    }
  }

  void ProductRegistry::throwIfNotFrozen() const {
    if (!frozen()) {
      throw cms::Exception("ProductRegistry", "throwIfNotFrozen")
          << "cannot read the ProductRegistry because it is not yet frozen\n";
    }
  }

  void ProductRegistry::addCalled(BranchDescription const&, bool) {}

  std::vector<std::string> ProductRegistry::allBranchNames() const {
    std::vector<std::string> result;
    result.reserve(productList().size());

    for (auto const& product : productList()) {
      result.push_back(product.second.branchName());
    }
    return result;
  }

  std::vector<BranchDescription const*> ProductRegistry::allBranchDescriptions() const {
    std::vector<BranchDescription const*> result;
    result.reserve(productList().size());

    for (auto const& product : productList()) {
      result.push_back(&product.second);
    }
    return result;
  }

  void ProductRegistry::updateFromInput(ProductList const& other) {
    for (auto const& product : other) {
      copyProduct(product.second);
    }
  }

  void ProductRegistry::updateFromInput(std::vector<BranchDescription> const& other) {
    for (BranchDescription const& branchDescription : other) {
      copyProduct(branchDescription);
    }
  }

  void ProductRegistry::addFromInput(edm::ProductRegistry const& other) {
    throwIfFrozen();
    for (auto const& prod : other.productList_) {
      ProductList::iterator iter = productList_.find(prod.first);
      if (iter == productList_.end()) {
        productList_.insert(std::make_pair(prod.first, prod.second));
        addCalled(prod.second, false);
      } else {
        assert(combinable(iter->second, prod.second));
        iter->second.merge(prod.second);
      }
    }
  }

  void ProductRegistry::setUnscheduledProducts(std::set<std::string> const& unscheduledLabels) {
    throwIfFrozen();

    bool hasAliases = false;
    std::vector<BranchID> onDemandIDs;
    for (auto& prod : productList_) {
      if (prod.second.produced() && prod.second.branchType() == InEvent &&
          unscheduledLabels.end() != unscheduledLabels.find(prod.second.moduleLabel())) {
        prod.second.setOnDemand(true);
        onDemandIDs.push_back(prod.second.branchID());
      }
      if (prod.second.produced() && prod.second.isAlias()) {
        hasAliases = true;
      }
    }

    // Need to loop over EDAliases to set their on-demand flag based on the pointed-to branch
    if (hasAliases) {
      std::sort(onDemandIDs.begin(), onDemandIDs.end());
      for (auto& prod : productList_) {
        if (prod.second.isAlias()) {
          if (std::binary_search(onDemandIDs.begin(), onDemandIDs.end(), prod.second.aliasForBranchID())) {
            prod.second.setOnDemand(true);
          }
        }
      }
    }
  }

  std::string ProductRegistry::merge(ProductRegistry const& other,
                                     std::string const& fileName,
                                     BranchDescription::MatchMode branchesMustMatch) {
    std::ostringstream differences;

    ProductRegistry::ProductList::iterator j = productList_.begin();
    ProductRegistry::ProductList::iterator s = productList_.end();
    ProductRegistry::ProductList::const_iterator i = other.productList().begin();
    ProductRegistry::ProductList::const_iterator e = other.productList().end();

    // Loop over entries in the main product registry.
    while (j != s || i != e) {
      if (j != s && j->second.produced()) {
        // Ignore branches just produced (i.e. not in input file).
        ++j;
      } else if (j == s || (i != e && i->first < j->first)) {
        if (i->second.present()) {
          differences << "Branch '" << i->second.branchName() << "' is in file '" << fileName << "'\n";
          differences << "    but not in previous files.\n";
        } else {
          productList_.insert(*i);
          transient_.branchIDToIndex_[i->second.branchID()] = getNextIndexValue(i->second.branchType());
          ++nextIndexValue(i->second.branchType());
        }
        ++i;
      } else if (i == e || (j != s && j->first < i->first)) {
        if (j->second.present() &&
            (branchesMustMatch == BranchDescription::Strict || j->second.branchType() == InProcess)) {
          differences << "Branch '" << j->second.branchName() << "' is in previous files\n";
          differences << "    but not in file '" << fileName << "'.\n";
        }
        ++j;
      } else {
        std::string difs = match(j->second, i->second, fileName);
        if (difs.empty()) {
          j->second.merge(i->second);
        } else {
          differences << difs;
        }
        ++i;
        ++j;
      }
    }
    return differences.str();
  }

  void ProductRegistry::initializeLookupTables(std::set<TypeID> const* productTypesConsumed,
                                               std::set<TypeID> const* elementTypesConsumed,
                                               std::string const* processName) {
    std::map<TypeID, TypeID> containedTypeMap;
    std::map<TypeID, std::vector<TypeID>> containedTypeToBaseTypesMap;

    std::vector<std::string> missingDictionaries;
    std::vector<std::string> branchNamesForMissing;
    std::vector<std::string> producedTypes;

    transient_.branchIDToIndex_.clear();

    for (auto const& product : productList_) {
      auto const& desc = product.second;

      checkForDuplicateProcessName(desc, processName);

      if (desc.produced() && !desc.transient()) {
        setProductProduced(desc.branchType());
      }

      //only do the following if the data is supposed to be available in the event
      if (desc.present()) {
        // Check dictionaries (we already checked for the produced ones earlier somewhere else).
        // We have to have the dictionaries to properly setup the lookup tables for support of
        // Views. Also we need them to determine which present products are declared to be
        // consumed in the case where the consumed type is a View<T>.
        if (!desc.produced()) {
          if (!checkDictionary(missingDictionaries, desc.className(), desc.unwrappedType())) {
            checkDictionaryOfWrappedType(missingDictionaries, desc.className());
            branchNamesForMissing.emplace_back(desc.branchName());
            producedTypes.emplace_back(desc.className() + std::string(" (read from input)"));
            continue;
          }
        }
        TypeID typeID(desc.unwrappedType().typeInfo());

        auto iterContainedType = containedTypeMap.find(typeID);
        bool alreadySawThisType = (iterContainedType != containedTypeMap.end());

        if (!desc.produced() && !alreadySawThisType) {
          if (!checkDictionary(missingDictionaries, desc.wrappedName(), desc.wrappedType())) {
            branchNamesForMissing.emplace_back(desc.branchName());
            producedTypes.emplace_back(desc.className() + std::string(" (read from input)"));
            continue;
          }
        }

        TypeID wrappedTypeID(desc.wrappedType().typeInfo());

        TypeID containedTypeID;
        if (alreadySawThisType) {
          containedTypeID = iterContainedType->second;
        } else {
          containedTypeID = productholderindexhelper::getContainedTypeFromWrapper(wrappedTypeID, typeID.className());
        }
        bool hasContainedType = (containedTypeID != TypeID(typeid(void)) && containedTypeID != TypeID());

        std::vector<TypeID>* baseTypesOfContainedType = nullptr;

        if (!alreadySawThisType) {
          bool alreadyCheckedConstituents = desc.produced() && !desc.transient();
          if (!alreadyCheckedConstituents && !desc.transient()) {
            // This checks dictionaries of the wrapped class and all its constituent classes
            if (!checkClassDictionaries(missingDictionaries, desc.wrappedName(), desc.wrappedType())) {
              branchNamesForMissing.emplace_back(desc.branchName());
              producedTypes.emplace_back(desc.className() + std::string(" (read from input)"));
              continue;
            }
          }

          if (hasContainedType) {
            auto iterBaseTypes = containedTypeToBaseTypesMap.find(containedTypeID);
            if (iterBaseTypes == containedTypeToBaseTypesMap.end()) {
              std::vector<TypeID> baseTypes;
              if (!public_base_classes(missingDictionaries, containedTypeID, baseTypes)) {
                branchNamesForMissing.emplace_back(desc.branchName());
                if (desc.produced()) {
                  producedTypes.emplace_back(desc.className() + std::string(" (produced in current process)"));
                } else {
                  producedTypes.emplace_back(desc.className() + std::string(" (read from input)"));
                }
                continue;
              }
              iterBaseTypes = containedTypeToBaseTypesMap.insert(std::make_pair(containedTypeID, baseTypes)).first;
            }
            baseTypesOfContainedType = &iterBaseTypes->second;
          }

          // Do this after the dictionary checks of constituents so the list of branch names for missing types
          // is complete
          containedTypeMap.emplace(typeID, containedTypeID);
        } else {
          if (hasContainedType) {
            auto iterBaseTypes = containedTypeToBaseTypesMap.find(containedTypeID);
            if (iterBaseTypes != containedTypeToBaseTypesMap.end()) {
              baseTypesOfContainedType = &iterBaseTypes->second;
            }
          }
        }

        if (productTypesConsumed != nullptr && !desc.produced()) {
          bool mainTypeConsumed = (productTypesConsumed->find(typeID) != productTypesConsumed->end());
          bool containedTypeConsumed =
              hasContainedType && (elementTypesConsumed->find(containedTypeID) != elementTypesConsumed->end());
          if (hasContainedType && !containedTypeConsumed && baseTypesOfContainedType != nullptr) {
            for (TypeID const& baseType : *baseTypesOfContainedType) {
              if (elementTypesConsumed->find(TypeID(baseType.typeInfo())) != elementTypesConsumed->end()) {
                containedTypeConsumed = true;
                break;
              }
            }
          }
          if (!containedTypeConsumed) {
            if (mainTypeConsumed) {
              // The main type is consumed, but either
              // there is no contained type, or if there is,
              // neither it nor any of its base classes are consumed.
              // Set the contained type, if there is one, to void,
              if (hasContainedType) {
                containedTypeID = TypeID(typeid(void));
              }
            } else {
              // The main type is not consumed, and either
              // there is no contained type, or if there is,
              // neither it nor any of its base classes are consumed.
              // Don't insert anything in the lookup tables.
              continue;
            }
          }
        }
        ProductResolverIndex index = productLookup(desc.branchType())
                                         ->insert(typeID,
                                                  desc.moduleLabel().c_str(),
                                                  desc.productInstanceName().c_str(),
                                                  desc.processName().c_str(),
                                                  containedTypeID,
                                                  baseTypesOfContainedType);

        transient_.branchIDToIndex_[desc.branchID()] = index;
      }
    }
    if (!missingDictionaries.empty()) {
      std::string context("Calling ProductRegistry::initializeLookupTables");
      throwMissingDictionariesException(missingDictionaries, context, producedTypes, branchNamesForMissing);
    }

    for (auto& iterProductLookup : transient_.productLookups_) {
      iterProductLookup->setFrozen();
    }

    unsigned int indexIntoNextIndexValue = 0;
    for (auto const& iterProductLookup : transient_.productLookups_) {
      transient_.nextIndexValues_[indexIntoNextIndexValue] = iterProductLookup->nextIndexValue();
      ++indexIntoNextIndexValue;
    }

    for (auto const& product : productList_) {
      auto const& desc = product.second;
      if (transient_.branchIDToIndex_.find(desc.branchID()) == transient_.branchIDToIndex_.end()) {
        transient_.branchIDToIndex_[desc.branchID()] = getNextIndexValue(desc.branchType());
        ++nextIndexValue(desc.branchType());
      }
    }
    checkDictionariesOfConsumedTypes(
        productTypesConsumed, elementTypesConsumed, containedTypeMap, containedTypeToBaseTypesMap);

    addElementTypesForAliases(elementTypesConsumed, containedTypeMap, containedTypeToBaseTypesMap);
  }

  void ProductRegistry::addElementTypesForAliases(
      std::set<TypeID> const* elementTypesConsumed,
      std::map<TypeID, TypeID> const& containedTypeMap,
      std::map<TypeID, std::vector<TypeID>> const& containedTypeToBaseTypesMap) {
    Transients::AliasToOriginalVector elementAliases;
    for (auto& item : transient_.aliasToOriginal_) {
      auto iterContainedType = containedTypeMap.find(std::get<Transients::kType>(item));
      if (iterContainedType == containedTypeMap.end()) {
        edm::Exception ex(errors::LogicError);
        ex << "containedTypeMap did not contain " << std::get<Transients::kType>(item).className()
           << " that is used in EDAlias " << std::get<Transients::kModuleLabel>(item)
           << ".\nThis should not happen, contact framework developers";
        ex.addContext("Calling ProductRegistry::initializeLookupTables()");
        throw ex;
      }
      auto const& containedTypeID = iterContainedType->second;
      bool const hasContainedType = (containedTypeID != TypeID(typeid(void)) && containedTypeID != TypeID());
      if (not hasContainedType) {
        continue;
      }

      if (elementTypesConsumed->find(containedTypeID) != elementTypesConsumed->end()) {
        elementAliases.emplace_back(ELEMENT_TYPE,
                                    containedTypeID,
                                    std::get<Transients::kModuleLabel>(item),
                                    std::get<Transients::kProductInstanceName>(item),
                                    std::get<Transients::kAliasForModuleLabel>(item));
      }

      auto iterBaseTypes = containedTypeToBaseTypesMap.find(containedTypeID);
      if (iterBaseTypes == containedTypeToBaseTypesMap.end()) {
        continue;
      }
      for (TypeID const& baseTypeID : iterBaseTypes->second) {
        if (elementTypesConsumed->find(baseTypeID) != elementTypesConsumed->end()) {
          elementAliases.emplace_back(ELEMENT_TYPE,
                                      baseTypeID,
                                      std::get<Transients::kModuleLabel>(item),
                                      std::get<Transients::kProductInstanceName>(item),
                                      std::get<Transients::kAliasForModuleLabel>(item));
        }
      }
    }
    transient_.aliasToOriginal_.insert(transient_.aliasToOriginal_.end(),
                                       std::make_move_iterator(elementAliases.begin()),
                                       std::make_move_iterator(elementAliases.end()));
  }

  void ProductRegistry::checkDictionariesOfConsumedTypes(
      std::set<TypeID> const* productTypesConsumed,
      std::set<TypeID> const* elementTypesConsumed,
      std::map<TypeID, TypeID> const& containedTypeMap,
      std::map<TypeID, std::vector<TypeID>>& containedTypeToBaseTypesMap) {
    std::vector<std::string> missingDictionaries;
    std::set<std::string> consumedTypesWithMissingDictionaries;

    if (productTypesConsumed) {
      // Check dictionaries for all classes declared to be consumed
      for (auto const& consumedTypeID : *productTypesConsumed) {
        // We use the containedTypeMap to see which types have already
        // had their dictionaries checked. We do not waste time rechecking
        // those dictionaries.
        if (containedTypeMap.find(consumedTypeID) == containedTypeMap.end()) {
          std::string wrappedName = wrappedClassName(consumedTypeID.className());
          TypeWithDict wrappedType = TypeWithDict::byName(wrappedName);
          if (!checkDictionary(missingDictionaries, wrappedName, wrappedType)) {
            checkDictionary(missingDictionaries, consumedTypeID);
            consumedTypesWithMissingDictionaries.emplace(consumedTypeID.className());
            continue;
          }
          bool transient = false;
          TDictAttributeMap* wp = wrappedType.getClass()->GetAttributeMap();
          if (wp && wp->HasKey("persistent") && !strcmp(wp->GetPropertyAsString("persistent"), "false")) {
            transient = true;
          }
          if (transient) {
            if (!checkDictionary(missingDictionaries, consumedTypeID)) {
              consumedTypesWithMissingDictionaries.emplace(consumedTypeID.className());
            }

            TypeID containedTypeID = productholderindexhelper::getContainedTypeFromWrapper(
                TypeID(wrappedType.typeInfo()), consumedTypeID.className());
            bool hasContainedType = (containedTypeID != TypeID(typeid(void)) && containedTypeID != TypeID());
            if (hasContainedType) {
              if (containedTypeToBaseTypesMap.find(containedTypeID) == containedTypeToBaseTypesMap.end()) {
                std::vector<TypeID> bases;
                // Run this to check for missing dictionaries, bases is not really used
                if (!public_base_classes(missingDictionaries, containedTypeID, bases)) {
                  consumedTypesWithMissingDictionaries.emplace(consumedTypeID.className());
                }
                containedTypeToBaseTypesMap.insert(std::make_pair(containedTypeID, bases));
              }
            }
          } else {
            if (!checkClassDictionaries(missingDictionaries, wrappedName, wrappedType)) {
              consumedTypesWithMissingDictionaries.emplace(consumedTypeID.className());
            }
          }
        }
      }
      if (!missingDictionaries.empty()) {
        std::string context(
            "Calling ProductRegistry::initializeLookupTables, checking dictionaries for consumed products");
        throwMissingDictionariesException(missingDictionaries, context, consumedTypesWithMissingDictionaries, false);
      }
    }

    if (elementTypesConsumed) {
      missingDictionaries.clear();
      consumedTypesWithMissingDictionaries.clear();
      for (auto const& consumedTypeID : *elementTypesConsumed) {
        if (containedTypeToBaseTypesMap.find(consumedTypeID) == containedTypeToBaseTypesMap.end()) {
          std::vector<TypeID> bases;
          // Run this to check for missing dictionaries, bases is not really used
          if (!public_base_classes(missingDictionaries, consumedTypeID, bases)) {
            consumedTypesWithMissingDictionaries.emplace(consumedTypeID.className());
          }
        }
      }
      if (!missingDictionaries.empty()) {
        std::string context(
            "Calling ProductRegistry::initializeLookupTables, checking dictionaries for elements of products consumed "
            "using View");
        throwMissingDictionariesException(missingDictionaries, context, consumedTypesWithMissingDictionaries, true);
      }
    }
  }

  void ProductRegistry::checkForDuplicateProcessName(BranchDescription const& desc,
                                                     std::string const* processName) const {
    if (processName && !desc.produced() && (*processName == desc.processName())) {
      throw Exception(errors::Configuration, "Duplicate Process Name.\n")
          << "The process name " << *processName << " was previously used for products in the input.\n"
          << "Please modify the configuration file to use a distinct process name.\n"
          << "Alternately, drop all input products using that process name and the\n"
          << "descendants of those products.\n";
    }
  }

  ProductResolverIndex ProductRegistry::indexFrom(BranchID const& iID) const {
    std::map<BranchID, ProductResolverIndex>::const_iterator itFind = transient_.branchIDToIndex_.find(iID);
    if (itFind == transient_.branchIDToIndex_.end()) {
      return ProductResolverIndexInvalid;
    }
    return itFind->second;
  }

  std::vector<std::string> ProductRegistry::aliasToModules(KindOfType kindOfType,
                                                           TypeID const& type,
                                                           std::string_view moduleLabel,
                                                           std::string_view productInstanceName) const {
    auto aliasFields = [](auto const& item) {
      return std::tie(std::get<Transients::kKind>(item),
                      std::get<Transients::kType>(item),
                      std::get<Transients::kModuleLabel>(item),
                      std::get<Transients::kProductInstanceName>(item));
    };
    auto const target = std::tuple(kindOfType, type, moduleLabel, productInstanceName);
    auto found =
        std::lower_bound(transient_.aliasToOriginal_.begin(),
                         transient_.aliasToOriginal_.end(),
                         target,
                         [aliasFields](auto const& item, auto const& target) { return aliasFields(item) < target; });
    std::vector<std::string> ret;
    for (; found != transient_.aliasToOriginal_.end() and aliasFields(*found) == target; ++found) {
      ret.emplace_back(std::get<Transients::kAliasForModuleLabel>(*found));
    }
    return ret;
  }

  void ProductRegistry::print(std::ostream& os) const {
    for (auto const& product : productList_) {
      os << product.second << "\n-----\n";
    }
  }

  ProductResolverIndex const& ProductRegistry::getNextIndexValue(BranchType branchType) const {
    return transient_.nextIndexValues_[branchType];
  }

  ProductResolverIndex& ProductRegistry::nextIndexValue(BranchType branchType) {
    return transient_.nextIndexValues_[branchType];
  }
}  // namespace edm
