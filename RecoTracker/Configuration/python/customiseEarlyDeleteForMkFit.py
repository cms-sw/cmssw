import FWCore.ParameterSet.Config as cms

import collections

def customiseEarlyDeleteForMkFit(process, products):
    def _branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for name, module in process.producers_().items():
        cppType = module.type_()
        if cppType == "MkFitSiPixelHitConverter":
            products[name].extend([
                _branchName("MkFitHitWrapper", name),
                _branchName("MkFitClusterIndexToHit", name),
            ])
        elif cppType == "MkFitSiStripHitConverter":
            products[name].extend([
                _branchName("MkFitHitWrapper", name),
                _branchName("MkFitClusterIndexToHit", name),
                _branchName("floats", name)
            ])
        elif cppType == "MkFitEventOfHitsProducer":
            products[name].append(_branchName("MkFitEventOfHits", name))
        elif cppType == "MkFitSeedConverter":
            products[name].append(_branchName("MkFitSeedWrapper", name))
        elif cppType == "MkFitProducer":
            products[name].append(_branchName("MkFitOutputWrapper", name))

    return products
