import FWCore.ParameterSet.Config as cms

import collections

def customiseEarlyDeleteForMkFit(process, products):
    
    references = collections.defaultdict(list)

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
            references[_branchName("MkFitEventOfHits", name)] = [
                    _branchName("MkFitHitWrapper", module.pixelHits.moduleLabel),
                    _branchName("MkFitHitWrapper", module.stripHits.moduleLabel)
                    ]
        elif cppType == "MkFitSeedConverter":
            products[name].append(_branchName("MkFitSeedWrapper", name))
        elif cppType == "MkFitProducer":
            products[name].append(_branchName("MkFitOutputWrapper", name))

    return (products,references)
