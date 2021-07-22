import FWCore.ParameterSet.Config as cms

import collections

def customiseEarlyDeleteForCandIsoDeposits(process, products):
    # Find the producers

    def _branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for name, module in process.producers_().items():
        cppType = module._TypedParameterizable__type
        if cppType == "CandIsoDepositProducer":
            if module.ExtractorPSet.ComponentName in ["CandViewExtractor", "PFCandWithSuperClusterExtractor"] :
                products[name].append(_branchName("recoIsoDepositedmValueMap", name))

    return products
