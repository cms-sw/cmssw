import itertools

import FWCore.ParameterSet.Config as cms

def producers_by_type(process, *types):
    "Find all EDProducers in the Process that are instances of the given C++ type."
    return (module for module in process._Process__producers.values() if module._TypedParameterizable__type in types)

def filters_by_type(process, *types):
    "Find all EDFilters in the Process that are instances of the given C++ type."
    return (filter for filter in process._Process__filters.values() if filter._TypedParameterizable__type in types)

def analyzers_by_type(process, *types):
    "Find all EDAnalyzers in the Process that are instances of the given C++ type."
    return (analyzer for analyzer in process._Process__analyzers.values() if analyzer._TypedParameterizable__type in types)

def esproducers_by_type(process, *types):
    "Find all ESProducers in the Process that are instances of the given C++ type."
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)


def insert_modules_before(process, target, *modules):
    "Add the `modules` before the `target` in any Sequence, Paths or EndPath that contains the latter."
    for sequence in itertools.chain(
        process._Process__sequences.itervalues(),
        process._Process__paths.itervalues(),
        process._Process__endpaths.itervalues()
    ):
        try:
            position = sequence.index(target)
        except ValueError:
            continue
        else:
            for module in reversed(modules):
                sequence.insert(position, module)


def insert_modules_after(process, target, *modules):
    "Add the `modules` after the `target` in any Sequence, Paths or EndPath that contains the latter."
    for sequence in itertools.chain(
        process._Process__sequences.itervalues(),
        process._Process__paths.itervalues(),
        process._Process__endpaths.itervalues()
    ):
        try:
            position = sequence.index(target)
        except ValueError:
            continue
        else:
            for module in reversed(modules):
                sequence.insert(position+1, module)


# logic from Modifier.toModify from FWCore/ParameterSet/python/Config.py
def replace_with(fromObj, toObj):
    """Replace one object with a different one of the same type.

    This function replaces the contents of `fromObj` object with those of `toObj`,
    so all references ot it remain valid.
    """

    if type(toObj) != type(fromObj):
        raise TypeError('replaceWith requires both arguments to be the same type')

    if isinstance(toObj, cms._ModuleSequenceType):
        fromObj._seq = toObj._seq

    elif isinstance(toObj, cms._Parameterizable):
        # delete the old items, in case `toObj` is not a complete superset of `fromObj`
        for p in fromObj.parameterNames_():
            delattr(fromObj, p)
        for p in toObj.parameterNames_():
            setattr(fromObj, p, getattr(toObj, p))
        if isinstance(toObj, cms._TypedParameterizable):
            fromObj._TypedParameterizable__type = toObj._TypedParameterizable__type

    else:
        raise TypeError('replaceWith does not work with "%s" objects' % str(type(fromObj)))

