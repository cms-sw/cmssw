import FWCore.ParameterSet.Config as cms

def NoPlot(name):
    return cms.PSet( name = cms.string(name), kind = cms.string("none")) # name is actually a parameter that is not used in the code, but used in python for modifications

def Count1D(name, nbins, xmin, xmax, title=""):
    return cms.PSet( name = cms.string(name), kind = cms.string("count1d"), nbins = cms.uint32(nbins), min = cms.double(xmin), max = cms.double(xmax), title = cms.string(title) )

def Plot1D(name, column, nbins, xmin, xmax, title="", bitset=False):
    return cms.PSet( name = cms.string(name), kind = cms.string("hist1d"), column = cms.string(column), nbins = cms.uint32(nbins), min = cms.double(xmin), max = cms.double(xmax), title = cms.string(title), bitset=cms.bool(bitset) )

def Profile1D(name, ycolumn, xcolumn, nbins, xmin, xmax, title=""):
    return cms.PSet( name = cms.string(name), kind = cms.string("prof1d"), ycolumn = cms.string(ycolumn), xcolumn = cms.string(xcolumn), nbins = cms.uint32(nbins), min = cms.double(xmin), max = cms.double(xmax), title = cms.string(title) )

def shortDump(pset):
    kind = pset.kind.value()
    if kind == "none":
        return  "NoPlot(%r)" % (pset.name.value())
    elif kind == "count1d":
        return ("Count1D(%r, %d, %g, %g%s)" % (pset.name.value(), pset.nbins.value(), pset.min.value(), pset.max.value(), ", %r" % pset.title.value() if pset.title.value() else ""))
    elif kind == "hist1d":
        return ("Plot1D(%r, %r, %d, %g, %g%s)" % (pset.name.value(), pset.column.value(), pset.nbins.value(), pset.min.value(), pset.max.value(), ", %r" % pset.title.value() if pset.title.value() else ""))
    elif kind == "prof1d":
        return ("Profile1D(%r, %r, %r, %d, %g, %g%s)" % (pset.name.value(), pset.ycolumn.value(), pset.xcolumn.value(), pset.nbins.value(), pset.min.value(), pset.max.value(), ", %r" % pset.title.value() if pset.title.value() else ""))


