import FWCore.ParameterSet.Config as cms

def NoPlot(name):
    return cms.PSet( name = cms.string(name), kind = cms.string("none"))

def Plot1D(name, column, nbins, xmin, xmax):
    return cms.PSet( name = cms.string(name), kind = cms.string("hist1d"), column = cms.string(column), nbins = cms.uint32(nbins), min = cms.double(xmin), max = cms.double(xmax) )

def Profile1D(name, ycolumn, xcolumn, nbins, xmin, xmax):
    return cms.PSet( name = cms.string(name), kind = cms.string("prof1d"), ycolumn = cms.string(ycolumn), xcolumn = cms.string(xcolumn), nbins = cms.uint32(nbins), min = cms.double(xmin), max = cms.double(xmax) )

def shortDump(pset):
    kind = pset.kind.value()
    if kind == "none": 
        return  "NoPlot(%r)" % (pset.name.value())
    elif kind == "hist1d":
        return ("Plot1D(%r, %r, %d, %g, %g),\n" % (pset.name.value(), pset.column.value(), pset.nbins.value(), pset.min.value(), pset.max.value()))
    elif kind == "prof1d":
        return ("Profile1D(%r, %r, %r, %d, %g, %g),\n" % (pset.name.value(), pset.ycolumn.value(), pset.xcolumn.value(), pset.nbins.value(), pset.min.value(), pset.max.value()))
    

