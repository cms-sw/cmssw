import FWCore.ParameterSet.Config as cms

# this function serves gen-level PU mixing
# it sets the parameters of the vertex generator of the PileUpProducer to the parameters of process.VtxSmeared
# process.VtxSmeared is the vertex generator used to smear the vertex of the signal events
def setVertexGeneratorPileUpProducer(process):
    
    # get right vertex generator parameters
    if not hasattr(process,"VtxSmeared"):
        "WARNING: no vtx smearing applied (ok for steps other than SIM)"
        return process

    vertexGenerator = process.VtxSmeared
    vertexGeneratorParameterNames = vertexGenerator.parameterNames_()
    vertexGeneratorType = vertexGenerator.type_() 

    # set vertex generator parameters in PileUpProducer
    vertexParameters = cms.PSet()
    for name in vertexGeneratorParameterNames:
        exec("vertexParameters.{0} = {1}".format(name,getattr(vertexGenerator,name).dumpPython()))
    
    if vertexGeneratorType.find("Betafunc") == 0:
        vertexParameters.type = cms.string("BetaFunc")
    elif vertexGeneratorType.find("Flat") == 0:
        vertexParameters.type = cms.string("Flat")
    elif vertexGeneratorType.find("Gauss"):
        vertexParameters.type = cms.string("Gaussian")
    else:
        raise Error("WARNING: given vertex generator type for vertex smearing is not supported")

    # add parameters to PileUpProducer
    process.famosPileUp.VertexGenerator = vertexParameters

    return process


