import FWCore.ParameterSet.Config as cms
def OVar(valtype, compression=None, doc=None, mcOnly=False,precision=-1):
    """ Create a PSet for a variable in the tree (without specifying how it is computed)

           valtype is the type of the value (float, int, bool, or a string that the table producer understands),
           compression is not currently used,
           doc is a docstring, that will be passed to the table producer,
           mcOnly can be set to True for variables that exist only in MC samples and not in data ones. 
    """
    if   valtype == float: valtype = "float"
    elif valtype == int:   valtype = "int"
    elif valtype == bool:  valtype = "bool"
    return cms.PSet( 
                type = cms.string(valtype),
                compression = cms.string(compression if compression else "none"),
                doc = cms.string(doc if doc else expr),
                mcOnly = cms.bool(mcOnly),
	        precision=cms.int32(precision)
           )
def Var(expr, valtype, compression=None, doc=None, mcOnly=False,precision=-1):
    """Create a PSet for a variable computed with the string parser

       expr is the expression to evaluate to compute the variable 
       (in case of bools, it's a cut and not a function)

       see OVar above for all the other arguments
    """
    return OVar(valtype, compression=compression, doc=(doc if doc else expr), mcOnly=mcOnly,precision=precision).clone(
                expr = cms.string(expr))

def ExtVar(tag, valtype, compression=None, doc=None, mcOnly=False,precision=-1):
    """Create a PSet for a variable read from the event

       tag is the InputTag to the variable. 

       see OVar in common_cff for all the other arguments
    """
    return OVar(valtype, compression=compression,precision=precision, doc=(doc if doc else tag.encode()), mcOnly=mcOnly).clone(
                src = tag if type(tag) == cms.InputTag else cms.InputTag(tag),
          )
           

PTVars = cms.PSet(
    pt  = Var("pt",  float, precision=-1),
    phi = Var("phi", float, precision=12),
)
P3Vars = cms.PSet(PTVars,
    eta  = Var("eta",  float,precision=12),
)
P4Vars = cms.PSet(P3Vars,
    mass = Var("mass", float,precision=10),
)
CandVars = cms.PSet(P4Vars,
    pdgId  = Var("pdgId", int, doc="PDG code assigned by the event reconstruction (not by MC truth)"),
    charge = Var("charge", int, doc="electric charge"),
)




