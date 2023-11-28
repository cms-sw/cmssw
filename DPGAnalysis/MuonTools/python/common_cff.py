from PhysicsTools.NanoAOD.common_cff import *
from typing import NamedTuple

class DEFAULT_VAL(NamedTuple):
        INT: int = -999
        INT8: int = -99
        INT_POS: int = -1
        FLOAT: int = -999.0
        FLOAT_POS: int = -1.0

defaults = DEFAULT_VAL()

def DetIdVar(expr, type, doc=None):
    """ Create a PSet for a DetId variable in the tree:
        - expr is the expression to evaluate to compute the variable,
        - type of the value (int, bool, or a string that the table producer understands),
        - doc is a docstring, that will be passed to the table producer
    """
    if   type == float: type = "float"
    elif type == bool: type = "bool"
    elif type == int: type = "int"
    return cms.PSet(
                type = cms.string(type),
                expr = cms.string(expr),
                doc = cms.string(doc if doc else expr)
           )

def GlobGeomVar(expr, doc=None, precision=-1):
    """ Create a PSet for a Global position/direction variable in the tree ,
        - expr is the expression to evaluate to compute the variable,
        - doc is a docstring, that will be passed to the table producer,
        - precision is an int handling mantissa reduction.
    """
    return cms.PSet(
                expr = cms.string(expr),
                doc = cms.string(doc if doc else expr),
	        precision=cms.string(precision) if type(precision)==str else cms.int32(precision)
           )




