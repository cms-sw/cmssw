from HLTrigger.Configuration.common import esproducers_by_type 

def customizePixelClusterShapeFilterOff(process):
    for prod in esproducers_by_type(process,'ClusterShapeHitFilterESProducer'):
        setattr(prod,'doPixelShapeCut',False)
    return process
