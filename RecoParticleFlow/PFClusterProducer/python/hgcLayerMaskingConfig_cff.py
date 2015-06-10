import FWCore.ParameterSet.Config as cms

#obvious hardcode to V5 geometry is obvious...
layer_mask = {}
layer_mask['EE']  = [1 for x in range(30)]
layer_mask['HEF'] = [1 for x in range(12)]
layer_mask['HEB'] = [1 for x in range(12)] 

def mask_layer(mask,layer):
    mask[layer-1] = 0

#define the masking
masked_layers = {}
masked_layers['EE']  = []
masked_layers['HEF'] = []
masked_layers['HEB'] = []

for det in layer_mask.keys():
    for layer in masked_layers[det]:
        mask_layer(layer_mask[det],layer)

hgcLayerMasking = cms.PSet(
    EE_layerMask  = cms.vuint32(layer_mask['EE']),
    HEF_layerMask = cms.vuint32(layer_mask['HEF']),
    HEB_layerMask = cms.vuint32(layer_mask['HEB'])
)




