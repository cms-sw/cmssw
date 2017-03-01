2016.06 EMigliore+ATricomi (INFN)

DataFormat/Phase2ITPixelCluster: class for clusters made of SiPixelDigis produced by the 
phase II PixelDigitizerAlgorithm (InnerPixels)

The new data format is essentially a clone of the present DataFormats/SiPixelCluster 
with modifications to the type of some data members to span the full module in case of small pitch pixels 
(e.g. pixel with column or row index larger than 2**8-1=255)

The EDProducer is in RecoLocalTracker/Phase2ITPixelClusterizer
