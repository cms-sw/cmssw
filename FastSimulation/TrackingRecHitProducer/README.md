#TrackingRecHitProducer
##Features
- tracker hit [producer](plugins/TrackingRecHitProducer.cc) can be configured through [plugins](interface/TrackingRecHitAlgorithm) which emulate reconstructed hits
- plugins can be restricted to run only on certain DetIds using a [string parser](interface/TrackerDetIdSelector.h)
- multiple plugins can be run sequencially per DetId

