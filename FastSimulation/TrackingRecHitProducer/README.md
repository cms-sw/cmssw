#TrackingRecHitProducer
##Features
- tracker hit [producer](plugins/TrackingRecHitProducer.cc) can be configured through [plugins](interface/TrackingRecHitAlgorithm.h) which emulate reconstructed hits from simulated hits
- plugins can be restricted to run only on certain DetIds using a [string parser](interface/TrackerDetIdSelector.h)
- multiple plugins can be run [sequencially](interface/TrackingRecHitPipe.h) per DetId
- plugins are mapped to DetIds and the resulting reconstructed hits are collected (reduced) and put into the event 

##In detail
###Plugins
###Pipes
###Selector

