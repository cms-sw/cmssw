#TrackingRecHitProducer
##Features
- tracker hit [producer](plugins/TrackingRecHitProducer.cc) can be configured through [plugins](interface/TrackingRecHitAlgorithm.h) which emulate reconstructed hits from simulated hits
- plugins can be restricted to run only on certain DetIds using a [string parser](interface/TrackerDetIdSelector.h)
- multiple plugins can be run [sequentially](interface/TrackingRecHitPipe.h) per DetId
- plugins are mapped to DetIds and the resulting reconstructed hits are collected (reduced) and put into the event 

##In detail
####Plugins
The main task of a plugin is the creation of reconstructed hits given the simulated ones. To create a plugin, one has to inherit from the [interface](interface/TrackingRecHitAlgorithm.h) and overide the `process` method. This method will be called for each selected DetId. A minimalistic example is provided with the ["no smearing plugin"](plugins/TrackingRecHitNoSmearingPlugin.cc) that can be used for performance studies or debugging.
####Pipes
A [pipe](interface/TrackingRecHitPipe.h) is created in the producer's `setupDetIdPipes` method for each DetId of the current tracker geometry. Then plugins are assigned to the pipes depending on their DetId selection. Usually, the same plugin instance is used in multiple pipes. The following visualizes how a certain list of plugins may be matched to pipes by parsing the selection string as configured for each plugin:

pipe/plugin | pipe for DetId1 | pipe for DetId2 | ...
------------|-----------------|-----------------|--------
plugin1     |       x         |          -      |  x
plugin2     |       x         |          x      |  -
plugin3     |       -         |          x      |  -

####Selector
The Boost Spirit package is used to define grammar rules for testing if a given string selects a certain DetId given the current tracker topology. Various [key words](src/TrackerDetIdSelector.cc
) are available.

