# Description of implementation of multiple mkFit iterations

- The branch is up-to-date with respect to devel.

- The main changes in this branch affect the following files:

(1) mkFit/SteeringParams.h:
 - three additional classes, which are iteration-dependent:
 (a) IterationParams: a container for 'nlayers_per_seed', 'maxCandsPerSeed', 'maxHolesPerCand', 'maxConsecHoles', and 'chi2Cut';
 (b) IterationLayerConfig: a container for layer-specific iteration-dependent configurations (e.g., hit selection windows);
 (c) IterationConfig: a container for all of the above, including a virtual functions to import seeds (import_seeds)
 - one additional struct, which is iteration-dependent:
 (a) MkSeedPacket: a container of iteration-specific event objects ('m_seedEtaSeparators_', 'm_seedMinLastLayer_', 'm_seedMaxLastLayer_', 'm_layerHits_', 'm_inseeds', 'm_outtrks')

(2) Geoms/CMS-2017.cc:
 - an instance of IterationConfig is created in Geoms/CMS-2017.cc, to be passed to MkBuilder constructor, which sets all iteration-dependent objects/parameters.

(3) mkFit/MkBuilder[.cc,.h]
 - all iteration-dependent parameters (regions, steering parameters, etc.) are moved out of MkBuilder, and into IterationConfig, which must be passed to MkBuilder constructor to have one MkBuilder per iteration.


-------------------------------------------------------------------------------

MT Notes:

* RegionOfSeedIndices rosi(m_event, region); <---- event

* bkfit --> takes tracks from event->candidateTracks
  This is a somewhat more general probelm ... flow of tracks through processing and
  when should they be copied out / extracted (found / fitted / etc).
  Especially re validation.



-------------------------------------------------------------------------------

VALIDATION

tested validation with 3 iterations scripts (for running validation forConf)

./val_scripts/validation-cmssw-benchmarks-multiiter.sh
./web/collectBenchmarks-multi.sh  

the features added to the validation Trees are the following

FR Trees
  - algorithm: as the tree entries are by seed, the seed "algorithm" (i.e. the iteration number used in cmssw) is saved

EFF Trees
  
  - itermask_[seed/build//fit]
  - iterduplmask_[seed/build//fit]
  - algo_seed

  these are 3 binary masks of 64 bits, where bits are tunred on depending on the iteration matching a sim track, as the entries are organized by sim track
  
  itermask_ :
    a sim track is matching at least to a track with algorithm M if the bit M of the bit mask is on
    multiple bits can be on, if the sim track matched to tracks of muktiple iterations
  
  iterduplmask_ :
    a sim track is matching at least twice (duplicate in the iteration) a track with algorithm M if the bit M of the bit mask is on

  algo_seed:
    to be used in SIMVALSEED
    bit M is on if the seed matching to the sim track comes from the iteration with code M
    
      
  how to use binary masks (example) : 

  the simtrack matches to the iteration with algo = 4, 22, 23 ... ->  (itermask_[]>>algo)&1 
  the simtrack matches twice to the iteration with algo = 4, 22, 23 ... ->  (iterduplmask_[]>>algo)&1 
  the simtrack matches to a seed with algo = 4, 22, 23 ... ->  (algo_seed>>algo)&1 


The script val_scripts/validation-cmssw-benchmarks-multiiter.sh produces 4 sets of plots: 3 iteration specific validation plots and 1 global validation

The settings are the same as for the forConf suite, i.e. comparing CE (build-mimi) to CMSSW - no STD build. the validation setups are the usual SIMVAL and SIMVALSEED (MTV).
In the iteration specific the itermask_[seed/build//fit] and iterduplmask_[seed/build//fit] are used to define efficiency and duplicate rates, algo_seed is also required in SIMVALSEED.
The global validation is similar to the one used for the initial step only (no bit masks used). It can be useful to check the global absolute efficiency after adding iterations after each other.
On the other hand, the comparison of fakes and duplicates between mkFit and cmssw is not totally fair, as different types of cleaning are applied to the two collections.


