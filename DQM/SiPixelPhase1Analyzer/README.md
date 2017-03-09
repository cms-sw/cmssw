SiPixelPhase1Analyzer package
=============================

The aim of this package is to provide a convinient way to present standard Pixel Detector histograms in a way that mimics the layout of real detector.

Imagine a standard histogram of Pixel Forward Detector where bins are simple squares with assigned fixed coordinates (x - disk, y - blade number) - it is not easy to visualize, how exactly bin entries are represented in the detector layout.

Here comes this package - it creates histograms with bins which are laid out in space in a very similar way to the real detector. In case of Forward Detector bins create fans in XY-plane.

Two modes of operation
----------------------

This package can operate in two modes:
   1. *MODE_REMAP* - it requires file with Oline histograms that will be remapped to Offline histograms
   2. *MODE_ANALYZE* - takes file to analyze collections contained in events 
   
How are bins booked and what kind of histograms may be produced
---------------------------------------------------------------

To obtain global position of each module of the Pixel Detector GeomDet::surface() is used. It returns the reference to the object that approximates real detector - position and dimensions (width, lenght, thickness). Moreover, so that to create more complicated layouts (like in forward detector) also orientation in space is required (GeomDet::rotation()). Based on this data transformation matrix is built (only in Forward case). As vertices for a given geometry unit are ready new bin in the appropriate TH2Poly histogram is created with an ID that is equal to the module rawId. 

Types of booked histograms:

 - `barrel_layer_n`, [1,...,4],
 - `barrel_summary` (keeps all 4 layers in one histogram),
 - `forward_disk_n`, [-3,...,3],
 - `forward_summary` (keeps all 6 disks in one histogram like here, disks on the negative side of x-axis correspond to disks on the negative side of the barrel, the higher the center of the disk the higher is also disk number).
 
Barrel histograms do not differ much from Online barrel histograms. Here is just Offline naming convention used. X axis represents global z-coordinate when y axis is a simple offline ladder number.

Concentric shape of Forward Disks and the existence of two panels on each of them forces another treatment. First of all each geometry unit (nth offline blade) is of trapezoidal shape in XY-plane which is split by the diagonal into two triangles:

 - smaller one represents the data of panel 1,
 - bigger one represents the data of panel 2. 

Secondly, vertices are multiplied by the transformation matrix which is a mean of two transformation matrices to obtain final layout.

How to configure the package
----------------------------

   1. *MODE_REMAP*
     * it is required to provide a dummy file with at least one event to make the analyze() method fire (`fileNames = cms.untracked.vstring("file:RECO_file.root")`)
	 * set `opMode = cms.untracked.uint32(MODE_REMAP)`
	 * give a file in which histograms to remap will be searched for `remapRootFileName = cms.untracked.vstring("dqmFile.root")`
	 * `baseHistogramName`, `pathToHistograms` and `isBarrelSource` are closely related to each other because every nth element describes one histogram set that will be remapped.
	 The plugin will be looking for all histograms which names are provided in `baseHistogramName` in a corresponding directory `pathToHistograms` in file `remapRootFileName` and assumes it is of type (Barrel/Forward : 1/0) as specified in `isBarrelSource`. If everything goes fine there will be produced output file with histograms contained in directories which are named exactly the same as entries in `baseHistogramName`. Besides of booking new histograms each directory contains also source histograms from the original file. If there is a problem with opening source histogram it will be skipped.
   2. *MODE_ANALYZE"
     * set your input file (`fileNames = cms.untracked.vstring("file:RECO_file.root")`)
	 * set `opMode = cms.untracked.uint32(MODE_ANALYZE)`
	 * set the source collection of your inputs `src = cms.InputTag("generalTracks")`, at this point of development only `recHits` are allowed - these are extracted from collection of tracks
	 After running the code in this mode user should get a root file which contains the distribution of __something__ in Barrel and Forward Detector.
	 
DEBUG mode
---------

There is another mode of operation that alters default contentents of REMAP/ANALYZE histograms and creates additional files in the working directory that are used for debugging purposes. You can see the results if you uncomment '#define DEBUG_MODE' and recompile the package.

This mode helps to understand mapping from Online to Offline DQM, produces files with bins' vertices to use by external tools, as well as helps to check if hits positions are properly represented in contents of the bins.