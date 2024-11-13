# TrackLooper


## Quick Start


### Setting up LSTPerformanceWeb (only for lnx7188 and lnx4555)

For lnx7188 and lnx4555 this needs to be done once

    cd /cdat/tem/${USER}/
    git clone git@github.com:SegmentLinking/LSTPerformanceWeb.git

### Setting up container (only for lnx7188)

For lnx7188 this needs to be done before compiling or running the code:

    singularity shell --nv --bind /mnt/data1:/data --bind /data2/segmentlinking/ --bind /opt --bind /nfs --bind /mnt --bind /usr/local/cuda/bin/ --bind /cvmfs  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el8:x86_64

### Setting up the code

    git clone git@github.com:SegmentLinking/TrackLooper.git
    cd TrackLooper/
    # Source one of the commands below, depending on the site
    source setup.sh # if on UCSD or Cornell
    source setup_hpg.sh # if on Florida

### Running the code

    sdl_make_tracklooper -mc
    sdl_<backend> -i PU200 -o LSTNtuple.root
    createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root
    lst_plot_performance.py LSTNumDen.root -t "myTag"
    # python3 efficiency/python/lst_plot_performance.py LSTNumDen.root -t "myTag" # if you are on cgpu-1 or Cornell

The above can be even simplified

    sdl_run -f -mc -s PU200 -n -1 -t myTag

The `-f` flag can be omitted when the code has already been compiled. If multiple backends were compiled, then the `-b` flag can be used to specify a backend. For example

    sdl_run -b cpu -s PU200 -n -1 -t myTag

## Command explanations

Compile the code with option flags. If none of `C,G,R,A` are used, then it defaults to compiling for CUDA and CPU.

    sdl_make_tracklooper -mc
    -m: make clean binaries
    -c: run with the cmssw caching allocator
    -C: compile CPU backend
    -G: compile CUDA backend
    -R: compile ROCm backend
    -A: compile all backends
    -h: show help screen with all options

Run the code
 
    sdl_<backend> -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset> -o <output>

    -i: PU200; muonGun, etc
    -n: number of events; default: all
    -v: 0-no printout; 1- timing printout only; 2- multiplicity printout; default: 0
    -s: number of streams/events in flight; default: 1
    -w: 0- no writeout; 1- minimum writeout; default: 1
    -o: provide an output root file name (e.g. LSTNtuple.root); default: debug.root
    -l: add lower level object (pT3, pT5, T5, etc.) branches to the output

Plotting numerators and denominators of performance plots

    createPerfNumDenHists -i <input> -o <output> [-g <pdgids> -n <nevents>]

    -i: Path to LSTNtuple.root
    -o: provide an output root file name (e.g. num_den_hist.root)
    -n: (optional) number of events
    -g: (optional) comma separated pdgids to add more efficiency plots with different sim particle slices
    
Plotting performance plots

    lst_plot_performance.py num_den_hist.root -t "mywork"

There are several options you can provide to restrict number of plots being produced.
And by default, it creates a certain set of objects.
One can specifcy the type, range, metric, etc.
To see the full information type

    lst_plot_performance.py --help

To give an example of plotting efficiency, object type of lower level T5, for |eta| < 2.5 only.

    lst_plot_performance.py num_den_hist.root -t "mywork" -m eff -o T5_lower -s loweta

NOTE: in order to plot lower level object, ```-l``` option must have been used during ```sdl``` step!

When running on ```cgpu-1``` remember to specify python3 as there is no python.
The shebang on the ```lst_plot_performance.py``` is not updated as ```lnx7188``` works with python2...

    python3 efficiency/python/lst_plot_performance.py num_den_hist.root -t "mywork" # If running on cgpu-1
                                                                                                                                                           
Comparing two different runs

    lst_plot_performance.py \
        num_den_hist_1.root \     # Reference
        num_den_hist_2.root \     # New work
        -L BaseLine,MyNewWork \   # Labeling
        -t "mywork" \
        --compare

## CMSSW Integration
This is the a complete set of instruction on how the TrackLooper code
can be linked as an external tool in CMSSW:

### Build TrackLooper
```bash
git clone git@github.com:SegmentLinking/TrackLooper.git
cd TrackLooper/
# Source one of the commands below, depending on the site
source setup.sh # if on UCSD or Cornell
source setup_hpg.sh # if on Florida
sdl_make_tracklooper -mc
cd ..
```

### Set up `TrackLooper` as an external
```bash
mkdir workingFolder # Create the folder you will be working in
cd workingFolder
cmsrel CMSSW_14_1_0_pre3
cd CMSSW_14_1_0_pre3/src
cmsenv
git cms-init
git remote add SegLink git@github.com:SegmentLinking/cmssw.git
git fetch SegLink CMSSW_14_1_0_pre3_LST_X
git cms-addpkg RecoTracker Configuration
git checkout CMSSW_14_1_0_pre3_LST_X
#To include both the CPU library and GPU library into CMSSW, create 3 xml files (headers file has no library).
#Before writing the following xml file, check that libsdl_cpu.so and libsdl_gpu.so can be found under the ../../../TrackLooper/SDL/ folder.
cat <<EOF >lst_headers.xml
<tool name="lst_headers" version="1.0">
  <client>
    <environment name="LSTBASE" default="$PWD/../../../TrackLooper"/>
    <environment name="INCLUDE" default="\$LSTBASE"/>
  </client>
  <runtime name="LST_BASE" value="\$LSTBASE"/>
</tool>
EOF
cat <<EOF >lst_cpu.xml
<tool name="lst_cpu" version="1.0">
  <client>
    <environment name="LSTBASE" default="$PWD/../../../TrackLooper"/>
    <environment name="LIBDIR" default="\$LSTBASE/SDL"/>
    <environment name="INCLUDE" default="\$LSTBASE"/>
  </client>
  <runtime name="LST_BASE" value="\$LSTBASE"/>
  <lib name="sdl_cpu"/>
</tool>
EOF
cat <<EOF >lst_cuda.xml
<tool name="lst_cuda" version="1.0">
  <client>
    <environment name="LSTBASE" default="$PWD/../../../TrackLooper"/>
    <environment name="LIBDIR" default="\$LSTBASE/SDL"/>
    <environment name="INCLUDE" default="\$LSTBASE"/>
  </client>
  <runtime name="LST_BASE" value="\$LSTBASE"/>
  <lib name="sdl_cuda"/>
</tool>
EOF
scram setup lst_headers.xml
scram setup lst_cpu.xml
scram setup lst_cuda.xml
cmsenv
git cms-checkdeps -a -A
scram b -j 12
```

### Run the LST reconstruction in CMSSW
A simple test configuration of the LST reconstruction can be run with the command:
```bash
cmsRun RecoTracker/LST/test/LSTAlpakaTester.py
```

For a more complete workflow, one can run a modified version of the 21034.1 workflow.
To get the commands of this workflow, one can run:
```bash
runTheMatrix.py -w upgrade -n -e -l 21034.1
```

For convenience, the workflow has been run for 100 events and the output is stored here:
```bash
/data2/segmentlinking/CMSSW_14_1_0_pre0/step2_21034.1_100Events.root
```

For enabling the LST reconstruction in the CMSSW tracking workflow, a modified step3 needs to be run.
This is based on the step3 command of the 21034.1 workflow with the following changes:
   - Remove the `--pileup_input` and `--pileup` flags.
   - The number of threads and streams for the job can be optionally controlled by the `--nThreads` and `--nStreams` command line options respectively (`1` ends up being the actual default value for both, and more info can be found by running `cmsDriver.py --help`).
   - Add at the end of the command: `--procModifiers gpu,trackingLST,trackingIters01 --no_exec`

Run the command and modify the output configuration file with the following:
   - If want to run a cpu version, remove the ```gpu``` in the line defining the `process` object:
     ```python
     process = cms.Process('RECO',...,gpu,...)
     ```
   - Add the following lines below the part where the import of the standard configurations happens:
     ```python
     process.load('Configuration.StandardSequences.Accelerators_cff')
     process.load("HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi")
     ```
   - Modify the input and output file names accordingly, as well as the number of events.

Then, run the configuration file with `cmsRun`.

To get the DQM files, one would have to run step4 of the 21034.1 workflow with the following modifications:
   - Add `--no_exec` to the end of command and then run it.
   - Modify the output configuration file by changing the input file (the one containing `inDQM` from the previous step) and number of events accordingly.

Running the configuration file with `cmsRun`, the output file will have a name starting with `DQM`. The name is the same every time this step runs,
so it is good practice to rename the file, e.g. to `tracking_Iters01LST.root`.
The MTV plots can be produced with the command:
```bash
makeTrackValidationPlots.py --extended tracking_Iters01LST.root
```
Comparison plots can be made by including multiple ROOT files as arguments.

**Note:** In case one wants to run step2 as well, similar modifications as in step4 (`--no_exec` flag and input file/number of events) need to be applied. Moreover, the PU files have better be modified to point to local ones. This can be done by inserting a dummy file when running the command (set the argument of the `--pileup_input` flag to `file:file.root`), and then change the PU input files in the configuration to the following line (by means of replacing the corresponding line in the configuration):
```python
process.mix.input.fileNames = cms.untracked.vstring(['file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/066fc95d-1cef-4469-9e08-3913973cd4ce.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/07928a25-231b-450d-9d17-e20e751323a1.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/26bd8fb0-575e-4201-b657-94cdcb633045.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/4206a9c5-44c2-45a5-aab2-1a8a6043a08a.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/55a372bf-a234-4111-8ce0-ead6157a1810.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/59ad346c-f405-4288-96d7-795f81c43fe8.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/7280f5ec-b71d-4579-a730-7ce2de0ff906.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/b93adc85-715f-477a-afc9-65f3241933ee.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/c7a0aa46-f55c-4b01-977f-34a397b71fba.root', 'file:/data2/segmentlinking/PUSamplesForCMSSW1263/CMSSW_12_3_0_pre5/RelValMinBias_14TeV/GEN-SIM/123X_mcRun4_realistic_v4_2026D88noPU-v1/e77fa467-97cb-4943-884f-6965b4eb0390.root'])
```

### Inclusion of LST in other CMSSW packages
Including the line
```
<use name="lst"/>
```
in the relevant package `BuildFile.xml` allows for
including our headers in the code of that package.

## Running LST in a CVMFS-less setup

The setup scripts included in this repository assume that the [CernVM File System (CVMFS)](https://cernvm.cern.ch/fs/) is installed. This provides a convenient way to fetch the required dependencies, but it is not necessary to run LST in standalone mode. Here, we briefly describe how to build and run it when CVMFS is not available.

The necessary dependencies are CUDA, ROOT, the Boost libraries, Alpaka, and some CMSSW headers. CUDA, ROOT, and Boost, are fairly standard libraries and are available from multiple package managers. For the remaining necessary headers you will need to clone the [Alpaka](https://github.com/alpaka-group/alpaka) and [CMSSW](https://github.com/cms-sw/cmssw) repositories. The Alpaka repository is reasonably sized, but the CMSSW one extremely large, especially considering that we only need a tiny fraction of its files to build LST. We can get only the Alpaka interface headers from CMSSW by running the following commands.

``` bash
git clone --filter=blob:none --no-checkout --depth 1 --sparse --branch CMSSW_14_1_X https://github.com/cms-sw/cmssw.git
cd cmssw
git sparse-checkout add HeterogeneousCore/AlpakaInterface
git checkout
```

Then all that is left to do is set some environment variables. We give an example of how to do this in lnx7188/cgpu-1.

```bash
# These two lines are only needed to set the right version of gcc and nvcc. They are not needed for standard installations.
export PATH=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/bin:/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/external/el8_amd64_gcc12/bin:$PATH
export LD_LIBRARY_PATH=/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/biglib/el8_amd64_gcc12:/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/lib/el8_amd64_gcc12:/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/external/el8_amd64_gcc12/lib:/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/lib64:/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/lib:$LD_LIBRARY_PATH

# These are the lines that you need to manually change for a CVMFS-less setup.
# In this example we use cvmfs paths since that is where the dependencies are in lnx7188/cgpu1, but they can point to local directories.
export BOOST_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/boost/1.80.0-60a217837b5db1cff00c7d88ec42f53a
export ALPAKA_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/alpaka/1.1.0-7d0324257db47fde2d27987e7ff98fb4
export CUDA_HOME=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/cuda/12.4.1-06cde0cd9f95a73a1ea05c8535f60bde
export ROOT_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/lcg/root/6.30.07-21947a33e64ceb827a089697ad72e468
export CMSSW_BASE=/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3

# These lines are needed to account for some extra environment variables that are exported in the setup script.
export LD_LIBRARY_PATH=$PWD/SDL/cuda:$PWD/SDL/cpu:$PWD:$LD_LIBRARY_PATH
export PATH=$PWD/bin:$PATH
export PATH=$PWD/efficiency/bin:$PATH
export PATH=$PWD/efficiency/python:$PATH
export TRACKLOOPERDIR=$PWD
export TRACKINGNTUPLEDIR=/data2/segmentlinking/CMSSW_12_2_0_pre2/
export LSTOUTPUTDIR=.
source $PWD/code/rooutil/thisrooutil.sh

# After this, you can compile and run LST as usual.
sdl_run -f -mc -s PU200 -n -1 -t myTag
```

## Code formatting and checking

The makefile in the `SDL` directory includes phony targets to run `clang-format` and `clang-tidy` on the code using the formatting and checks used in CMSSW. The following are the available commands.

- `make format`
  Formats the code in the `SDL` directory using `clang-format` following the rules specified in `.clang-format`.
- `make check`
  Runs `clang-tidy` on the code in the `SDL` directory to performs the checks specified in `.clang-tidy`.
- `make check-fix`
  Same as `make check`, but fixes the issues that it knows how to fix.
 