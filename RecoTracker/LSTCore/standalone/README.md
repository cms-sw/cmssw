# How to set up standalone LST

## Setting up LSTPerformanceWeb (only for lnx7188 and lnx4555)

For lnx7188 and lnx4555 this needs to be done once

    cd /cdat/tem/${USER}/
    git clone git@github.com:SegmentLinking/LSTPerformanceWeb.git

## Setting up container (only for lnx7188)

For lnx7188 this needs to be done before compiling or running the code:

    singularity shell --nv --bind /mnt/data1:/data --bind /data2/segmentlinking/ --bind /opt --bind /nfs --bind /mnt --bind /usr/local/cuda/bin/ --bind /cvmfs  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el8:x86_64

## Setting up LST

There are two way to set up LST as a standalone, either by setting up a full CMSSW area, which provides a unified setup for standalone and CMSSW tests, or by `sparse-checkout` only the relevant package and using them independent of CMSSW. A CVMFS-less setup is also provided for the second option.

### Setting up LST within CMSSW (preferred option)

```bash
CMSSW_VERSION=CMSSW_14_2_0_pre4 # Change with latest/preferred CMSSW version
cmsrel ${CMSSW_VERSION}
cd ${CMSSW_VERSION}/src/
cmsenv
git cms-init
# If necessary, add the remote git@github.com:SegmentLinking/cmssw.git
# and checkout a development/feature branch
git cms-addpkg RecoTracker/LST RecoTracker/LSTCore
# If modifying some dependencies, run `git cms-checkdeps -a -A`
scram b -j 12
cd RecoTracker/LSTCore/standalone
```

The data files for LST will be fetched from CVMFS. However, if new data files are needed, the need to be manually placed (under `$CMSSW_BASE/external/$SCRAM_ARCH/data/RecoTracker/LSTCore/data/`). This is done by running:

```bash
mkdir -p $CMSSW_BASE/external/$SCRAM_ARCH/data/RecoTracker/LSTCore/
cd $CMSSW_BASE/external/$SCRAM_ARCH/data/RecoTracker/LSTCore/
git clone git@github.com:cms-data/RecoTracker-LSTCore.git data
<modify the files or checkout a different branch>
cd -
```

### Setting up LST outside of CMSSW

For this setup, dependencies are still provided from CMSSW through CVMFS but no CMSSW area is setup. This is done by running the following commands.

``` bash
LST_BRANCH=master # Change to the development branch
git clone --filter=blob:none --no-checkout --depth 1 --sparse --branch ${LST_BRANCH} https://github.com/SegmentLinking/cmssw.git TrackLooper
cd TrackLooper
git sparse-checkout add RecoTracker/LSTCore
git checkout
cd RecoTracker/LSTCore/standalone/
```

As in the sectino above, the data files are fetched from CVMFS, but they can also be copied manually under `RecoTracker/LSTCore/data/`.


## Running the code

Each time the standalone version of LST is to be used, the following command should be run from the `RecoTracker/LSTCore/standalone` directory:
```bash
source setup.sh
```

For running the code:

    lst_make_tracklooper -m
    lst_<backend> -i PU200 -o LSTNtuple.root
    createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root
    lst_plot_performance.py LSTNumDen.root -t "myTag" # or
    python3 efficiency/python/lst_plot_performance.py LSTNumDen.root -t "myTag" # if you are on cgpu-1 or Cornell

The above can be even simplified

    lst_run -f -m -s PU200 -n -1 -t myTag

The `-f` flag can be omitted when the code has already been compiled. If multiple backends were compiled, then the `-b` flag can be used to specify a backend. For example

    lst_run -b cpu -s PU200 -n -1 -t myTag

### Command explanations

Compile the code with option flags. If none of `C,G,R,A` are used, then it defaults to compiling for CUDA and CPU.

    lst_make_tracklooper -m
    -m: make clean binaries
    -C: compile CPU backend
    -G: compile CUDA backend
    -R: compile ROCm backend
    -A: compile all backends
    -h: show help screen with all options

Run the code
 
    lst_<backend> -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset> -o <output>

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

# How to set up CMSSW with LST

## Setting up the area

Follow the instructions in the ["Setting up LST within CMSSW" section](#setting-up-lst-within-cmssw-preferred-option).

## Run the LST reconstruction in CMSSW (read to the end, before running)

Two complete workflows have been implemented within CMSSW to run a two-iteration, tracking-only reconstruction with LST:
 - 24834.703 (CPU)
 - 24834.704 (GPU)

We will use the second one in the example below. To get the commands of this workflow, one can run:

    runTheMatrix.py -w upgrade -n -e -l 24834.704

For convenience, the workflow has been run for 100 events and the output is stored here:

    /data2/segmentlinking/step2_29834.1_100Events.root

The input files in each step may need to be properly adjusted to match the ones produced by the previous step/provided externally, hence it is better to run the commands with the `--no_exec` option included.

Running the configuration file with `cmsRun`, the output file will have a name starting with `DQM`. The name is the same every time this step runs,
so it is good practice to rename the file, e.g. to `step4_24834.704.root`.
The MTV plots can be produced with the command:

    makeTrackValidationPlots.py --extended step4_24834.704.root

Comparison plots can be made by including multiple ROOT files as arguments.

## Code formatting and checking

Using the first setup option above, it is prefered to run the checks provided by CMSSW using the following commands.

```
scram b -j 12 code-checks >& c.log && scram b -j 12 code-format >& f.log
```