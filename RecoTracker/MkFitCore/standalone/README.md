# mkFit: a repository for vectorized, parallelized charged particle track reconstruction

**Intro**: Below is a short README on setup steps, code change procedures, and some helpful pointers. Please read this thoroughly before checking out the code! As this is a markdown file, it is best viewed via a web browser.

### Outline
1) Test platforms
2) How to checkout the code
3) How to run the code
4) How to make changes to the main development branch
5) The benchmark and validation suite
   1) Running the main script
   2) Some (must read) advice on benchmarking
   3) (Optional) Using additional scripts to display plots on the web
   4) Interpreting the results
      1) Benchmark results
      2) Validation results
      3) Other plots
6) Submit an issue
7) Condensed description of code
8) Other helpful README's in the repository
9) CMSSW integration
   1) Considerations for `mkFit` code
   2) Building and setting up `mkFit` for CMSSW
      1) Build `mkFit`
         1) Lxplus
         2) Phi3
      2) Set up `mkFit` as an external
      3) Pull CMSSW code and build
   3) Recipes for the impatient on phi3
      1) Offline tracking
      2) HLT tracking (iter0)
   4) More thorough running instructions
      1) Offline tracking
         1) Customize functions
         2) Timing measurements
         3) Producing MultiTrackValidator plots
      2) HLT tracking (iter0)
   5) Interpretation of results
      1) MultiTrackValidator plots
      2) Timing
10) Other useful information
    1) Important Links
    2) Tips and Tricks
       1) Missing Libraries and Debugging
       2) SSH passwordless login for benchmarking and web scripts
    3) Acronyms/Abbreviations

## Section 1: Test platforms

- **phi1.t2.ucsd.edu**: [Intel Xeon Processor E5-2620](https://ark.intel.com/products/64594/Intel-Xeon-Processor-E5-2620-15M-Cache-2_00-GHz-7_20-GTs-Intel-QPI) _Sandy Bridge_ (referred to as SNB, phiphi, phi1)
- **phi2.t2.ucsd.edu**: [Intel Xeon Phi Processor 7210](https://ark.intel.com/products/94033/Intel-Xeon-Phi-Processor-7210-16GB-1_30-GHz-64-core) _Knights Landing_ (referred to as KNL, phi2)
- **phi3.t2.ucsd.edu**: [Intel Xeon Gold 6130 Processor](https://ark.intel.com/products/120492/Intel-Xeon-Gold-6130-Processor-22M-Cache-2_10-GHz) _Skylake Scalable Performance_ (referred to as SKL-Au, SKL-SP, phi3)
- **lnx4108.classe.cornell.edu**: [Intel Xeon Silver 4116 Processor](https://ark.intel.com/products/120481/Intel-Xeon-Silver-4116-Processor-16_5M-Cache-2_10-GHz) _Skylake Scalable Performance_ (referred to as SKL-Ag, SKL-SP, lnx4108, LNX-S)
- **lnx7188.classe.cornell.edu**: [Intel Xeon Gold 6142 Processor](https://ark.intel.com/content/www/us/en/ark/products/120487/intel-xeon-gold-6142-processor-22m-cache-2-60-ghz.html) _Skylake Scalable Performance_ (referred to as lnx7188,LNX-G)

phi1, phi2, and phi3 are all managed across a virtual login server and therefore the home user spaces are shared. phi1, phi2, phi3, lnx7188, and lnx4108 also have /cvmfs mounted so you can source the environment needed to run the code.

The main development platform is phi3. This is the recommended machine for beginning development and testing. Login into any of the machines is achieved through ```ssh -X -Y <phi username>@phi<N>.t2.ucsd.edu```. It is recommended that you setup ssh key forwarding on your local machine so as to avoid typing in your password with every login, and more importantly, to avoid typing your password during the benchmarking (see Section 10.ii.b).

**Extra platform configuration information**
- phi1, phi3, and lnx4108 are dual socket machines and have two identical Xeons on each board
- phi1, phi2, and phi3 all have TurboBoost disabled to disentangle some effects of dynamic frequency scaling with higher vectorization

For further info on the configuration of each machine, use your favorite text file viewer to peruse the files ```/proc/cpuinfo``` and ```/proc/meminfo``` on each machine.

## Section 2: How to checkout the code

The master development branch is ```devel```, hosted on a [public GH repo](https://github.com/trackreco/mkFit) (referred to as ```trackreco/devel``` for the remainder of the README). This is a public repository, as are all forks of this repository. Development for mkFit is done on separate branches within a forked repository. Make sure to fork the repository to your own account first (using the "Fork" option at the top of the webpage), and push any development branches to your own forked repo first.

Once forked, checkout a local copy by simply doing a git clone:

```
git clone git@github.com:<user>/mkFit
```

where ```<user>``` is your GH username if renamed your remote to your username. Otherwise ```<user>``` will be ```origin```.

If you wish to add another user's repo to your local clone, do:

```
git remote add <user> git@github.com:<user>/mkFit
```

This is useful if you want to submit changes to another user's branches. To checkout a remote branch, do:

```
git fetch <user>
git fetch <user> <branch>
git checkout -b <branch> <user>/<branch>
```

## Section 3: How to run the code

As already mentioned, the recommended test platform to run the code is phi3. Checkout a local repo on phi3 from your forked repo. To run the code out-of-the-box from the main ```devel``` branch, you will first need to source the environment:

```
source xeon_scripts/init-env.sh
```

You are free to put the lines from this script in your login scripts (.bashrc, .bash_profile, etc). However, encapsulate them within a function and then call that function upon logging into phi3. We want clean shells before launching any tests. Therefore, if you have any setup that sources something, disable it and do a fresh login before running any tests! 

Now compile the code:

```
make -j 32 AVX2:=1
```

To run the code with some generic options, do:

```
./mkFit/mkFit --cmssw-n2seeds --input-file /data2/slava77/samples/2017/pass-c93773a/initialStep/PU70HS/10224.0_TTbar_13+TTbar_13TeV_TuneCUETP8M1_2017PU_GenSimFullINPUT+DigiFullPU_2017PU+RecoFullPU_2017PU+HARVESTFullPU_2017PU/memoryFile.fv3.clean.writeAll.CCC1620.recT.082418-25daeda.bin --build-ce --num-thr 64 --num-events 20
```

Consult Sections 7-8 for where to find more information on descriptions of the code, which list resources on where to find the full set of options for running the code.

There are ways to run this code locally on macOS. Instructions for how to to this will be provided later. You will need to have XCode installed (through the AppStore), XCode command line tools, a ROOT6 binary (downloaded from the ROOT webpage), as well as TBB (through homebrew). 

## Section 4: How to make changes to the main development branch

Below are some rules and procedures on how to submit changes to the main development branch. Although not strictly enforced through settings on the main repo, please follow the rules below. This ensures we have a full history of the project, as we can trace any changes to compute or physics performance that are introduced (whether intentional or unintentional). 

**Special note**: Do not commit directly to ```cerati/devel```! This has caused issues in the past that made it difficult to track down changes in compute and physics performance. Please always submit a Pull Request first, ensuring it is reviewed and given the green light before hitting "Merge pull request". 

1. Checkout a new branch on your local repo: ```git checkout -b <branch>```
2. Make some changes on your local repo, and commit them to your branch: ```git commit -m "some meaningful text describing the changes"```
3. If you have made multiple commits, see if you can squash them together to make the git history legibile for review. If you do not know what you are doing with this, make sure to save a copy of the local branch as backup by simplying checking out a new branch from the branch you are with something like: ```git checkout -b <branch_copy>```. Git provides a [tutorial on squashing commits](https://git-scm.com/book/en/v2/Git-Tools-Rewriting-History).
4. Ensure you have pulled down the latest changes from the main development branch merged into your local development branch. ```git merge cerati devel``` can make a mess, so the preferred option is ```git rebase --onto <new_base_hash> <old_base_hash> <branch>```. CMSSW provides a nice explanation of [this rebase option](https://cms-sw.github.io/tutorial-resolve-conflicts.html).
5. Test locally!
   1. If you have not done so, clone your forked repo onto phi3, checking out your new branch.
   2. Source the environment for phi3 as explained in Section 3.
   3. Compile test: ```make -j 32 AVX2:=1```. Fix compilation errors if they are your fault or email the group / person responsible to fix their errors! 
   4. Run benchmark test: ```./mkFit/mkFit --cmssw-n2seeds --input-file /data2/slava77/samples/2017/pass-4874f28/initialStep/PU70HS/10224.0_TTbar_13+TTbar_13TeV_TuneCUETP8M1_2017PU_GenSimFullINPUT+DigiFullPU_2017PU+RecoFullPU_2017PU+HARVESTFullPU_2017PU/a/memoryFile.fv3.clean.writeAll.recT.072617.bin --build-ce --num-thr 64 --num-events 20```. Ensure the test did not crash, and fix any segfaults / run-time errors! 
   5. Compile with ROOT test: ```make -j 32 AVX2:=1 WITH_ROOT:=1```. Before compiling, make sure to do a ```make distclean```, as we do not want conflicting object definitions. Fix errors if compilation fails.
   6. Run validation test:  ```./mkFit/mkFit --cmssw-n2seeds --input-file /data2/slava77/samples/2017/pass-4874f28/initialStep/PU70HS/10224.0_TTbar_13+TTbar_13TeV_TuneCUETP8M1_2017PU_GenSimFullINPUT+DigiFullPU_2017PU+RecoFullPU_2017PU+HARVESTFullPU_2017PU/a/memoryFile.fv3.clean.writeAll.recT.072617.bin --build-ce --num-thr 64 --num-events 20 --backward-fit-pca --cmssw-val-fhit-bprm```. Ensure the test did not crash! 
6. Run the full benchmarking + validation suite on all platforms: follow procedure in Section 5 (below)! If you notice changes to compute or physics performance, make sure to understand why! Even if you are proposing a technical two-line change, please follow this step as it ensures we have a full history of changes.
7. Prepare a Pull Request (PR)
   1. Push your branch to your forked repo on GitHub: ```git push <forked_repo_name> <branch>```
   2. [Navigate to the main GH](https://github.com/trackreco/mkFit)
   3. Click on "New Pull Request"
   4. Click on "Compare across forks", and navigate to your fork + branch you wish to merge as the "head fork + compare"
   5. Provide a decent title, give a brief description of the proposed commits. Include a link to the benchmarking and validation plots in the description. If there are changes to the compute or physics performance, provide an explanation for why! If no changes are expected and none are seen, make sure to mention it.
   6. (Optional) Nominate reviewers to check over the proposed changes.
   7. Follow up on review comments! After pushing new commits to your branch, repeat big steps 5 and 6 (i.e. test locally and re-run the validation). Post a comment to the PR with the new plots.
   8. Once given the green light, you can hit "Merge Pull Request", or ask someone else to do it.

## Section 5: The benchmark and validation suite

**Notes on nomenclature**
- "benchmark": these are the compute performance tests (i.e. time and speedup)
- "validation": these are the physics performance tests (i.e. track-finding efficiency, fake rate, etc.)

We often use these words interchangibly to refer to the set of benchmark and validation tests as a single suite. So if you are asked to "run the benchmarking" or "run the validation": please run the full suite (unless specifically stated to run one or the other). In fact, the main scripts that run the full suite use "benchmark" in their name, even though they may refer to both the running of the compute and physics performance tests and plot comparisons.

**Notes on samples**

Currently, the full benchmark and validation suite uses simulated event data from CMSSW for ttbar events with an average 70 pileup collisions per event. The binary file has over 5000 events to be used for high statistics testing of time performance. There also exists samples with lower number of events for plain ttbar no pileup and ttbar + 30 pileup, used to measure the effects on physics performance when adding more complexity. Lastly, there also exists a sample for muon-gun events: 10 muons per event with no pileup. The muon-gun sample is used to show physics performance in a very clean detector environment. All of these samples are replicated on disk on all three platforms to make time measurements as repeatable and representative as possible. 

### Section 5.i: Running the main script

The main script for running the full suite can be launched from the top-level directory with:

```
./xeon_scripts/runBenchmark.sh ${suite} ${useARCH} ${lnxuser}
```

There are three options for running the full suite by passing one of the three strings to the parameter ```${suite}```:
- ```full``` : runs compute and physics tests for all track finding routines (BH, STD, CE, FV)
- ```forPR``` : runs compute and physics tests for track finding routines used for comparisons in pull requests (default setting: BH and CE for benchmarks, STD and CE for validation)
- ```forConf``` : runs compute and physics tests for track finding routines used for conferences only (currently only CE)

The ```full``` option currently takes little more than a half hour, while the other tests take about 25 minutes. 

Additionally, the ```${useARCH}``` option allows the benchmarks to be run on different computer clusters: 
- ```${useARCH} = 0```: (default) runs on phi3 computers only. This option should be run from phi3.
- ```${useARCH} = 1```: runs on lnx7188 and lnx4108 only. This option should be run from lnx7188.
- ```${useARCH} = 2```: runs on both phi3 and lnx. This option should be run from phi3.
- ```${useARCH} = 3```: runs on both all phi computers (phi1, phi2 and phi3). This option should be run from phi3.
- ```${useARCH} = 4```: runs on both all phi computers (phi1, phi2 and phi3) as well as lnx7188 and lnx4108. This option should be run from phi3.


- ```${lnxuser}``` denotes the username on the lnx computers. This is only need if running on the lnx computers when the lnx username is different from the phi3 username.  

Inside the main script, tests are submitted for phi1, phi2, and phi3 concurrently by: tarring up the local repo, sending the tarball to a disk space on the remote platform, compiling the untarred directory natively on the remote platform, and then sending back the log files to be analyzed on phi3. It should be noted that the tests for phi3 are simply run on in the user home directory when logged into phi3 (although we could in principle ship the code to the work space disk on phi3). Because we run the tests for phi3 in the home directory, which is shared by all three machines, we pack and send the code to a remote _disk_ space _before_ launching the tests on phi3 from the home directory. The scripts that handle the remote testing are: 

```
./xeon_scripts/tarAndSendToRemote.sh ${remote_arch} ${suite}
./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build-remote.sh ${ben_arch} ${suite}
```

When these scripts are called separately to run a test on particular platform, one of three options must be specified for ```${remote_arch}``` or ```${ben_arch}```: ```SNB```, ```KNL```, or ```SKL-SP```. The main script ```xeon_scripts/runBenchmark.sh``` will do this automatically for all three platforms. If the code is already resident on a given machine, it is sufficient to run:

```
./xeon_scripts/benchmark-cmssw-ttbar-fulldet-build.sh ${ben_arch} ${suite}
```

The appropriate strings should appear in place of ```${ben_arch}``` and ```${suite}```. In fact, this is the script called by ```xeon_scripts/runBenchmark.sh``` to launch tests on each of the platforms once the code is sent and unpacked.

Within the main ```xeon_scripts/runBenchmark.sh``` script, there are two other scripts that make performance plots from the log files of compute performance tests:

```
./plotting/benchmarkPlots.sh ${suite} 
./plotting/textDumpPlots.sh ${suite}
```

The first will produce the time and speedup plots, while the second produces distributions of basic kinematic quantites of the candidate track collections, comparing the results across the different platforms and different number of vector units and threads.

The main physics performance script that is run is:

```
./val_scripts/validation-cmssw-benchmarks.sh ${suite}
```

The physics validation scripts supports also an option to produce results compatible with the standard tracking validation in CMSSW, the MultiTrackValidator (MTV). This can run as:
```
./val_scripts/validation-cmssw-benchmarks.sh ${suite} --mtv-like-val
```

This script will run the validation on the building tests specified by the ```${suite}``` option. It will also produce the full set of physics performance plots and text files detailing the various physics rates.

It should be mentioned that each of these scripts within ```./xeon_scripts/runBenchmark.sh``` can be launched on their own, as again, they each set the environment and run tests and/or plot making. However, for simplicity's sake, it is easiest when prepping for a PR to just run the master ```./xeon_scripts/runBenchmark.sh```.  If you want to test locally, it is of course possible to launch the scripts one at a time.

### Section 5.ii: Some (must read) advice on benchmarking

1. Since the repo tarball and log files are sent back and forth via ```scp``` in various subscripts, it is highly recommended you have SSH-forwarding set up to avoid having to type your password every time ```scp``` is called. This can be particularly annoying since the return of the log files is mostly indeterminate, since it is just when the scripts finish running on the remote they will be sent back. Coupled with ```nohup``` when launching the main script, the prompt will never appear, and the log files will then be lost, as the final step in remote testing is removing the copy of repo on the remote platform at the end of ```xeon_scripts/benchmark-cmssw-ttbar-fulldet-build-remote.sh```. See Section 10.ii.b for more information on how to set up SSH-forwarding and passwordless login.

2. Before launching any tests, make sure the machines are quiet: we don't want to disturb someone who already is testing! Tests from different users at the same time will also skew the results of your own tests as the scripts make use of the full resources available on each platform at various points. 

3. Please run the full suite from phi3 with a clean login: make sure nothing has been sourced to set up the environment. The main script (as well as the called subscripts) will set the environment and some common variables shared between all subscripts by sourcing two scripts:

4. Check the logs! A log with standard out and error is generated for each test launched. If a plot is empty, check the log corresponding to the test point that failed as this will be the first place to say where and how the test died (hopefully with a somewhat useful stack trace). If you are sure you are not responsible for the crash, email the group listserv to see if anyone else has experienced the issue (attaching the log file(s) for reference). If it cannot be resolved via email, it will be promoted to the a GH Issue.

```
source xeon_scripts/init-env.sh
source xeon_scripts/common-variables.sh ${suite}
```

### Section 5.iii: (Optional) Using additional scripts to display plots on the web

After running the full suite, there is an additional set of scripts within the ```web/``` directory for organizing the output plots and text files for viewing them on the web. Make sure to read the ```web/README_WEBPLOTS.md``` first to setup an /afs or /eos web directory on LXPLUS. If you have your own website where you would rather post the results, just use ```web/collectBenchmarks.sh``` to tidy up the plots into neat directories before sending them somewhere else. More info on this script is below.

The main script for collecting plots and sending them to LXPLUS can be called by:

```
./web/move-benchmarks.sh ${outdir_name} ${suite} ${afs_or_eos}
```

where again, ```${suite}``` defaults to ```forPR```. ```${outdir_name}``` will be the top-level directory where the output is collected and eventually shipped to LXPLUS. This script first calls ```./web/collectBenchmarks.sh ${outdir_name} ${suite}```, which will sort the files, and then calls the script ```./web/copyphp.sh```, which copies ```web/index.php``` into the ```${outdir_name}``` to have a nice GUI on the web, and finally calls ```./web/tarAndSendToLXPLUS.sh ${outdir_name} ${suite} ${afs_or_eos}```, which packs up the top-level output dir and copies it to either an /afs or /eos userspace on LXPLUS. 

The option ```${afs_or_eos}``` takes either of the following arguments: ```afs``` or ```eos```, and defaults to ```eos```. The mapping of the username to the remote directories is in ```web/tarAndSendToLXPLUS.sh```. If an incorrect string is passed, the script will exit. 

**IMPORTANT NOTES**
1) AFS is being phased out at CERN, so the preferred option is ```eos```.

2) There are some assumptions on the remote directory structure, naming, and files present in order for ```web/tarAndSendToLXPLUS.sh``` to work. Please consult ```web/README_WEBPLOTS.md``` for setting this up properly!

**IMPORTANT DISCLAIMERS**

1. There is a script: ```./xeon_scripts/trashSKL-SP.sh``` that is run at the very end of the ```./web/move-benchmarks.sh``` script that will delete log files, pngs, validation directories, root files, and the neat directory created to house all the validation plots.  This means that if the scp fails, the plots will still be deleted locally, and you will be forced to re-run the whole suite!!  You can of course comment this script out if this bothers you.

2. ```web/tarAndSendToLXPLUS.sh``` executes a script remotely on LXPLUS when using AFS, which makes the directory readable to outside world. If you are uncomfortable with this, you can comment it out. If your website is on EOS, then please ignore this disclaimer.

### Section 5.iv: Interpreting the results

This section provides a brief overview in how to interpret the plots and logs from the tests that produced them. This section assumes the plots were organized with the ```web/collectBenchmarks.sh``` script.

#### Section 5.iv.a: Benchmark results

The "main" benchmark plots are organized into two folders:
- Benchmarks: Will contain plots of the form ```${ben_arch}_CMSSW_TTbar_PU70_${ben_test}_${ben_trend}```
- MultEvInFlight: Will contain plots of the form ```${ben_arch}_CMSSW_TTbar_PU70_MEIF_${ben_trend}```

where the variables in the plot names are:  
- ```${ben_arch}```: SNB (phi1 results), KNL (phi2 results), or SKL (phi3 results)
- ```${ben_test}```: VU (vector units) or TH (threads)
- ```${ben_trend}```: time or speedup, i.e. the y-axis points

The plots in "Benchmarks" measure the time of the building sections only. These tests run over 20 events total, taking the average to measure the per event time for each building section. We discard the first event's time when computing the timing. The logs used for extracting the time information into plots are of the form: ```log_${ben_arch}_CMSSW_TTbar_PU70_${build}_NVU${nVU}_NTH${nTH}.txt```, where ```${build}``` is the building routine tested. 

The plots in "MultEvInFlight" measure the perfomance of the full event loop time which includes I/O, seed cleaning, etc. These tests run over 20 events times the number of events in flight. The time plotted is the total time for all events divided by the number of events.

The points in the speedup plots are simply produced by dividing the first point by each point in the trend. The ideal scaling line assumes that with an N increase in resources, the speedup is then N, i.e. the code is fully vectorized and parallelized with no stalls from memory bandwidth, latency, cache misses, etc. Ideal scaling also assumes no penalty from [dynamic frequency scaling](https://en.wikichip.org/wiki/intel/frequency_behavior). Intel lowers the base and turbo frequency as a function of the occupancy of the number of cores, which can make speedup plots look much worse than they really are. In addition, different instruction sets have different base and turbo frequency settings. Namely, SSE has the highest settings, AVX2 is at the midpoint, while AVX512 has the lowest.

The "VU" tests measure the performance of the building sections as a function of the vector width. In hardware, of course, vector width is a fixed property equal to the maximum number of floats that can be processed by a VPU. Call this number N_max. One can force the hardware to underutilize its VPUs by compiling the code with an older instruction set, e.g., SSE instead of AVX; however, this would have effects beyond just shrinking the vectors. Therefore, for our "VU" tests, we mimic the effect of reducing vector width by setting the width of Matriplex types to various nVU values up to and including N_max. At nVU=1, the code is effectively serial: the compiler might choose not to vectorize Matriplex operations at all. At the maximum size, e.g. nVU=16 on SKL, Matriplex operations are fully vectorized and the VPU can be fully loaded with 16 floats to process these operations. For intermediate values of nVU, full-vector instructions probably will be used, but they may be masked so that the VPU is in reality only partially utilized.

The vectorization tests only use a single thread. There is an additional point at the VU=N_max (SNB: 8, KNL, SKL: 16) with an open dot: this is a measure of the vectorization using intrinsics. 

The "TH" tests measure the performance of the building sections as a function of the number of threads launched. These tests have vectorization fully enabled with instrinsics. It should be noted that we do not account for frequency scaling in the speedup plots.

The building section has sections of code that are inherently serial (hit chi2 comparisons, copying tracks, etc.), so the vectorization and parallelization is not perfect. However, it is important to consider the effect of [Amdahl's Law](https://en.wikipedia.org/wiki/Amdahl%27s_law). Amdahl's law can be rewritten as:
```
    1-1/S
p = -----
    1-1/R
```

where, ```p``` is the fraction of the code that is vectorized/parallelized, ```S``` is the measured speedup, and ```R``` is the amount of speedup from increased resources. For example, we have seen that SKL clocks in at about a factor of three in speedup (S=3) for vectorization when fully vectorized (i.e. nVU=R=16), which suggests the code is 70% vectorized. Of course, this assumes no issues with memory bandwidth, cache misses, etc.
 
We have seen that moving from nVU=1 to nVU=2 the improvement is minimal (and sometimes a loss in performance). One hypothetical reason for this (yet unconfirmed) is that the compiler is using an instruction set other than expected: either finding a way to use vector instructions with nVU=1, or choosing not to vectorize at nVU=2. Furthermore, at run time, the CPU will adjust its frequency depending on the instruction set being used (it runs slower for wider vectors). At present, the exact reasons for the detailed shape of the speedup-vs.-nVU curve are unknown.

Lastly, it is important to consider the effects of hyperthreading in the "TH" tests. At nTH=number of cores, we typically see a clear discontinuation in the slope. The main hypothesis is that this is likely due to resource contention as two threads now share the same cache.

#### Section 5.iv.b: Validation results

The physics validation results are organized into two directories:
- SimVal: SimTracks are used as the reference set of tracks
- CMSSWVal: CMSSW tracks are used as the reference set of tracks

Three different matching criteria are used for making associations between reconstructed tracks and reference tracks. Many of the details are enumerated in the validation manifesto, however, for simplicity, the main points are listed here. 

- CMSSW tracks = "initial step" CMSSW tracks, after fitting in CMSSW (i.e. includes outlier rejection)
- Reference tracks must satisfy:
  - "Findable": require least 12 layers (includes 4 seed layers, so only 8 outside of the seed are required)
  - Sim tracks are required to have four hits that match a seed
- To-be-validated reconstructed tracks must satisfy:
  - "Good" tracks: require at least 10 layers
  - If a mkFit track, 4 hits from seed are included in this 10. So, 6 additional hits must be found during building to be considered a "good" track.
  - If a CMSSW track, up to 4 hits are in included, as a seed hit may be missing from outlier rejection. So, a CMSSW track may have to find more than 6 layers during building to be considered a "good" track, as some hits from the seed may have been removed.
- Matching Criteria:
  - SimVal: reco track is matched to a sim track if >= 50% of hits on reco track match hits from a single sim track, excluding hits from the seed
  - CMSSWVal + Build Tracks: reco track is matched to a CMSSW track if >= 50% of hits on reco track match hits from a single CMSSW track, excluding hits from the seed. Given that CMSSW can produce duplicates (although very low), if a reco track matches more than one CMSSW track, the CMSSW track with the highest match percentage is chosen.
  - CMSSWVal + Fit Tracks: reco track is matched to a CMSSW track via a set of binned helix chi2 (track eta and track pT) and delta phi cuts
- Fake = reco track NOT matching a ref. track, excluding matching to non-findable tracks
- Figures of merit: 
  - Efficiency = fraction of findable ref. tracks matched to a reco track
  - Duplicate rate = fraction of matched ref. tracks with more than one match to a reco track
  - Fake rate = fraction of "good" reco tracks without a match to a ref. track 

In case the MTV-like validation is selected with the option ```mtv-like-val```, the above requirements are replaced with the following:
- Reference tracks:
  - Sim tracks required to come from the hard-scatter interaction, originate from R<3.5 cm and |z|<30 cm, and with pseudorapidity |eta|<2.5 (no requirement to have four hits that match a seed)
- All reconstructed tracks are considered "To-be-validated"
- Matching Criteria:
  - Reco track is matched to a sim track if > 75% of hits on reco track match hits from a single sim track (including hits from the seed)

There are text files within these directories that contain the average numbers for each of the figures of merit, which start with "totals\_\*.txt." In addition, these directories contain nHit plots, as well as kinematic difference plots for matched tracks. Best matched plots are for differences with matched reco tracks with the best track score if more than one reco track matches a ref. track. 

#### Section 5.iv.c: Other plots

The last set of plots to consider are those that produce some kinematic distributions from the text file logs, in the directory: "PlotsFromDump." The distributions compare for each building routine run during the benchmarking the differences across platform and vector + thread setup. Ideally, the distributions should have all points lie on top of each other: there should be no dependency on platform or parallelization/vectorization setting for a specific track-finding routine. The text files that produce these plots have nearly the same form as those for benchmarking, except they also have "DumpForPlots" at the very end.

The subdirectory for "Diffs" in "PlotsFromDump" are kinematic difference plots between mkFit and CMSSW. The matching is simple: we compare mkFit to CMSSW tracks for those that share the exact same CMSSW seed (since we clean some seeds out and CMSSW does not produce a track for every seed as well). The printouts that produce the dump have info to compare to sim tracks using the standard 50% hit matching as done in the SimVal. However, we do not produce these plots as it is redundant to the diff plots already in the validation plots.

## Section 6: Submit an issue

It may so happen that you discover a bug or that there is a known problem that needs further discussion outside of private emails/the main list-serv. If so, make sure to open issue on the main repo by clicking on "Issues" on GH, then "Open an issue".  Provide a descriptive title and a description of the issue. Provide reference numbers to relevant PRs and other Issues with"#<number>".  Include a minimal working example to reproduce the problem, attaching log files of error messages and/or plots demonstrating the problem. 

Assign who you think is responsible for the code (which could be yourself!). If you have an idea that could solve the problem: propose it! If it requires a large change to the code, or may hamper performance in either physics or computing, make sure to detail the pros and cons of different approaches. 

Close an issue after it has been resolved, providing a meaningful message + refence to where/how it was resolved.

## Section 7: Condensed description of code

### mkFit/mkFit.cc

This file is where the ```main()``` function is called for running the executable ```./mkFit/mkFit```. The ```main()``` call simply setups the command line options (and lists them), while the meat of the code is called via ```test_standard()```. Some of the command line options will set global variables within mkFit.cc, while others will set the value of variables in the ```Config``` nampespace. Options that require strings are mapped to via enums in the code, with the mapping specified via global functions at the top of mkFit.cc

```test_standard()``` does the majority of the work: running the toy simulation, reading or writing binary files, and running the various tests. The outer loop is a TBB parallel-for over the number of threads used for running multiple-events-in-flight (MEIF). The default is one event in flight. The inner loop is over the number of events specified for that thread. The number of events in total to run over can be specified as a command line option. When running multiple-events-in-flight, in order to have reasonable statistics from variable load from different events, it is advised to have at least 20 events per thread.  When we refer to "total loop time" of the code, we are timing the inner loop section for each event, which includes I/O. However, for the sake of the plots, we simply sum the time for all events and all threads, and divide by the number of events run to obtain an average per event time.

Within the inner loop, a file is read in, then the various building and fitting tests are run. At the end of each event there is optional printout, as well as at the end of all tthe events for a thread. If running the validation with multiple-events-in-flight is enabled, you will have to ```hadd``` these files into one file before making plots. This is handled automatically within the scripts. 

### mkFit/buildtestMPlex.[h,cc]

This code calls the various building routines, setting up the event, etc. The functions defined here are called in mkFit.cc. Functions called within this file are from MkBuilder.

### mkFit/MkBase.h + mkFit/MkFitter.[h,cc] + mkFit/MkFinder.[h,cc]

MkFinder and MkFitter derive from MkBase. High-level code for objects used by building and fitting routines in mkFit. These objects specify I/O operations from standard format to Matriplex format for different templated Matriplex objects (see Matrix[.h,.cc] for template definitions). 

### mkFit/MkBuilder.[h,cc]

Specifies building routines, seed prepping, validation prepping, etc. Code for building and backward fit routines using MkFinders, while seed fitting uses MkFitters. Objects from Event object are converted to their Matriplex-ready formats. Uses the layer plan to navigate which layer to go to for each track. Foos for the navigation are defined in SteerinParams.h.

### Math/ directory

Contains SMatrix headers, used for some operations on track objects (mostly validation and deprecated SMatrix building code -- see below).

### Matriplex/ directory

Contains low-level Matriplex library code for reading/writing into matriplex objects as well as elementary math operations (add, multiply). Includes perl scripts to autogenerate code based on matrix dimension size.

### Geoms/ dir + TrackerInfo.[h,cc]

Geometry plugin info. TrackerInfo setups classes for layer objects. Geoms/ dir contains the actual layout (number scheme, layer attributes, etc) for each of the different geoemetries.

### mkFit/PropagationMPlex.[h,cc,icc] + mkFit/KalmanUtilsMPlex.[h,cc,icc]

Underlying code for propagation and Kalman upate (gain) calculations in Matriplex form. The .icc files contain the low-level computations. Chi2 computations specified in KalmanUtilsMPlex.

### mkFit/CandCloner.[h,cc]

Code used in Clone Engine for bookkeeping + copying candidates after each layer during building. 

### mkFit.HitStructures.[h,cc]

Specifies MkBuilder + Matriplex friendly data formats for hits. Hits are placed in these containers before building.

### Event.[h,cc]

Most of the code is vestigial (see below). However, the Event object is a container for the different track collections and hit collection. There is code for seed processing, namely cleaning. There is also code relevant for validation and validation prep for different track collections.

### Hit.[h,cc] + Track.[h,cc]

Contain the Hit, Track, and TrackExtra classes. These are the "native" formats read from the binary file (read in from the Tracking NTuple). In principle, since we are planning to migrate to CMSSW eventually, these classes (as well Event) may be trimmed to just read straight from CMSSW native formats.

- Hit object contains hit parameters, covariance, and a global ID. The global ID is used for gaining more information on the MC generation of that hit.
- Track object is simply the track parameters, covariance, charge, track ID, and hit indices + layers. 
- TrackExtra contains additional information about each track, e.g. associated MC info, seed hits, etc. A Track's TrackExtra is accessed through the track label, which is the index inside the vector of tracks. 

### Config.[h,cc]

Contains the Config namespace. Specifies configurable parameters for the code. For example: number of candidates to create for each track, chi2 cut, number of seeds to process per thread, etc. Also contains functions used for dynamically setting other parameters based on options selected. 

Tracker Geometry plugin also initialized here.

### Validation code

Described in validation manifesto. See Section 8 for more info on manifesto.

### TO DO

- flesh out sections as needed
- GPU specific code?

### Vestigial code

There are some sections of code that are not in use anymore and/or are not regularly updated. A short list is here:
- main.cc : Old SMatrix implementation of the code, which is sometimes referred to as the "serial" version of the code.
- USolids/ : Directory for implementing USolids geometry package. Originally implemented in SMatrix code.
- seedtest[.h,.cc] : SMatrix seeding
- buildtest[.h,.cc] : SMatrix building
- fittest[.h,.cc] : SMatrix fitting
- ConformalUtils[.h,.cc] : SMatrix conformal fitter for seeding/fitting
- (possibly) Propagation[.h,.cc] : currently in use by the currently defunct Simulation[.h,.cc]. In reality, will probably move simulation code to MPlex format, which will deprecate this code.
- KalmanUtils[.h,.cc] : SMatrix Kalman Update
- mkFit/seedtestMPlex[.h,.cc] and all code in MkBuilder[.h,.cc] related to finding seeds with our own algorithm
- mkFit/ConformalUtils[.h,.cc] : used by the seeding, although could be revived for fitting
- additional val_scripts/ and web/ scripts not automatically updated outside of main benchmarking code
- mtorture test/ code 

## Section 8: Other helpful README's in the repository

Given that this is a living repository, the comments in the code may not always be enough. Here are some useful other README's within this repo:
- afer compiling the code, do: ```./mkFit/mkFit --help``` : Describes the full list of command line options, inputs, and defaults when running mkFit. The list can also be seen in the code in mkFit/mkFit.cc, although the defaults are hidden behind Config.[h,cc], as well as mkFit.cc.
- cmssw-trackerinfo-desc.txt : Describes the structure of the CMS Phase-I geometry as represented within this repo.
- index-desc.txt : Desribes the various hit and track indices used by different sets of tracks throughout the different stages of the read in, seeding, building, fitting, and validation.
- validation-desc.txt : The validation manifesto: (somewhat) up-to-date description of the full physics validation suite. It is complemented by a somewhat out-of-date [code flow diagram](https://indico.cern.ch/event/656884/contributions/2676532/attachments/1513662/2363067/validation_flow_diagram-v4.pdf).
- web/README_WEBPLOTS.md : A short markdown file on how to setup a website with an AFS or EOS directory on LXPLUS (best when viewed from a web browser, like this README).

## Section 9: CMSSW integration

The supported CMSSW version is currently `11_2_0`. The
integration of `mkFit` in CMSSW is based on setting it up as a CMSSW
external.

### Section 9.i: Considerations for `mkFit` code

The multi-threaded CMSSW framework, and the iterative nature of CMS
tracking impose some constraints on `mkFit` code (that are not all met
yet). Note that not all are mandatory per se, but they would make the
life easier for everybody.

* A single instance of `mkFit` should correspond to a single track building iteration
* There should be no global non-const variables
  - Currently there are non-const global variables e.g. in `Config` namespace
* All iteration-specific parameters should be passed from CMSSW to `mkFit` at run time

### Section 9.ii: Building and setting up `mkFit` for CMSSW

#### Section 9.ii.a: Build `mkFit`

To be used from CMSSW the `mkFit` must be built with the CMSSW
toolchain. Assuming you are in an empty directory, the following
recipe will set up a CMSSW developer area and a `mkFit` area there,
and compile `mkFit` using the CMSSW toolchain.

**Note:** The recipes have been tested on `lxplus` and on `phi3`.
Currently there is no working recipe to compile with `icc` on LPC.

##### Section 9.ii.a.a: Lxplus

```bash
cmsrel CMSSW_11_2_0
pushd CMSSW_11_2_0/src
cmsenv
git cms-init
popd
git clone git@github.com:trackreco/mkFit
pushd mkFit
make -j 12 TBB_PREFIX=$(dirname $(cd $CMSSW_BASE && scram tool tag tbb INCLUDE)) CXX=g++ WITH_ROOT=1 VEC_GCC="-march=core2"
popd
```

##### Section 9.ii.a.b: Phi3

```bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
source /opt/intel/bin/compilervars.sh intel64
export SCRAM_ARCH=slc7_amd64_gcc900
cmsrel CMSSW_11_2_0
pushd CMSSW_11_2_0/src
cmsenv
git cms-init
popd
git clone git@github.com:trackreco/mkFit
pushd mkFit
# for gcc CMSSW "default" build:
#   1) call "unset INTEL_LICENSE_FILE", or do not source compilevars.sh above
#   2) replace AVX* with VEC_GCC="-msse3"
make -j 12 TBB_PREFIX=$(dirname $(cd $CMSSW_BASE && scram tool tag tbb INCLUDE)) WITH_ROOT=1 AVX2:=1
popd
```

#### Section 9.ii.b: Set up `mkFit` as an external

Assuming you are in the aforementioned parent directory, the following
recipe will create a scram tool file, and set up scram to use it

```bash
pushd CMSSW_11_2_0/src
cat <<EOF >mkfit.xml
<tool name="mkfit" version="1.0">
  <client>
    <environment name="MKFITBASE" default="$PWD/../../mkFit"/>
    <environment name="LIBDIR" default="\$MKFITBASE/lib"/>
    <environment name="INCLUDE" default="\$MKFITBASE"/>
  </client>
  <runtime name="MKFIT_BASE" value="\$MKFITBASE"/>
  <lib name="MicCore"/>
  <lib name="MkFit"/>
</tool>
EOF
scram setup mkfit.xml
cmsenv
```

#### Section 9.ii.c: Pull CMSSW code and build

The following recipe will pull the necessary CMSSW-side code and build it

```bash
# in CMSSW_11_2_0/src
git cms-remote add trackreco
git fetch trackreco
git checkout -b CMSSW_11_2_0_mkFit_X trackreco/CMSSW_11_2_0_mkFit_X
git cms-addpkg $(git diff $CMSSW_VERSION --name-only | cut -d/ -f-2 | uniq)
git cms-checkdeps -a
scram b -j 12
```

### Section 9.iii Recipes for the impatient on phi3

#### Section 9.iii.a: Offline tracking

`trackingOnly` reconstruction, DQM, and VALIDATION.

```bash
# in CMSSW_11_2_0/src

# sample = 10mu, ttbarnopu, ttbarpu35, ttbarpu50, ttbarpu70
# mkfit = 'all', 'InitialStep', ..., 'InitialStep,LowPtQuadStep', ..., ''
# timing = '', 'framework', 'FastTimerService'
# (maxEvents = 0, <N>, -1)
# nthreads = 1, <N>
# nstreams = 0, <N>
# trackingNtuple = '', 'generalTracks', 'InitialStep', ...
# jsonPatch = '', <path-to-JSON-file>
# for core pinning prepend e.g. for nthreads=8 "taskset -c 0,32,1,33,2,34,3,35" 
#     0,32 will correspond to the same physical core with 2-way hyperthreading
#     the step is 32 for phi3; check /proc/cpuinfo for same physical id
cmsRun RecoTracker/MkFit/test/reco_cfg.py sample=ttbarpu50 timing=1
```
* The default values for the command line parameters are the first ones.
* `mkfit=1` runs MkFit, `0` runs CMSSW tracking
* The job produces `step3_inDQM.root` that needs to be "harvested" to
  get a "normal" ROOT file with the histograms.
* If `maxEvents` is set to `0`, the number of events to be processed
  is set to a relatively small value depending on the sample for short
  testing purposes.
* Setting `maxEvents=-1` means to process all events.
* `nthreads` sets the number of threads (default 1), and `nstreams`
  the number of EDM streams (or events in flight, default 0, meaning
  the same value as the number of threads)
* [TrackingNtuple](https://github.com/cms-sw/cmssw/blob/master/Validation/RecoTrack/README.md#ntuple)
  can be enabled either for general tracks (`generalTracks`) for for
  individual iterations (e.g. `InitialStep`). See
  [here](https://github.com/cms-sw/cmssw/blob/master/Validation/RecoTrack/README.md#using-tracks-from-a-single-iteration-as-an-input)
  for how the track selection MVA and vertex collection are set
  differently between the two modes.
* Iteration configuration can be patched with a JSON file with
  `jsonPatch` parameter (corresponds to `--json-patch` in the
  standalone program)

DQM harvesting
```bash
cmsRun RecoTracker/MkFit/test/reco_harvest_cfg.py
```
* Produces `DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root`

Producing plots
```bash
makeTrackValidationPlots.py --extended --ptcut <DQM file> [<another DQM file>]
```
* Produces `plots` directory with PDF files and HTML pages for
  navigation. Copy the directory to your web area of choice.
* See `makeTrackValidationPlots.py --help` for more options

#### Section 9.iii.b HLT tracking (iter0)

**Note: this subsection has not yet been updated to 11_2_0**

HLT reconstruction

```bash
# in CMSSW_10_4_0_patch1/src

# in addition to the offline tracking options
# hltOnDemand = 0, 1
# hltIncludeFourthHit = 0, 1
cmsRun RecoTracker/MkFit/test/hlt_cfg.py sample=ttbarpu50 timing=1
```
* The default values for the command line parameters are the first ones.
* For options that are same as in offline tracking, see above
* Setting `hltOnDemand=1` makes the strip local reconstruction to be
  run in the "on-demand" mode (which is the default in real HLT but
  not here). Note that `hltOnDemand=1` works only with `mkfit=0`.
* Setting `hltIncludeFourthHit=1` changes the (HLT-default) behavior
  of the EDProducer that converts pixel tracks to `TrajectorySeed`
  objects to include also the fourth, outermost hit of the pixel track
  in the seed.

DQM harvesting (unless running timing)
```bash
cmsRun RecoTracker/MkFit/test/hlt_harvest.py
```

Producing plots (unless running timing)
```bash
makeTrackValidationPlots.py --extended <DQM file> [<another DQM file>]
```

### Section 9.iv More thorough instructions

#### Section 9.iv.a: Offline tracking

**Note: this subsection has not yet been updated to 11_2_0**

The example below uses 2018 tracking-only workflow

```bash
# Generate configuration
runTheMatrix.py -l 10824.1 --apply 2 --command "--customise RecoTracker/MkFit/customizeInitialStepToMkFit.customizeInitialStepToMkFit --customise RecoTracker/MkFit/customizeInitialStepOnly.customizeInitialStepOnly" -j 0
cd 10824.1*
# edit step3*RECO*.py to contain your desired (2018 RelVal MC) input files
cmsRun step3*RECO*.py
```

The customize function replaces the initialStep track building module
with `mkFit`. In principle the customize function should work with any
reconstruction configuration file.

By default `mkFit` is configured to use Clone Engine with N^2 seed
cleaning, and to do the backward fit (to the innermost hit) within `mkFit`.

For profiling it is suggested to replace the
`customizeInitialStepOnly` customize function with
`customizeInitialStepOnlyNoMTV`. See below for more details.

##### Section 9.iv.a.a: Customize functions

* `RecoTracker/MkFit/customizeInitialStepOnly.customizeInitialStepOnly`
  * Run only the initialStep tracking. In practice this configuration
    runs the initialStepPreSplitting iteration, but named as
    initialStep. MultiTrackValidator is included, and configured to
    monitor initialStep. Intended to provide the minimal configuration
    for CMSSW tests.
* `RecoTracker/MkFit/customizeInitialStepOnly.customizeInitialStepOnlyNoMTV`
  * Otherwise same as `customizeInitialStepOnly` except drops
    MultiTrackValidator. Intended for profiling.

##### Section 9.iv.a.b: Timing measurements

There are several options for the CMSSW module timing measurements:

- [FastTimerService](https://twiki.cern.ch/twiki/bin/viewauth/CMS/FastTimerService)
  * Produces timing measurements as histograms in the DQM root file
  * `makeTrackValidationPlots.py` (see next subsection) produces plots of those
     - "Timing" -> "iterationsCPU.pdf", look for "initialStep" histogram and "Building" bin
- Framework report `process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))`
  * Prints module timings to the standard output
  * Look for the timing of `initialStepTrackCandidates`
- [Timing module](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideEDMTimingAndMemory)
  * Prints module timings to the standard output
  * Look for the timing of `initialStepTrackCandidates`


#### Section 9.iv.a.c: Producing MultiTrackValidator plots

The `step3` above runs also the [MultiTrackValidator](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMultiTrackValidator).

To produce the plots, first run the DQM harvesting step

```bash
cmsRun step4_HARVESTING.py
```

which produces a `DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root` file that contains all the histograms. Rename the file to something reflecting the contents, and run

```bash
makeTrackValidationPlots.py --extended --limit-tracking-algo initialStep <DQM file> [<another DQM file> ...]
```

The script produces a directory `plots` that can be copied to any web
area. Note that the script produces an `index.html` to ease the
navigation.

### Section 9.v: Interpretation of results

#### Section 9.v.a: MultiTrackValidator plots

As the recipe above replaces the initialStep track building, we are
interested in the plots of "initialStep" (in the main page), and in
the iteration-specific page the plots on the column "Built tracks".
Technically these are the output of the final fit of the initialStep,
but the difference wrt. `TrackCandidate`s of `MkFitProducer` should be
negligible.

In short, the relevant plots are
- `effandfake*` show efficiency and fake+duplicate rate vs. various quantities
- `dupandfake*` show fake, duplicate, and pileup rates vs. various quantities (pileup rate is not that interesting for our case)
- `distsim*` show distributions for all and reconstructed TrackingParticles (numerators and denominators of efficiencies)
- `dist*` show distributions for all, true, fake, and duplicate tracks (numerators and denominators of fake and duplicate rates)
- `hitsAndPt` and hitsLayers shows various information on hits and layers
- `resolutions*` show track parameter resolutions vs eta and pT
- `residual*` show track parameter residuals (bias) vs eta and pT
- `pulls` shows track parameter pulls
- `tuning` shows chi2/ndof, chi2 probability, chi2/ndof vs eta and pT residual
- `mva1*` show various information on the BDT track selection

The tracking MC truth matching criteria are different from the mkFit
SimVal. In MTV a track is classified as a "true track" (and a matched
SimTrack as "reconstructed") if more than 75 % of the clusters of the
track are linked to a single SimTrack. A cluster is linked to a
SimTrack if the SimTrack has induced any amount of charge to any of
the digis (= pixel or strip) of the cluster.

#### Section 9.v.b: Timing

When looking the per-module timing numbers, please see the following
table for the relevant modules to look for, and what is their purpose.

| **Module in offline** | **Module in HLT** | **Description** |
|-----------------------|-------------------|-----------------|
| `initialStepTrackCandidatesMkFitInput` | `hltIter0PFlowCkfTrackCandidatesMkFitInput` | Input data conversion |
| `initialStepTrackCandidatesMkFit` | `hltIter0PFlowCkfTrackCandidatesMkFit` | MkFit itself |
| `initialStepTrackCandidates` | `hltIter0PFlowCkfTrackCandidates` | Output data conversion |

The MTV timing plot of initialStep "Building" includes the
contributions of all three modules.



## Section 10: Other useful information

### Section 10.i: Important Links

Project Links
- [Main development GitHub](https://github.com/trackreco/mkFit)
- [Our project website](https://trackreco.github.io) and the [GH repo](https://github.com/trackreco/trackreco.github.io-source) hosting the web files. Feel free to edit the website repo if you have contributed a presentation, poster, or paper. 
- Out-of-date and no longer used [project twiki](https://twiki.cern.ch/twiki/bin/viewauth/CMS/MicTrkRnD)
- [Indico meeting page](https://indico.cern.ch/category/8433)
- Vidyo room: Parallel_Kalman_Filter_Tracking
- Email list-serv: mic-trk-rd@cern.ch

Other Useful References
- [CMS Run1 Tracking Paper](https://arxiv.org/abs/1405.6569)
- [CMS Public Tracking Results](https://twiki.cern.ch/twiki/bin/view/CMSPublic/PhysicsResultsTRK)
- [Kalman Filter in Particle Physics, paper by Rudi Fruhwirth](https://inspirehep.net/record/259509?ln=en)
- [Kalman Filter explained simply](https://128.232.0.20/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf)

### Section 10.ii: Tips and Tricks

#### Section 10.ii.a: Missing Libraries and Debugging

When sourcing the environment on phi3 via ```source xeon_scripts/init-env.sh```, some paths will be unset and access to local binaries may be lost. For example, since we source ROOT (and its many dependencies) over CVMFS, there may be some conflicts in loading some applications. In fact, the shell may complain about missing environment variables (emacs loves to complain about TIFF). The best way around this is to simply use CVMFS as a crutch to load in what you need.

This is particularly noticeable when trying to run a debugger. To compile the code, at a minimum, we must source icc + toolkits that give us libraries for c++14. We achieve this through the dependency loading of ROOT through CVMFS (previously, we sourced devtoolset-N to grab c++14 libraries). 

After sourcing and compiling and then running only to find out there is some crash, when trying to load ```mkFit``` into ``gdb`` via ```gdb ./mkFit/mkFit```, it gives rather opaque error messages about missing Python paths.

This can be overcome by loading ```gdb``` over CVMFS: ```source /cvmfs/cms.cern.ch/slc7_amd64_gcc630/external/gdb/7.12.1-omkpbe2/etc/profile.d/init.sh```. At this point, the application will run normally and debugging can commence.

#### Section 10.ii.b: SSH passwordless login for benchmarking scripts and web scripts

When running the benchmarks, a tarball of the working directory will be ```scp```'ed to phi2 and phi1 before running tests on phi3. After the tests complete on each platform, the log files will be ```scp```'ed back to phi3 concurrently. If you do not forward your ssh keys upon login to phi3, you will have to enter your password when first shipping the code over to phi2 and phi1, and also, at some undetermined point, enter it again to receive the logs.

With your favorite text editor, enter the text below into ```~/.ssh/config``` on your local machine to avoid having to type in your password for login to any phi machine (N.B. some lines are optional):

```
Host phi*.t2.ucsd.edu
     User <phi* username>
     ForwardAgent yes
# lines below are for using X11 on phi* to look at plots, open new windows for emacs, etc.
     ForwardX11 yes
     XAuthLocation /opt/X11/bin/xauth
# lines below are specific to macOS	     
     AddKeysToAgent yes 
     UseKeychain yes
```

After the benchmarks run, you may elect to use the ```web/``` scripts to transfer plots to CERN website hosted on either LXPLUS EOS or AFS. The plots will be put into a tarball, ```scp```'ed over, and then untarred remotely via ```ssh```. To avoid typing in your password for the ```web/``` scripts, you will need to use a Kerberos ticket and also modify your ```.ssh/config``` file in your home directory on the _phi_ machines with the text below:

```
Host lxplus*.cern.ch
     User <lxplus username>
     ForwardAgent yes
     ForwardX11 yes
     GSSAPIAuthentication yes
     GSSAPIDelegateCredentials yes
```

The last two lines are specific to Kerberos's handling of ssh, which is installed on all of the _phi_ machines. In order to open a Kerberos ticket, you will need to do:

```
kinit -f <lxplus username>@CERN.CH
```

and then enter your LXPLUS password. Kerberos will keep your ticket open for a few days to allow passwordless ```ssh``` into LXPLUS. After the ticket expires, you will need to enter that same command again. So, even if you only send plots once every month to LXPLUS, this reduces the number of times of typing in your LXPLUS password from two to one :).

### Section 10.iii: Acronyms/Abbreviations:

[Glossary of acronyms from CMS](https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookGlossary)

- AVX: Advanced Vector Extensions [flavors of AVX: AVX, AVX2, AVX512]
- BH: Best Hit (building routine that selects only the best hit per layer when performing track building)
- BkFit: (B)ac(k)wards Fit, i.e. perform a KF fit backwards from the last layer on the track to the first layer / PCA
- BS: Beamspot (i.e. the luminous region of interactions)
- CCC: Charge Cluster Cut, used to remove hits that come from out-of-time pileup
- CE: Clone Engine (building routine that keeps N candidates per seed, performing the KF update after hits have been saved)
- CMS: Compact Muon Solenoid
- CMSSW: CMS Software
- CMSSWVal: CMSSWTrack Validation, use cmssw tracks as reference set of tracks for association
- FV: Full Vector (building routine that uses a clever way of filling Matriplexes of tracks during track building to boost vectorization, current status: deprecated)
- GH: GitHub
- GPU: Graphical Processing Unit
- GUI: Graphical User Interface
- KF: Kalman Filter
- KNL: Knights Landing
- MEIF: Multiple-Events-In-Flight (method for splitting events into different tasks)
- mkFit: (m)atriplex (k)alman filter (Fit)
- MP: Multi-Processing
- MTV: MultiTrackValidator
- N^2: Local seed cleaning algorithm developed by Mario and Slava
- PCA: Point of closest approach to either the origin or the BS
- PR: Pull Request
- Reco: Reconstruction
- SimVal: SimTrack validation, use simtracks as reference set of tracks for association
- SKL-SP: Skylake Scalable Performance
- SNB: Sandy Bridge
- SSE: Streaming SIMD Extensions
- STD: Standard (building routine, like Clone Engine, but performs KF update before hits are saved to a track)
- TBB: (Intel) Threaded Building Blocks, open source library from Intel to perform tasks in a multithreaded environment
- TH: Threads
- VU: (loosely) Vector Units
