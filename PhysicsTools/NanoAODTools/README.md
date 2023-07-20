# NanoAODTools
A simple set of python tools to post-process NanoAODs to: 
* skim events 
* add variables
* produce plots
* perform simple analyses (but beware that performance may be unsatisfactory beacuse of the inherently sequential design model).

It can be used directly from a CMSSW environment, or checked out as a standalone package.

Originally imported to CMSSW from [cms-nanoAOD/nanoAOD-tools](https://github.com/cms-nanoAOD/nanoAOD-tools) (post-processor functionality only).

## Usage in CMSSW
No specific setup is needed.

It is recommended to add external modules (eg correction modules) in separate packages.

## Standalone usage (without CMSSW): checkout instructions

You need to setup python 3 and a recent ROOT version first.

    wget https://raw.githubusercontent.com/cms-sw/cmssw/master/PhysicsTools/NanoAODTools/standalone/checkoutStandalone.sh
    bash checkoutStandalone.sh -d MyProject 
    cd MyProject
    source PhysicsTools/NanoAODTools/standalone/env_standalone.sh

Repeat only the last command at the beginning of every session.

It is recommended to add analysis code, correction modules etc. in a separate package and repository rather than in a CMSSW fork.

Please note that some limitations apply:
* Bindings to C++ code are up to the user.
* Adding packed variables is not supported, as this requires binding to the corresponding code.
Using from a CMSSW environment is thus recommended.

## General instructions to run the post-processing step
The post-processor can be run in two different ways:
* In a normal python script.
* From the command line, using the script under `scripts/nano_postproc.py` (more details [below](#command-line-invocation)).

## How to write and run modules

It is possible to define modules that will be run on each entry passing the event selection, and can be used to calculate new variables that will be included in the output tree (both in friend and full mode) or to apply event filter decisions.

A first, very simple example is available in `test/exampleAnalysis.py`. It can be executed directly, and implements a module to fill a plot.

An example of an module coded to be imported in scripts or called with the command-line interface is available in `python/postprocessing/examples/exampleModule.py`. This module adds one new variable, which can be stored in skimmed NanoAOD and also used in the subsequent Modules in the same job. The example `test/example_postproc.py` shows how to import and use it in a script while skimming events.

Let us now examine the structure of a module class. 
* All modules must inherit from `PhysicsTools.NanoAODTools.postprocessing.framework.eventloop.Module`.
* the `__init__` constructor function should be used to set the module options.
* the `beginFile` function should create the branches that you want to add to the output file, calling the `branch(branchname, typecode, lenVar)` method of `wrappedOutputTree`. `typecode` should be the ROOT TBranch type ("F" for float, "I" for int etc.). `lenVar` should be the name of the variable holding the length of array branches (for instance, `branch("Electron_myNewVar","F","nElectron")`). If the `lenVar` branch does not exist already - it can happen if you create a new collection - it will be automatically created.
* the `analyze` function is called on each event. It should return `True` if the event is to be retained, `False` if it should be dropped.

The event interface, defined in `PhysicsTools.NanoAODTools.postprocessing.framework.datamodule`, allows to dynamically construct views of objects organized in collections, based on the branch names, for instance:

    electrons = Collection(event, "Electron")
    if len(electrons)>1: print electrons[0].someVar+electrons[1].someVar
    electrons_highpt = filter(lambda x: x.pt>50, electrons)

and this will access the elements of the `Electron_someVar`, `Electron_pt` branch arrays. Event variables can be accessed simply by `event.someVar`, for instance `event.rho`.

The output branches should be filled calling the `fillBranch(branchname, value)` method of `wrappedOutputTree`. `value` should be the desired value for single-value branches, an iterable with the correct length for array branches. It is not necessary to fill the `lenVar` branch explicitly, as this is done automatically using the length of the passed iterable.



### Command-line invocation
The basic syntax of the command line invocation is the following:

    nano_postproc.py /path/to/output_directory /path/to/input_tree.root

(in standalone mode, should be invoked as `./scripts/nano_postproc.py`).

Here is a summary of its features:
* the `-s`,`--postfix` option is used to specify the suffix that will be appended to the input file name to obtain the output file name. It defaults to *_Friend* in friend mode, *_Skim* in full mode.
* the `-c`,`--cut` option is used to pass a string expression (using the same syntax as in TTree::Draw) that will be used to select events. It cannot be used in friend mode.
* the `-J`,`--json` option is used to pass the name of a JSON file that will be used to select events. It cannot be used in friend mode.
* if run with the `--full` option (default), the output will be a full nanoAOD file. If run with the `--friend` option, instead, the output will be a friend tree that can be attached to the input tree. In the latter case, it is not possible to apply any kind of event selection, as the number of entries in the parent and friend tree must be the same.
* the `-b`,`--branch-selection` option is used to pass the name of a file containing directives to keep or drop branches from the output tree. The file should contain one directive among `keep`/`drop` (wildcards allowed as in TTree::SetBranchStatus) or `keepmatch`/`dropmatch` (python regexp matching the branch name) per line. More details are provided in the section [Keep/drop branches](#keepdrop-branches) below.
  * `--bi` and `--bo` allows to specify the keep/drop file separately for input and output trees.  
* the `--justcount` option will cause the script to printout the number of selected events, without actually writing the output file.

Please run with `--help` for a complete list of options.

Let's take the already mentioned [exampleModule.py](python/postprocessing/examples/exampleModule.py). It contains a simple constructor:
```
   exampleModuleConstr = lambda : exampleProducer(jetSelection= lambda j : j.pt > 30)
```
whih can be imported using the following syntax:

```
nano_postproc.py outDir /eos/cms/store/user/andrey/f.root -I PhysicsTools.NanoAODTools.postprocessing.examples.exampleModule exampleModuleConstr
```

### Keep/drop branches
See the effect of keep/drop instructions by creating a `keep_and_drop.txt` file:

```
drop *
keep Muon*
keep Electron*
keep Jet*
```
and specify it with thne --bi option:
```
python scripts/nano_postproc.py outDir /eos/cms/store/user/andrey/f.root -I PhysicsTools.NanoAODTools.postprocessing.examples.exampleModule exampleModuleConstr -s _exaModu_keepdrop --bi keep_and_drop_input.txt 
```
comparing to the previous command (without `--bi`), the output branch created by _exampleModuleConstr_ is the same, but with --bi  all other branche are dropped when creating output tree. It also runs faster.
Option --bo can be used to further fliter output branches.  

The keep and drop directive also accept python lists __when called from a python script__, e.g:
```
outputbranchsel=["drop *", "keep EventMass"]
```

### Calling C++ helpers 
Now, let's have a look at another example, `python/postprocessing/examples/mhtjuProducerCpp.py` ([link](python/postprocessing/examples/mhtjuProducerCpp.py)). Similarly, it should be imported using the following syntax:

```
nano_postproc.py outDir /eos/cms/store/user/andrey/f.root -I PhysicsTools.NanoAODTools.postprocessing.examples.mhtjuProducerCpp mhtju
```
This module has the same structure of its producer as `exampleProducer`, but in addition it utilizes a C++ code to calculate the mht variable, `test/examples/mhtjuProducerCppWorker.cc`. This code is loaded in the `__init__` method of the producer.


