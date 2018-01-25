# Parallelization script
Parallelization script for Herwig7 in CMSSW

Table of Contents
=================

  * [Parallelization script](#parallelization-script)
    * [Possible Options](#possible-options)
    * [Examples](#examples)
    * [Warnings](#warnings)


## Parallelization script
* This script sets up parallel jobs for the build, integrate and run step when using Herwig with the CMSSW framework.
* It takes a cmsRun file, adjusts the parameters in it accordingly to the options and saves them to temporary cmsRun files. For each step a different cmsRun file is created. The original file remains unaltered.
* In order to adjust the options in the cmsRun file quite simple regular expressions are used. Comments in the file in the process.generator part may confuse this script. Check the temporary cmsRun files if errors occur.

## Possible options:
* -b/--build : sets the number of build jobs and starts the build step.
* -i/--integrate : sets the maximal number of integration jobs
  * This option already has to be set when the build step is invoked. The integration step will be performed if this option is set, unless --nointegration is chosen.
  * The actual number of integration jobs may be smaller. It is determined by the number of files in Herwig-scratch/Build.
* -r/--run : sets the number of run jobs and starts the run step.
  * A parallelized run step is achieved by calling cmsRun an according number of times with different seeds for Herwig. The built in feature of Herwig won't be used.


* --nointegration : use this option to set up several integration jobs without actually performing them
* --stoprun: use this option if you want to create the cmsRun files without calling cmsRun
  * This option may be useful if one wants to change some file names, in order to avoid conflicts if multiple run steps are chosen
* --resumerun: no new cmsRun files for the run step will be created
  * For this option to work 'temporary' cmsRun files complying to the naming scheme have to be availible. Only files up to the number of jobs defined by --run will be considered.
* --keepfiles : don't remove the created temporary cmsRun files
* --l/--log: write the output of each shell command called in a seperate log file
  * This avoids clutter in the terminal. Every process can be watched seperately by `tail -f INSERTNAME.log`

## Examples
### Short example
  * Set up 10 build jobs, 10 integrate jobs and 10 run jobs with the configuration in INSERT\_CMSRUN\_FILENAME.py:
```
./parallelization.py INSERT_CMSRUN_FILENAME.py --build 10 --integrate 10 --run 10
```

### Long example
  * Set up one build job, which prepares 4 integrate jobs, but stopping before the integration:
    * This imitates the behavior of `Herwig build --maxjobs 10`
```
./parallelization.py INSERT_CMSRUN_FILENAME.py --build 1 --integrate 4 --nointegration
```
  * Integrate those jobs:
```
./parallelization.py INSERT_CMSRUN_FILENAME.py --integrate 4
```
  * Create ten cmsRun files (with ten different seeds):
```
./parallelization.py INSERT_CMSRUN_FILENAME.py --run 10 --stoprun
```
  * Now would be the moment to adjust the created files (Maybe change the filenames for the output to ten different names). They are named INSERT\_CMSRUN\_FILENAME\_py\_run\_X.py where X goes from 0 to 9.
  * Execute the cmsRun files:
```
./parallelization.py INSERT_CMSRUN_FILENAME.py --run 10 --resumerun
```
    * Note that the file name is the original one. In this case this file won't be read anymore, it's name just determines the prefix in front of \_py\_run\_X.py .
    * No files will be deleted after this run. This script only deletes files it has created in the same run.

### Additional option
  * If one plans to make a second run with the existing config (or with some small changes, which don't affect build or integrate step), one can keep the files.
```
./parallelization INSERT_CMSRUN_FILENAME.py --build 1 --integrate 1 --run 1 --keepfiles
```
    * Now the files INSERT\_CMSRUN\_FILENAME\_py\_build.py, INSERT\_CMSRUN\_FILENAME\_py\_integrate\_0.py and INSERT\_CMSRUN\_FILENAME\_py\_run\_0.py are kept in the current directory. 

  * Rerun the existing file INSERT\_CMSRUN\_FILENAME\_py\_run\_0.py via:
```
./parallelization INSERT_CMSRUN_FILENAME.py --run 1 --resumerun

```

## Warnings
* Existing files with the same names as described in the naming scheme will be overwritten.
 * in the given file name all dots are replaced by underscores and either \_build.py, \_integrate\_X.py or \_run\_X.py is appended
* Comments in the cmsRun file may confuse this script.











