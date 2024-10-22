#!/bin/bash

#script to run generic lhe generation tarballs
#kept as simply as possible to minimize need
#to update the cmssw release
#(all the logic goes in the run script inside the tarball
# on frontier)
#J.Bendavid

#exit on first error
set -e

echo "[MT] NOTE: The script provides a cure for earlier MadGraph LO gridpacks to enable multi-threading (MT). It is also flexible enough to handle all versions of input gridpack, by applying necessary patches and/or fix bugs depending on the gridpack. In this sense, it disobeys the original goal to make this code simplest and keep ALL the logic in 'runcmsgrid.sh' inside the tarball. It is hence expected to switch back to the original 'run_generic_tarball_cvmfs.sh' at a proper time when the new gridpacks are starting to equip with the MT feature."

echo "   ______________________________________     "
echo "         Running Generic Tarball/Gridpack     "
echo "   ______________________________________     "

path=${1}
echo "gridpack tarball path = $path"

nevt=${2}
echo "%MSG-MG5 number of events requested = $nevt"

rnum=${3}
echo "%MSG-MG5 random seed used for the run = $rnum"

ncpu=${4}
echo "%MSG-MG5 thread count requested = $ncpu"

echo "%MSG-MG5 residual/optional arguments = ${@:5}"

if [ -n "${5}" ]; then
  use_gridpack_env=${5}
  echo "%MSG-MG5 use_gridpack_env = $use_gridpack_env"
fi

if [ -n "${6}" ]; then
  scram_arch_version=${6}
  echo "%MSG-MG5 override scram_arch_version = $scram_arch_version"
fi

if [ -n "${7}" ]; then
  cmssw_version=${7}
  echo "%MSG-MG5 override cmssw_version = $cmssw_version"
fi

LHEWORKDIR=`pwd`

if [ "$use_gridpack_env" = false -a -n "$scram_arch_version" -a -n  "$cmssw_version" ]; then
  echo "%MSG-MG5 CMSSW version = $cmssw_version"
  export SCRAM_ARCH=${scram_arch_version}
  scramv1 project CMSSW ${cmssw_version}
  cd ${cmssw_version}/src
  eval `scramv1 runtime -sh`
  cd $LHEWORKDIR
fi

if [[ -d lheevent ]]
    then
    echo 'lheevent directory found'
    echo 'Setting up the environment'
    rm -rf lheevent
fi
mkdir lheevent; cd lheevent

#untar the tarball directly from cvmfs
tar -xaf ${path}

#########################################
# Here starts the new implementation: 
# fix the code depending on the gridpack version to enable multi-thread
#########################################

# exit if the gridpack is not a MG LO one
if [[ ! -e process/madevent/SubProcesses/MGVersion.txt ]]; then
    echo "[MT] Error: this script only works for the MG LO gridpack, while this gridpack might be a MG NLO or non-MG one. Please set 'scriptName' as 'GeneratorInterface/LHEInterface/data/run_generic_tarball_cvmfs.sh' instead."
    exit 1
fi

MGVersion=$(cat process/madevent/SubProcesses/MGVersion.txt)
echo "[MT] Detected MG verion: ${MGVersion}"

MGVersion=(${MGVersion//./ })

if [[ ${MGVersion[1]} -lt 6 ]] || [[ ${MGVersion[1]} -eq 6 && ${MGVersion[2]} -eq 0 ]]; then
    echo "[MT] Warning: multi-threading is not supported in MG version < 2.6.1. Will not activate the multi-thread feature."
elif ([[ ${MGVersion[1]} -eq 6 ]] || [[ ${MGVersion[1]} -eq 7 && ${MGVersion[2]} -le 2 ]]) && [[ -e process/madevent/Cards/MadLoopParams.dat ]]; then
    echo "[MT] Warning: multi-threading is not supported for loop-induced processes in MG version <= 2.7.2. Will not activate the multi-thread feature."
else
    # will activate multi-thread feature
    # first fix a multi-thread bug for MG<=2.7.2
    if [[ ${MGVersion[1]} -eq 6 ]] || [[ ${MGVersion[1]} -eq 7 && ${MGVersion[2]} -le 2 ]]; then 
        echo "[MT] Apply a patch to fix multithread bug in 2.6.1<=MG=2.7.2"
        patch process/madevent/bin/internal/madevent_interface.py << EOF
=== modified file 'madgraph/interface/madevent_interface.py'
--- madgraph/interface/madevent_interface.py	2020-04-23 12:03:18 +0000
+++ madgraph/interface/madevent_interface.py	2020-04-23 15:49:28 +0000
@@ -6667,11 +6667,11 @@
                 sum_axsec += result.get('axsec')*gscalefact[Gdir]
                 
                 if len(AllEvent) >= 80: #perform a partial unweighting
-                    AllEvent.unweight(pjoin(self.me_dir, "Events", self.run_name, "partials%s.lhe.gz" % partials),
+                    AllEvent.unweight(pjoin(outdir, self.run_name, "partials%s.lhe.gz" % partials),
                           get_wgt, log_level=5,  trunc_error=1e-2, event_target=self.nb_event)
                     AllEvent = lhe_parser.MultiEventFile()
                     AllEvent.banner = self.banner
-                    AllEvent.add(pjoin(self.me_dir, "Events", self.run_name, "partials%s.lhe.gz" % partials),
+                    AllEvent.add(pjoin(outdir, self.run_name, "partials%s.lhe.gz" % partials),
                                  sum_xsec,
                                  math.sqrt(sum(x**2 for x in sum_xerru)),
                                  sum_axsec) 
EOF

        # fix another bug related to cross-section computing in multiththread mode, as identified in: https://bugs.launchpad.net/mg5amcnlo/+bug/1884085
        patch process/madevent/bin/internal/madevent_interface.py << EOF
=== modified file 'madgraph/interface/madevent_interface.py'
--- madgraph/interface/madevent_interface.py	2020-06-08 15:08:17 +0000
+++ madgraph/interface/madevent_interface.py	2020-06-18 20:18:17 +0000
@@ -6490,9 +6490,12 @@
             os.chdir(self.me_dir)
         else:
             for line in open(pjoin(self.me_dir,'SubProcesses','subproc.mg')):
-                os.mkdir(line.strip())
+                p = line.strip()
+                os.mkdir(p)
+                files.cp(pjoin(self.me_dir,'SubProcesses',p,'symfact.dat'),
+                         pjoin(p, 'symfact.dat'))
             
 
     def launch(self, nb_event, seed):
EOF
    fi

    # fix another "readonly" mode issue related to proper handle of the integration grid,
    # as identified in: https://answers.launchpad.net/mg5amcnlo/+question/696856
    # this is fixed since 2.9.4 and 3.1.1, so we add a patch if MG is lower then these versions
    if [[ ( ${MGVersion[0]} -eq 2 ) && ( ${MGVersion[1]} -lt 9 || ( ${MGVersion[1]} -eq 9 && ${MGVersion[2]} -le 3 ) ) ]] || \
       [[ ( ${MGVersion[0]} -eq 3 ) && ( ${MGVersion[1]} -lt 1 || ( ${MGVersion[1]} -eq 1 && ${MGVersion[2]} -le 0 ) ) ]]; then 
        patch process/madevent/bin/internal/gen_ximprove.py << EOF
=== modified file 'madgraph/interface/gen_ximprove.py'
--- madgraph/interface/gen_ximprove.py
+++ madgraph/interface/gen_ximprove.py
@@ -1903,5 +1903,8 @@ class gen_ximprove_gridpack(gen_ximprove_v4):
                     'packet': None,
                     }
+            if self.readonly:
+                basedir = pjoin(os.path.dirname(__file__), '..','..','SubProcesses', info['P_dir'], info['directory'])
+                info['base_directory'] = basedir


             jobs.append(info)
EOF
    fi
    
    # fix another multi-thread related bug for MG 2.6.1 only
    if [[ ${MGVersion[1]} -eq 6 && ${MGVersion[2]} -eq 1 ]]; then
        echo "[MT] Apply another patch to fix multithread bug in MG 2.6.1"
        sed -i "/def collect\_result/a\    main_dir = '$(pwd)/process/madevent/SubProcesses'" process/madevent/bin/internal/sum_html.py
    fi
    
    # patch on runcmsgrid.sh if old version is detected
    if grep -q "succ_setreadonly" runcmsgrid.sh; then
        echo "[MT] Congratulations. You are using the new runcmsgrid.sh script with the MG LO multi-thread feature already implemented. Will use this script for event generation without any patch."
    else
        echo "[MT] Old runcmsgrid.sh script detected. This means you are working on an earlier gridpack where MG LO multi-thread feature is not implemented. Will patch on the runcmsgrid.sh code to enable multi-thread feature."
        PATCHDIR=${0%/*}
        cp runcmsgrid.sh runcmsgrid.sh.bak
        cp ${PATCHDIR}/runcmsgrid_LO_support_multithread.patch .
        patch runcmsgrid.sh runcmsgrid_LO_support_multithread.patch
    fi
fi
#########################################


# If TMPDIR is unset, set it to the condor scratch area if present
# and fallback to /tmp
export TMPDIR=${TMPDIR:-${_CONDOR_SCRATCH_DIR:-/tmp}}

#generate events
./runcmsgrid.sh $nevt $rnum $ncpu ${@:5}

mv cmsgrid_final.lhe $LHEWORKDIR/

cd $LHEWORKDIR

#cleanup working directory (save space on worker node for edm output)
rm -rf lheevent

exit 0

