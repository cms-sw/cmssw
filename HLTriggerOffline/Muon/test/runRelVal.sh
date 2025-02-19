#!/bin/bash

# Accepts a DBS path, so use looks like:
# runRelVal.sh /RelValZMM/CMSSW_3_1_1-STARTUP31X_V1-v2/GEN-SIM-RECO

# This script creates a copy of hltMuonValidator_cfg.py and includes the
# root files from the chosen sample in the PoolSource, then runs the new 
# ana.py file.  It then runs the copies hltMuonPostProcessor_cfg.py into 
# post.py using the ouput from the analyzer job.

# Use the --post (or -p) option to only run the post processing, using
# validation content already existing in the DBS dataset.

# Use the -n option to specify how many events to run over



getFiles () {
    FILE_NAMES=`mktemp`
    $DBS_CMD lsf --path=$1 | grep .root | \
        sed "s:\(/store/.*\.root\):'\1', :" >> $FILE_NAMES
    echo $FILE_NAMES
}



reduceToMuonContent () {

    if [ $# -ne 2 ]; then echo "Wrong number of arguments"; exit; fi

    cat >> reduceToMuonContent.C << EOF
void CopyDir(TDirectory *source) {
   //copy all objects and subdirs of directory source as a subdir of the current directory   
   source->ls();
   TDirectory *savdir = gDirectory;
   TDirectory *adir = savdir->mkdir(source->GetName());
   adir->cd();
   //loop on all entries of this directory
   TKey *key;
   TIter nextkey(source->GetListOfKeys());
   while ((key = (TKey*)nextkey())) {
      const char *classname = key->GetClassName();
      TClass *cl = gROOT->GetClass(classname);
      if (!cl) continue;
      if (cl->InheritsFrom("TDirectory")) {
         source->cd(key->GetName());
         TDirectory *subdir = gDirectory;
         adir->cd();
         CopyDir(subdir);
         adir->cd();
      } else if (cl->InheritsFrom("TTree")) {
         TTree *T = (TTree*)source->Get(key->GetName());
         adir->cd();
         TTree *newT = T->CloneTree();
         newT->Write();
      } else {
         source->cd();
         TObject *obj = key->ReadObj();
         adir->cd();
         obj->Write();
         delete obj;
     }
  }
  adir->SaveSelf(kTRUE);
  savdir->cd();
}

void reduceToMuonContent(TString oldFileName, TString newFileName) {

  TFile *oldFile = new TFile(oldFileName);
  gDirectory->cd("/DQMData/Run 1/HLT/Run summary/Muon/");
  TDirectory *oldDir = gDirectory;

  TFile *newFile = new TFile(newFileName, "recreate");
  newFile->cd();
  TDirectory *newDir = newFile->mkdir("DQMData");
  newDir = newDir->mkdir("Run 1");
  newDir = newDir->mkdir("HLT");
  newDir = newDir->mkdir("Run summary");
  newDir->cd();
  CopyDir(oldDir);

  newFile->Save();
  newFile->Close();

}

EOF

    root -b -q "reduceToMuonContent.C(\"$1\",\"$2\")" &> /dev/null
    rm reduceToMuonContent.C

}



POST_ONLY=false
N_EVENTS=-1

LONGOPTSTRING=`getopt --long post -o pn: -- "$@"`
eval set -- "$LONGOPTSTRING"
while true ; do
    case "$1" in
        --post) POST_ONLY=true ; shift ;;
        -p)     POST_ONLY=true ; shift ;;
        -n)     shift ; N_EVENTS=$1 ; shift ;;
        --)     shift ; break ;;
        *)      echo "Internal error!" ; exit 1 ;;
    esac
done



if [ -z "$CMSSW_VERSION" ] ; then 
   echo "CMSSW environment not set up; run cmsenv"
   exit
else
   DBS_CMD="python $DBSCMD_HOME/dbsCommandLine.py -c "
fi

if [[ $1 =~ .*GEN-SIM-RECO ]]; then
    HLTDEBUGPATH=`echo $1 | sed 's/GEN-SIM-RECO/GEN-SIM-DIGI-RAW-HLTDEBUG/'`
    RECOPATH=$1
elif [[ $1 =~ .*HLTDEBUG ]]; then
    HLTDEBUGPATH=$1
    RECOPATH=`echo $1 | sed 's/GEN-SIM-DIGI-RAW-HLTDEBUG/GEN-SIM-RECO/'`
elif [[ $1 =~ .*FastSim.* ]]; then
    HLTDEBUGPATH=
    RECOPATH=$1
else
    echo "The given path does not appear to be valid.  Exiting."
    exit
fi

echo "Using dataset(s): "
if [[ ! -z "$HLTDEBUGPATH" ]] ; then echo "    $HLTDEBUGPATH"; fi
if [[ ! -z "$RECOPATH"     ]] ; then echo "    $RECOPATH    "; fi

RECOFILES="$(getFiles $RECOPATH)"
HLTDEBUGFILES="$(getFiles $HLTDEBUGPATH)"

ORIGDIR=`pwd`
TEMPDIR=`mktemp -d`
ANA_PY="ana.py"
POST_PY="post.py"
cd $TEMPDIR

if [ $POST_ONLY = true ]; then
    cp $ORIGDIR/hltMuonPostProcessor_cfg.py $POST_PY
    echo "" >> $POST_PY
    echo "process.source.fileNames = [" >> $POST_PY
    cat $RECOFILES >> $POST_PY
    echo "]" >> $POST_PY
    echo "" >> $POST_PY
    echo "process.maxEvents.input = $N_EVENTS" >> $POST_PY
    cmsRun $POST_PY
    LONGNAME=$RECOPATH
else
    cp $ORIGDIR/hltMuonValidator_cfg.py $ANA_PY
    cp $ORIGDIR/hltMuonPostProcessor_cfg.py $POST_PY
    echo "" >> $ANA_PY
    echo "process.source.fileNames = [" >> $ANA_PY
    cat $RECOFILES >> $ANA_PY
    echo "]" >> $ANA_PY
    echo "process.source.secondaryFileNames = [" >> $ANA_PY
    cat $HLTDEBUGFILES >> $ANA_PY
    echo "]" >> $ANA_PY
    echo "process.maxEvents.input = $N_EVENTS" >> $ANA_PY
    cmsRun $ANA_PY
    cmsRun $POST_PY
    LONGNAME=$RECOPATH
fi

SHORTNAME=`echo $LONGNAME | sed "s/\/RelVal\(.*\)\/CMSSW_\(.*\)\/.*/\1_\2/"`
mv DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root temp.root
reduceToMuonContent temp.root $ORIGDIR/$SHORTNAME.root
cd $ORIGDIR
rm -r $TEMPDIR
echo "Produced $SHORTNAME.root"
