#!/bin/bash
#
# This script is the main interface for pattern recognition on
# CMSSW files.
#
# It is called by AMPR.csh

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# You're not supposed to touch anything here
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#
# Case 1: CMSSW pattern recognition on a given bank
#
# Use and customize the script AMPR_base.py
#


if [ ${1} = "PR" ]; then

    INPUT=${2}                # The input file with full address
    BK=${3}                   # The pbk file with full address
    OUTPUT=${4}               # Output file name 
    OUTPUTFULL=${7}/${4}      # The output file with full address
    START=${5}                # The first event to process in the input file
    STOP=${6}                 # The last event to process in the input file
    CMSSW_PROJECT_SRC=${8}    # The CMSSW project release dir
    SEC=${9}                  # The bank number in the directory (for branch name)
    GT=${10}                  # The global tag

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc5_amd64_gcc472
    eval `scramv1 runtime -sh`   
    voms-proxy-info

    cd /tmp/$USER
    TOP=$PWD

    #
    # And we tweak the python generation script according to our needs
    #  

    cd $TOP
    cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMPR_base.py BH_dummy.py 

    # Finally the script is modified according to the requests
    
    sed "s/NEVTS/$STOP/"                                   -i BH_dummy.py
    sed "s/NSKIP/$START/"                                  -i BH_dummy.py
    sed "s#INPUTFILENAME#$INPUT#"                          -i BH_dummy.py
    sed "s#OUTPUTFILENAME#$OUTPUT#"                        -i BH_dummy.py
    sed "s#BANKFILENAME#$BK#"                              -i BH_dummy.py
    sed "s/MYGLOBALTAG/$GT/"                               -i BH_dummy.py
    sed "s/PATTCONT/AML1Patternsb$SEC/"                    -i BH_dummy.py

    cmsRun BH_dummy.py -j4

    # Recover the data
    #  

    ls -l
    lcg-cp file://$TOP/$OUTPUT $OUTPUTFULL

fi

#
# Case 2: CMSSW merging of the files 
#
# Use and customize the script AMPR_MERGER_base.py and AM_FINAL_MERGER_base.py
#

if [ ${1} = "MERGE" ]; then

    TAG=${2}                      # The tag of the files we merge (***_START_STOP.root) 
    INPUTDIR=${3}                 # Input/Output dirs (lcg friendly) 
    INPUTROOTDIR=${4}             # Input/Output dirs (ROOT friendly) 
    OUTPUTFILE="MERGED_"${5}$TAG  # Name of the output file 
    CMSSW_PROJECT_SRC=${6}        # The CMSSW project release dir
    GT=${7}                       # The global tag

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc5_amd64_gcc472
    eval `scramv1 runtime -sh`   
    voms-proxy-info

    cd /tmp/$USER
    TOP=$PWD

    cd $TOP

    rm temp

    compteur=0

    for ll in `lcg-ls $INPUTDIR | grep $TAG | grep ${5}`	
    do

      l=`basename $ll`

      echo $l

      echo "cms.InputTag(\"TTPatternsFromStub\", \"AML1Patternsb"${compteur}"\")" >> temp

      compteur2=$(( $compteur + 1))

      cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMPR_MERGER_base.py BH_dummy.py 

      sed "s#OUTPUTFILENAME#merge_${compteur2}.root#"              -i BH_dummy.py

      if [ $compteur = 0 ]; then # Special treatment for the first merging

	  sed "s#FILE1#$INPUTROOTDIR/$l#"                        -i BH_dummy.py
	  sed "s#FILE2#$INPUTROOTDIR/$l#"                        -i BH_dummy.py

      else # Normal case

	  sed "s#FILE1#file:merge_${compteur}.root#"             -i BH_dummy.py
	  sed "s#FILE2#$INPUTROOTDIR/$l#"                        -i BH_dummy.py

      fi
      
      # Do the merging and remove the previous file 
      cmsRun BH_dummy.py -j4
      rm merge_${compteur}.root
      compteur=$(( $compteur + 1))

    done

    # The first merging step is done, we now have to merge the branches 

    cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMPR_FINAL_MERGER_base.py BH_dummy.py 

    sed "s#OUTPUTFILENAME#$OUTPUTFILE#"                 -i BH_dummy.py
    sed "s#INPUTFILENAME#file:merge_${compteur}.root#"  -i BH_dummy.py
    sed "s/MYGLOBALTAG/$GT/"                            -i BH_dummy.py

    branchlist=`cat temp | tr '\n' ','`
    branchlist2=${branchlist%?}
    #echo $branchlist2

    sed "s#INPUTBRANCHES#$branchlist2#"  -i BH_dummy.py

    cmsRun BH_dummy.py -j4


    # Recover the data
    #
  
    OUTPUTFULL=$INPUTDIR/$OUTPUTFILE

    echo $OUTPUTFULL

    ls -l
    lcg-cp file://$TOP/$OUTPUTFILE $OUTPUTFULL

fi



#
# Case 3: Final CMSSW merging of the files 
#
# When all the ***_START_STOP.root files have been processed
#

if [ ${1} = "FINAL" ]; then

    echo "Doing the final merging"

    TAG=${2} 
    INPUTDIR=${3}  
    INPUTROOTDIR=${4}  
    OUTPUTFILE=${5}  
    CMSSW_PROJECT_SRC=${6}

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc5_amd64_gcc472
    eval `scramv1 runtime -sh`   
    voms-proxy-info

    cd /tmp/$USER
    TOP=$PWD

    cd $TOP

    rm list.txt

    nfiles=`lcg-ls $INPUTDIR | grep $TAG | wc -l` 
	
    for ll in `lcg-ls $INPUTDIR | grep $TAG`	
    do

      l=`basename $ll`
      echo $l
      echo "$INPUTROOTDIR/$l" >> list.txt

      if [ ${nfiles} = "1" ]; then

	  lcg-cp $INPUTDIR/$l file://$TOP/$l
	  cp $l $OUTPUTFILE 

      fi

    done

    # Do the merging (this one is simple)

    if [ ${nfiles} != "1" ]; then

	edmCopyPickMerge inputFiles_load=list.txt outputFile=$OUTPUTFILE 

    fi

    # Recover the data
    #  

    OUTPUTFULL=$INPUTDIR/$OUTPUTFILE

    ls -l
    lcg-cp file://$TOP/$OUTPUTFILE $OUTPUTFULL

fi

#
# Case 4: Fit and extraction 
#
# When the ***_with_AMPR.root files have been processed
#

if [ ${1} = "FIT" ]; then

    echo "Doing the fit"

    INPUT=${2}                # The input xrootd file name and address
    OUTPUT=${3}               # Output file name 
    OUTPUTE=${4}              # Output extracted file name 
    NEVT=${5}                 # #evts/file
    OUTDIR=${6}               # The first event to process in the input file
    CMSSW_PROJECT_SRC=${7}    # The CMSSW project release dir
    GT=${8}                   # The global tag

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc5_amd64_gcc472
    eval `scramv1 runtime -sh`   
    voms-proxy-info

    cd /tmp/$USER
    TOP=$PWD

    #
    # And we tweak the python generation script according to our needs
    #  

    cd $TOP
    cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMFIT_base.py BH_dummy.py 

    # Finally the script is modified according to the requests
    
    sed "s/NEVTS/$NEVT/"                                   -i BH_dummy.py
    sed "s#INPUTFILENAME#$INPUT#"                          -i BH_dummy.py
    sed "s#OUTPUTFILENAME#$OUTPUT#"                        -i BH_dummy.py
    sed "s/MYGLOBALTAG/$GT/"                               -i BH_dummy.py

    cmsRun BH_dummy.py -j4

    # Recover the data
    #  

    ls -l
    lcg-cp file://$TOP/$OUTPUT  ${OUTDIR}/$OUTPUT
    lcg-cp file://$TOP/extracted.root ${OUTDIR}/$OUTPUTE

fi
