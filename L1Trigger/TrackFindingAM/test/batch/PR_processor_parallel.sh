#!/bin/bash
#
# This script is the main interface for pattern recognition on
# CMSSW files.
#
# It is called by AMPR_parallel.csh

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
    SECBK=${9}                # The bank number in the directory (for branch name)
    GT=${10}                  # The global tag
    SEC=${11}                 # The sector number (default is 6x8 config: 48 sectors)
    INTMP=${12}               # The directory where CMSSW will ran
 

    # Here we decide which threshold we shall use for a given sector
    # In the future, threshold info will be included in the bank name
    #
    # Current default is 4 for hybrid sectors 

    thresh=4
    nmiss=1

    # First we decide the threshold to apply (5 for barrel sectors only)

    if [[ $SEC -ge 16 && $SEC -le 31 ]]; then 
	thresh=5
	nmiss=-1
    fi

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc6_amd64_gcc472
    eval `scramv1 runtime -sh`   

    cd $INTMP
    TOP=$PWD

    #
    # And we tweak the python generation script according to our needs
    #  

    cd $TOP
    cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMPR_base.py BH_dummy_${SECBK}_${OUTPUT}.py 

    # Finally the script is modified according to the requests

    echo "Threshold set to ",$thresh   
 
    sed "s/NEVTS/$STOP/"                                   -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s/NSKIP/$START/"                                  -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s#INPUTFILENAME#file:$INPUT#"                     -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s#OUTPUTFILENAME#$OUTPUT#"                        -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s#BANKFILENAME#$BK#"                              -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s/MYGLOBALTAG/$GT/"                               -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s/THRESHOLD/$thresh/"                             -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s/NBMISSHIT/$nmiss/"                              -i BH_dummy_${SECBK}_${OUTPUT}.py
    sed "s/PATTCONT/AML1Patternsb$SEC/"                    -i BH_dummy_${SECBK}_${OUTPUT}.py

    cmsRun BH_dummy_${SECBK}_${OUTPUT}.py 

    rm BH_dummy_${SECBK}_${OUTPUT}.py 

    # Store the data in the temporary directory
    #  

    mv $TOP/$OUTPUT $OUTPUTFULL

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
    FNAME=${8}                    # A tag to enable parallel processing
    INTMP=${9}                    # 

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc6_amd64_gcc472
    eval `scramv1 runtime -sh`   
 
    cd $INTMP
    TOP=$PWD

    cd $TOP

    rm temp_$FNAME

    compteur=0

    for ll in `\ls $INPUTDIR | grep $TAG | grep ${5}`	
    do

      l=`basename $ll`

      echo $l

      SECNUM=`echo $l | sed s/^.*sec// | cut -d_ -f1` 

      echo "cms.InputTag(\"TTPatternsFromStub\", \"AML1Patternsb"${SECNUM}"\")" >> temp_$FNAME

      echo $SECNUM,$l

      compteur2=$(( $compteur + 1))

      cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMPR_MERGER_base.py BH_dummy_${FNAME}.py 

      sed "s#OUTPUTFILENAME#merge_${compteur2}_${FNAME}.root#"        -i BH_dummy_${FNAME}.py

      if [ $compteur = 0 ]; then # Special treatment for the first merging

	  sed "s#FILE1#file:$INPUTROOTDIR/$l#"                        -i BH_dummy_${FNAME}.py
	  sed "s#FILE2#file:$INPUTROOTDIR/$l#"                        -i BH_dummy_${FNAME}.py

      else # Normal case

	  sed "s#FILE1#file:merge_${compteur}_${FNAME}.root#"         -i BH_dummy_${FNAME}.py
	  sed "s#FILE2#file:$INPUTROOTDIR/$l#"                        -i BH_dummy_${FNAME}.py

      fi
      
      # Do the merging and remove the previous file 
      cmsRun BH_dummy_${FNAME}.py
      rm merge_${compteur}_${FNAME}.root
      compteur=$(( $compteur + 1))

    done

    rm BH_dummy_${FNAME}.py

    # The first merging step is done, we now have to merge the branches 

    cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMPR_FINAL_MERGER_base.py BH_dummy_${FNAME}.py 

    sed "s#OUTPUTFILENAME#$OUTPUTFILE#"                          -i BH_dummy_${FNAME}.py
    sed "s#INPUTFILENAME#file:merge_${compteur}_${FNAME}.root#"  -i BH_dummy_${FNAME}.py
    sed "s/MYGLOBALTAG/$GT/"                                     -i BH_dummy_${FNAME}.py

    branchlist=`cat temp_${FNAME} | tr '\n' ','`
    branchlist2=${branchlist%?}
    echo $branchlist2

    sed "s#INPUTBRANCHES#$branchlist2#"  -i BH_dummy_${FNAME}.py

    cmsRun BH_dummy_${FNAME}.py

    rm BH_dummy_${FNAME}.py
    rm merge_${compteur}_${FNAME}.root

    # Recover the data
    #
  
    OUTPUTFULL=$INPUTDIR/$OUTPUTFILE

    echo $OUTPUTFULL

    #ls -l
    mv $TOP/$OUTPUTFILE $OUTPUTFULL
    rm temp_$FNAME
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
    FNAME=${7}               # A tag to enable parallel processing
    INTMP=${8}               # 

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc6_amd64_gcc472
    eval `scramv1 runtime -sh`   

    cd $INPUTDIR
    TOP=$PWD

    cd $TOP

    rm list_${FNAME}.txt

    nfiles=`\ls $INPUTDIR | grep $TAG | wc -l` 
	
    for ll in `\ls $INPUTDIR | grep $TAG`	
    do

      l=`basename $ll`
      echo $l
      echo "file:$INPUTROOTDIR/$l" >> list_${FNAME}.txt

      if [ ${nfiles} = "1" ]; then

	  cp $INPUTDIR/$l $TOP/$l
	  mv $l $OUTPUTFILE 

      fi

    done

    # Do the merging (this one is simple)

    if [ ${nfiles} != "1" ]; then

	edmCopyPickMerge inputFiles_load=list_${FNAME}.txt outputFile=$OUTPUTFILE 

    fi

    # Recover the data
    #  

    OUTPUTFULL=$INPUTDIR/$OUTPUTFILE

    #ls -l
    mv $TOP/$OUTPUTFILE $OUTPUTFULL
    rm list_${FNAME}.txt
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
    FNAME=${9}                # A tag to enable parallel processing
    INTMP=${10}               # 

    INFILE=`basename $INPUT`
    echo $INPUT,$INFILE
    

    #
    # Setting up environment variables
    #   

    cd $CMSSW_PROJECT_SRC
    export SCRAM_ARCH=slc6_amd64_gcc472
    eval `scramv1 runtime -sh`   

    cd $INTMP
    TOP=$PWD

    mkdir ${INTMP}/RECOVERY

    #
    # And we tweak the python generation script according to our needs
    #  

    cd $TOP
    cp $CMSSW_PROJECT_SRC/src/L1Trigger/TrackFindingAM/test/batch/base/AMFIT_base.py BH_dummy_${FNAME}.py 

    # Finally the script is modified according to the requests
    
    sed "s/NEVTS/$NEVT/"                                   -i BH_dummy_${FNAME}.py
    sed "s#INPUTFILENAME#file:$INPUT#"                     -i BH_dummy_${FNAME}.py
    sed "s#OUTPUTFILENAME#$OUTPUT#"                        -i BH_dummy_${FNAME}.py
    sed "s#EXTRFILENAME#EXTR_$OUTPUT#"                     -i BH_dummy_${FNAME}.py
    sed "s/MYGLOBALTAG/$GT/"                               -i BH_dummy_${FNAME}.py

    cmsRun BH_dummy_${FNAME}.py 

    rm BH_dummy_${FNAME}.py 

    # Recover the data
    #  

    lcg-cp file://$INPUT            ${OUTDIR}/$INFILE
    lcg-cp file://$TOP/$OUTPUT      ${OUTDIR}/$OUTPUT
    lcg-cp file://$TOP/EXTR_$OUTPUT ${OUTDIR}/$OUTPUTE

    deal=`lcg-ls ${OUTDIR}/$INFILE | wc -l`

    if [ $deal = "0" ]; then
	mv $INPUT ${INTMP}/RECOVERY/$INFILE
    fi

    deal=`lcg-ls ${OUTDIR}/$OUTPUT | wc -l`

    if [ $deal = "0" ]; then
	mv $TOP/$OUTPUT ${INTMP}/RECOVERY/$OUTPUT
    fi

    deal=`lcg-ls ${OUTDIR}/$OUTPUTE | wc -l`

    if [ $deal = "0" ]; then
	mv $TOP/EXTR_$OUTPUT ${INTMP}/RECOVERY/$OUTPUTE
    fi

    rm $OUTPUT
    rm EXTR_$OUTPUT
    rm $INPUT

fi
