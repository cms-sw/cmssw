#!/bin/csh


###########################################
#
# Main script for CMSSW AM-based pattern 
# recognition in batch
# 
# !!!Working on EDM files!!!
#
# The jobs themselves are launched by PR_processor.sh
#
# source AMPR.csh p1 p2 p3 p4 p5 p6
# with:
# p1 : The SE subdirectory containing the data file you want to analyze
# p2 : The directory where you will retrieve the bank files, the pattern reco will
#      run over all the pbk files contained in this directory
# p3 : How many events per input data file? 
# p4 : How many events per job (should be below p3...)?
# p5 : The global tag name
# p6 : BATCH or nothing: launch lxbatch jobs or not 
#
# For more details, and examples, have a look at:
# 
# http://sviret.web.cern.ch/sviret/Welcome.php?n=CMS.HLLHCTuto (STEP V.2)
#
# Author: S.Viret (viret@in2p3.fr)
# Date  : 19/02/2014
#
# Script tested with release CMSSW_6_2_0_SLHC7
#
###########################################


# Here we retrieve the main parameters for the job 

set MATTER  = ${1}   # Directory where the input root files are
set BANKDIR = ${2}   # Directory where the bank (.pbk) files are
set NTOT    = ${3}   # How many events per data file?
set NPFILE  = ${4}   # How many events per job?
set GTAG    = ${5}   # Global tag

###################################
#
# The list of parameters you can modify is here
#
###################################

# You have to adapt this to your situation

# Useful for all steps

# The SE directory containing the input EDM file you want to process
set INDIR       = /dpm/in2p3.fr/home/cms/data/store/user/sviret/SLHC/GEN/$MATTER 

# The SE directory containing the output EDM file with the PR output
set OUTDIR      = /dpm/in2p3.fr/home/cms/data/store/user/sviret/SLHC/PR/$MATTER

setenv LFC_HOST 'lyogrid06.in2p3.fr'
set INDIR_GRID   = srm://$LFC_HOST/$INDIR
set INDIR_XROOT  = root://$LFC_HOST/$INDIR
set OUTDIR_GRID  = srm://$LFC_HOST/$OUTDIR
set OUTDIR_XROOT = root://$LFC_HOST/$OUTDIR

###########################################################
###########################################################
# You are not supposed to touch the rest of the script !!!!
###########################################################
###########################################################


# Following lines suppose that you have a certificate installed on lxplus. To do that follow the instructions given here:
#
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideLcgAccess
#

source /afs/cern.ch/cms/LCG/LCG-2/UI/cms_ui_env.csh
voms-proxy-init --voms cms --valid 100:00 -out $HOME/.globus/gridproxy.cert
setenv X509_USER_PROXY ${HOME}'/.globus/gridproxy.cert'


# Then we setup some environment variables

cd  ..
set PACKDIR      = $PWD           # This is where the package is installed 
cd  ../..
set RELEASEDIR   = $CMSSW_BASE    # This is where the release is installed

cd $PACKDIR/batch


echo 'The data will be read from directory: '$INDIR_XROOT
echo 'The pattern reco output files will be written in: '$OUTDIR

lfc-mkdir $OUTDIR

# We loop over the data directory in order to find all the files to process

@ ninput = 0	 

foreach ll (`lcg-ls $INDIR_GRID | grep EDM`) 
   
    set l = `basename $ll`

    @ i = 0
    @ j = $NPFILE

    @ ninput += 1

    # Uncomment this only if you want to limit the number of input files to deal with
    #    if ($ninput > 10) then
    #	continue
    #    endif

    echo 'Working with file '$l

    # First look if the file has been processed

    set OUTF  = `echo $l | cut -d. -f1`"_with_AMPR.root"
    set OUTE  = `echo $l | cut -d. -f1`"_with_FIT.root"
    set OUTD  = `echo $l | cut -d. -f1`"_extr.root"

    set deale = `lcg-ls $OUTDIR_GRID/${OUTE} | wc -l`

    if ($deale != "0") then
	continue
    endif

    set dealf = `lcg-ls $OUTDIR_GRID/${OUTF} | wc -l`

    if ($dealf != "0") then
	rm -f final_job_${OUTF}.sh

	echo "#\!/bin/bash" > fit_job_${OUTF}.sh
	echo "source $PACKDIR/batch/PR_processor.sh FIT $OUTDIR_XROOT/${OUTF} $OUTE $OUTD $NTOT $OUTDIR_GRID $RELEASEDIR $GTAG" >> fit_job_${OUTF}.sh
	chmod 755 fit_job_${OUTF}.sh

	if (${6} == "BATCH") then	
	    bsub -q 1nd -e /dev/null -o /tmp/${LOGNAME}_out.txt fit_job_${OUTF}.sh			
        endif

	continue
    endif

    @ processed = 0
    @ section   = 0

    while ($i < $NTOT)
		
	@ sec = 0
        @ secdone = 0
        @ section += 1


	#
	# First step, we loop over the banks and run the 
	# AM PR on the given data sample
	#
	
	# Check if the file has already been processed

	set OUTM  = `echo $l | cut -d. -f1`
	set dealm = `lcg-ls $OUTDIR_GRID/MERGED_${OUTM}_${i}_${j}.root | wc -l`

	#echo $OUTDIR_GRID/MERGED_${OUTM}_${i}_${j}.root

	if ($dealm != "0") then
	    rm -f merge_job_${OUTM}_${i}_${j}.sh 
	    @ processed += 1
	    @ i += $NPFILE
	    @ j += $NPFILE
	    continue
	endif

        foreach k (`\ls $BANKDIR | grep _sec`) 

	    # By default, for CMSSW, we loop over all available bank in the directory provided
	    echo 'Working with bank file '$k

	    set OUTS1 = `echo $l | cut -d. -f1`_`echo $k | cut -d. -f1`_${i}_${j}

	    # Check if the file has already been processed
	    set deal = `lcg-ls $OUTDIR_GRID/$OUTS1.root | wc -l`

	    if ($deal != "0") then # This process was ran
		rm -f fpr_job_$OUTS1.sh
		@ secdone += 1
		@ sec += 1
		continue
	    endif
	    
	    if ($deal == "0") then

		set running = `\ls fpr_job_$OUTS1.sh | wc -l`

		if ($running == "0") then

		    echo "#\!/bin/bash" > fpr_job_${OUTS1}.sh
		    echo "source $PACKDIR/batch/PR_processor.sh PR ${INDIR_XROOT}/$l $BANKDIR/$k $OUTS1.root  ${i} $NPFILE $OUTDIR_GRID $RELEASEDIR $sec $GTAG" >> fpr_job_${OUTS1}.sh
		    chmod 755 fpr_job_${OUTS1}.sh

		    if (${6} == "BATCH") then	
			bsub -q 1nd -e /dev/null -o /tmp/${LOGNAME}_out.txt fpr_job_${OUTS1}.sh			
		    endif
		endif
	    endif

	    @ sec += 1

	end # End of bank loop

	#
	# Second step, for this given part of the file, all the  
	# banks output are available. We then launch the merging 
	# procedure
	#

	if ($secdone == $sec) then

	    # If not process the file
	    if ($dealm == "0") then

		set running = `\ls merge_job_${OUTM}_${i}_${j}.sh | wc -l`

		if ($running == "0") then

		    echo 'Launching the merging for serie '${i}_${j}' in directory '$OUTDIR_GRID 

		    echo "#\!/bin/bash" > merge_job_${OUTM}_${i}_${j}.sh
		    echo "source $PACKDIR/batch/PR_processor.sh  MERGE ${i}_${j}.root $OUTDIR_GRID $OUTDIR_XROOT ${OUTM}_ $RELEASEDIR $GTAG" >> merge_job_${OUTM}_${i}_${j}.sh
		    chmod 755 merge_job_${OUTM}_${i}_${j}.sh

		    if (${6} == "BATCH") then	
			bsub -q 8nh -e /dev/null -o /tmp/${LOGNAME}_out.txt merge_job_${OUTM}_${i}_${j}.sh 	
		    endif
		endif
	    endif

	endif

	@ i += $NPFILE
	@ j += $NPFILE

    end # End of loop over one input file
	
    #
    # Third step, all the merged files for the given input
    # file have been processed. Then launch the final merging 
    # 

    echo $processed,$section

    if ($processed == $section) then

        # If not process the file
        if ($dealf == "0") then

	    set running = `\ls final_job_${OUTF}.sh | wc -l`

	    if ($running == "0") then

		echo 'Launching the final merging for file '$OUTF' in directory '$OUTDIR_GRID 

		echo "#\!/bin/bash" > final_job_${OUTF}.sh
		echo "source $PACKDIR/batch/PR_processor.sh  FINAL MERGED_${OUTM}_ $OUTDIR_GRID $OUTDIR_XROOT $OUTF $RELEASEDIR" >> final_job_${OUTF}.sh
		chmod 755 final_job_${OUTF}.sh

		if (${6} == "BATCH") then	
		    bsub -q 1nh -e /dev/null -o /tmp/${LOGNAME}_out.txt final_job_${OUTF}.sh			
		endif
	    endif
	endif
    endif
end


