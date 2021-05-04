# request_disk will have to be adjusted when skimming large data sets.
# For now, it is a low value so that condor jobs will start queueing 
# more quickly.

condorSubTemplate="""
Executable = {path}/workingArea/skim_{name}.tcsh
Universe = vanilla
Output = {path}/workingArea/skim_out_{name}.txt
Error  = {path}/workingArea/skim_err_{name}.txt
Log  = {path}/workingArea/skim_condor_{name}.log
request_memory = 2000M
request_disk = 800M
batch_name = skim
+JobFlavour = "workday"
Queue Arguments from (
{name}
)
"""


condorSubTemplateCAF="""
Executable = {path}/workingArea/skim_{name}.tcsh
Universe = vanilla
Output = {path}/workingArea/skim_out_{name}.txt
Error  = {path}/workingArea/skim_err_{name}.txt
Log  = {path}/workingArea/skim_condor_{name}.log
request_memory = 2000M
request_disk = 800M
batch_name = skim
+JobFlavour = "workday"
+AccountingGroup = "group_u_CMS.CAF.ALCA"
Queue Arguments from (
{name}
)
"""

# all {{ and }} will become { and } after using format
skimScript = """#!/bin/tcsh

set curDir=$PWD
echo $curDir
cd {base}/src

eval `scramv1 runtime -csh`
cd $curDir

cmsRun {base}/src/Alignment/APEEstimation/test/SkimProducer/skimProducer_cfg.py isTest=False useTrackList=False sample=$1 > skimLog_$1.txt

cat skimLog_$1.txt

# renaming routine starts here
# cut the last word from the matched lines
set filename=`grep 'Using output name' skimLog_$1.txt | rev | cut -d ' ' -f 1 | rev | cut -d . -f 1`
set filepath=`grep 'Using output path' skimLog_$1.txt | rev | cut -d ' ' -f 1 | rev`

rm -f skimLog_$1.txt

echo "Files to be renamed:"
ls ${{filename}}*.root
# rename files that are filename00X.root to filename_(X+1).root (if there was more than output file created, this happens
foreach fi ( ${{filename}}*.root )
    if ( ${{fi}} == "${{filename}}.root" ) then
        # this file always exists
        mv ${{filename}}.root ${{filename}}_1.root
    else
        # extract the number part from filename and remove leading zeros
        set num=`echo ${{fi}} | cut -d '.' -f1 | grep -o '...$' | sed 's/^0*//'`
        set increased=`echo "$num+1" | bc`
        mv ${{fi}} ${{filename}}_${{increased}}.root
    endif
end

echo "Renaming done:"
ls ${{filename}}*.root

# renaming routine ends here

# move files to remove if path is defined

if ( "${{filepath}}" == "" ) then
    echo "No moving is done as no target path is defined"
else
    if (! -d ${{filepath}} ) then
        mkdir ${{filepath}}
    endif
    xrdcp ${{filename}}*.root ${{filepath}}
    rm ${{filename}}*.root
endif
"""

