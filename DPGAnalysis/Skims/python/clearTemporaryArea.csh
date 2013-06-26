#!/bin/tcsh 

#
# so: try and have files lasting at least $retention_time; 
#
#
# dquota is the quota of the area
# minfree must be the minimum free area to complete current operations
#

# if disk used more than $maxdisk, delete the oldest ones respecting the previous requirement
# if disk used more than $maxdisk, delete the oldest ones without respecting the previous requirement, but then send a WARNING


set verb=0

set AREA=/afs/cern.ch/cms/CAF/CMSCOMM/COMM_GLOBAL/EventDisplay/RootFileTempStorageArea

#
# in hours
#

set retention_time=1

#
# disk quota (in kB)
#

# this is 10 GB
set dquota=10000000

#
# minfree (in kB)
#

# this is 1 GB
set minfree=1000000

@ maxdisk= $dquota - $minfree

if ($verb) then
    echo Setting maxdisk to $maxdisk
endif
#
# get disk used
#
cd $AREA
set used=`du -s |awk '{print $1}'`

if ($verb) then
    echo Used disk is $used
endif


if ($used < $maxdisk) then
#
# nothing to do
#
if ($verb) then
    echo Exit with code 0
endif

exit 0
endif

# first test - see if you can clean applying retention time
if ($used > $maxdisk) then
if ($verb) then
    echo Running tmpwatch
endif
echo    tmpwatch --verbose --atime $retention_time . 
endif
#
# now look whether situation is good
#
set newused=`du -s |awk '{print $1}'`

if ($verb) then
    echo Now used is $newused
endif

if ($newused < $maxdisk) then
#
# I am happy, I bail out
# exit 2 = i had to delete, but just stuff I could delete
exit 2
endif

#
# else, delete files in order of age, one by one
#
while ($newused > $maxdisk)
 #
 # find the oldest file
 set file=`ls -t1|tail -1`


 echo rm -f $file
 #calculate new disk free
 set newused=`du -s |awk '{print $1}'`
#
end

#exit three means I had to delete stuff not expired
exit 3

#
