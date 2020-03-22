#!/bin/tcsh

### Make list of files

set runorigped=225529
set runorigled=135077
set runoriglas=224708

set runold = -1
touch list_of_myruns_led
touch list_of_myruns_las
touch list_of_myruns_ped

set j=0
cat $1 | grep CMS.HCAL | awk '{print $1"!"$2"!"$3"!"$4"!"$5"!"$6"!"$7}' > tmp.list

foreach i (`cat tmp.list`)
set run=`echo ${i} | awk -F! '{print $1}'`
set year=`echo ${i} | awk -F! '{print $2}' | awk -F - '{print $3}'`
#set year=`echo "20"${year1}`
set month=`echo ${i} | awk -F! '{print $2}' | awk -F - '{print $2}'`
set mymonth = ""
set lll=1
foreach kkk (JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC)
if( ${kkk} == ${month} ) then
set mymonth=`echo "0"${lll}`
endif
@ lll = ${lll} + "1"
end
set mypm=`echo ${i} | awk -F! '{print $4}'`
set day=`echo ${i} | awk -F! '{print $2}' | awk -F - '{print $1}'`
set hour=`echo ${i} | awk -F! '{print $3}' | awk -F . '{print $1}'`
if(${mypm} == "PM") then
@ hour = ${hour} + "12"
endif
set minute=`echo ${i} | awk -F! '{print $3}' | awk -F . '{print $2}'`
set second=`echo ${i} | awk -F! '{print $3}' | awk -F . '{print $3}'`
#set mytime=`echo ${year}"-"${mymonth}"-"${day}" "${hour}":"${minute}":"${second} ${mypm}`
set time=`echo ${day}"-"${mymonth}"-"${year}"_"${hour}":"${minute}":"${second}`
#echo ${mytime}
#set time=`date -d "${mytime}" '+%d-%m-%y_%H:%M:%S'`
#set time=${mytime}
echo ${time}

if( ${run} != ${runold} ) then
set j=0
set led=100
echo ${i} | grep LED
if( ${status} == "0" ) then
set led=1
endif

echo ${i} | grep Laser
if( ${status} == "0" ) then
set led=2
endif

echo ${i} | grep pedestal
if( ${status} == "0" ) then
set led=3
endif


set runold=${run}
@ j = ${j} + "1"
else
if( ${j} == "1" ) then
echo ${i}
set nevents=`echo ${i} | awk -F! '{print $6}'`
if( ${led} == "1" ) then
echo ${run}"_"${runorigled}"_"${time}"_"${nevents} >>list_of_myruns_led
endif
if( ${led} == "2" ) then
echo ${run}"_"${runoriglas}"_"${time}"_"${nevents} >>list_of_myruns_las
endif
if( ${led} == "3" ) then
echo ${run}"_"${runorigped}"_"${time}"_"${nevents} >>list_of_myruns_ped
endif
@ j = ${j} + "1"
endif
endif
end



