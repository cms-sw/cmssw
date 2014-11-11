#/bin/bash

name=dqm_sourceclient-live

for i in fed ee eb hcal sistrip pixel hlt l1t l1temulator rpc dt csc hlx ispy; do
cfgfile=$i\_$name.cfg
pyfile=$i\_$name\_cfg.py
echo "==================="

if [ -e $pyfile ] ; then
echo removing $cfgfile
rm $cfgfile
echo recreating $cfgfile from $pyfile
cat >> $pyfile << EOF
print process.dumpConfig()
EOF
python $pyfile > $cfgfile

sed -i 's/srv-c2d05-18/srv-c2d05-19/' $cfgfile
sed -i 's/dqmdev/dqmpro/' $cfgfile
sed -i 's/cmsmon:50082\/urn:xdaq-application:lid=29/srv-c2d05-14.cms:22100\/urn:xdaq-application:lid=30/' $cfgfile 
sed -i 's/DQM\/Integration\/test\/inputsource.cfi/DQM\/Integration\/rcms\/inputsource.cfi/' $cfgfile
sed -i 's/DQM\/Integration\/test\/environment.cfi/DQM\/Integration\/rcms\/environment.cfi/' $cfgfile 

#fix for sequenceplaceholder
sed -i 's/cms.SequencePlaceholder("//g' $cfgfile
sed -i 's/")//g' $cfgfile


echo removing $pyfile
rm $pyfile

else
echo $pyfile does not exist
fi

done
