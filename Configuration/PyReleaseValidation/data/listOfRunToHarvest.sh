#! /bin/bash
echo "Creating list of runs to harvest from file " $1

file=$1
cat $file | egrep -i "fastsim|bash" > dqm.sh
cat $file | grep "/DQM" | egrep "relvalmc|cosmics" >> dqm.sh
cat $file | grep "2010B" | grep "/DQM" | egrep "146644|147115|147929|148822|149011|149181|149182|149291|149294|149442" >> dqm.sh
cat $file | grep "2010A" | grep "/DQM" | egrep "138937|138934|138924|138923|139790|139789|139788|139787|144086|144085|144084|144083|144011" >> dqm.sh
cat $file | grep "2011B" | grep "/DQM" | egrep "177719|177790|177096|175874" >> dqm.sh
cat $file | grep "2011A" | grep "/DQM" | egrep "165121|172802" >> dqm.sh

echo "Done"
