TIMESTAMP=`date '+%Y%m%d%H%M%S'`
LOGLEV="WARN"

mkdir -p  ../log/builderUnit
mkdir -p  ../log/consumer
mkdir -p  ../log/filterUnit
mkdir -p  ../log/storageManager
mkdir -p  ../log/watchdog


cd ../log/builderUnit
xdaq.exe -h $HOSTNAME -l ${LOGLEV} -e /nfshome0/dqmdev/xml/profile.xml -p 50080 -c ../../cfg/sm_playback_atcp.xml >& BU${TIMESTAMP}.log &

#sleep 5

cd ../filterUnit
xdaq.exe -h $HOSTNAME -l ${LOGLEV} -e /nfshome0/dqmdev/xml/profile.xml -p 50081 -c ../../cfg/sm_playback_atcp.xml >& FU${TIMESTAMP}.log &

#sleep 5

cd ../storageManager
xdaq.exe -h $HOSTNAME -l ${LOGLEV} -e /nfshome0/dqmdev/xml/profile.xml -p 50082 -c ../../cfg/sm_playback_atcp.xml >& SM${TIMESTAMP}_1.log &


sleep 10

cd ../../bin
source startSystem.sh $HOSTNAME
