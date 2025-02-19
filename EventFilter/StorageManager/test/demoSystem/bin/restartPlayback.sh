date
echo "Stopping everything..."
killall -9 xdaq.exe
sleep 5
FUShmCleanUp_t
sleep 10

date
echo "Starting everything..."
source startEverything.sh playback
