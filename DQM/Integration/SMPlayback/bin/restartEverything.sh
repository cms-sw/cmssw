date
echo "Stopping everything..."
killall -9 xdaq.exe
sleep 5
FUShmCleanUp_t
sleep 5
echo "Starting everything..."
source startEverything.sh
