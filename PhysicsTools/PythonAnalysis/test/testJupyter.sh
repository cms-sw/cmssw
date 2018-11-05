#!/bin/bash

port=8080

#look for open port
while [ $port -lt 8090 ]; do
d=`netstat -tulpn 2>/dev/null | grep LISTEN | grep 127.0.0.1:$port |wc -l`
if [ $d -eq "0" ]
then
  break
fi
let port=port+1
done

echo "using port $port to test jupyter"

#open a notebook server, run for 20s and stop
timeout 20s jupyter notebook --no-browser --port=$port

#timeout has an exit code of 124 if a timeout has occured
if [ $? -eq 124 ]
then
   echo "Success (notebook stopped after 15s)"
   exit 0
else
   echo "There was an error starting the notebook"
   exit $?
fi


