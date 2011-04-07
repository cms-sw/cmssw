#!/bin/sh

proxyHost=dvsrv-c2f37-01.cms
proxyPort=31110

ps wwxau|sed -nre 's/^[[:alpha:] ]+([[:digit:]]+).*xdaq.exe.*smps_daqval.xml$/\1/p' | xargs kill -9 

/opt/xdaq/bin/xdaq.exe -h $proxyHost -p $proxyPort -c ../cfg/smps_daqval.xml &
sleep 3
../../demoSystem/soap/sendSimpleCmdToApp $proxyHost $proxyPort SMProxyServer 0 Configure
sleep 3
../../demoSystem/soap/sendSimpleCmdToApp $proxyHost $proxyPort SMProxyServer 0 Enable
