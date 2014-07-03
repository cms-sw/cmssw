#!/bin/bash

echo "========================================"
echo "Configuring the ATCPs..."
echo "========================================"
./sendSimpleCmdToApp $1 50080 pt::atcp::PeerTransportATCP 0 Configure
./sendSimpleCmdToApp $1 50082 pt::atcp::PeerTransportATCP 1 Configure

sleep 2

echo "========================================"
echo "Enabling the ATCPs..."
echo "========================================"
./sendSimpleCmdToApp $1 50080 pt::atcp::PeerTransportATCP 0 Enable
./sendSimpleCmdToApp $1 50082 pt::atcp::PeerTransportATCP 1 Enable

sleep 2

echo "========================================"
echo "Configuring the storage manager..."
echo "========================================"
./sendSimpleCmdToApp $1 50082 StorageManager 0 Configure

sleep 5

echo ""
echo "========================================"
echo "Configuring the RB..."
echo "========================================"
./sendSimpleCmdToApp $1 50080 evf::FUResourceBroker 0 Configure

sleep 5

echo ""
echo "========================================"
echo "Configuring the event processor..."
echo "========================================"
./sendSimpleCmdToApp $1 50081 evf::FUEventProcessor 0 Configure

sleep 13

echo ""
echo "========================================"
echo "Enabling the storage manager..."
echo "========================================"
./sendSimpleCmdToApp $1 50082 StorageManager 0 Enable
echo ""

sleep 10

echo ""
echo "========================================"
echo "Enabling the RB..."
echo "========================================"
./sendSimpleCmdToApp $1 50080 evf::FUResourceBroker 0 Enable
echo ""

sleep 6

echo ""
echo "========================================"
echo "Enabling the event processor..."
echo "========================================"
./sendSimpleCmdToApp $1 50081 evf::FUEventProcessor 0 Enable
echo ""
