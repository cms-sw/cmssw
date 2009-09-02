#!/bin/sh

# determine the EP and SM URLs from either input arguments or defaults
source ./setDemoUrlEnvVar.sh $@
if [ "$BUILDER_UNIT_URL" == "" ] || [ "$FILTER_UNIT_URL" == "" ] || \
   [ "$FILTER_UNIT2_URL" == "" ] || [ "$FILTER_UNIT3_URL" == "" ] || \
   [ "$FILTER_UNIT5_URL" == "" ] || [ "$FILTER_UNIT6_URL" == "" ] || \
   [ "$FILTER_UNIT7_URL" == "" ] || [ "$FILTER_UNIT8_URL" == "" ] || \
   [ "$FILTER_UNIT4_URL" == "" ] || [ "$STORAGE_MANAGER_URL" == "" ] || \
   [ "$SM_PROXY_URL" == "" ] || [ "$CONSUMER_FU_URL" == "" ]
then
    echo "Missing env var URL!"
    exit
fi

# for each transition, we create a temporary copy of the xml file with
# the appropriate URL filled in, and send the modified xml to XDAQ

echo ""
echo "========================================"
echo "Halting the BU..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$BUILDER_UNIT_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,24," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl --stderr /dev/null -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $BUILDER_UNIT_URL
echo ""

sleep 5

echo ""
echo "========================================"
echo "Halting the RB..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$BUILDER_UNIT_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,25," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl --stderr /dev/null -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $BUILDER_UNIT_URL
echo ""

sleep 1

echo ""
echo "========================================"
echo "Halting the event processor..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,21," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT_URL
savedStatus=$?
echo ""

if [ $savedStatus -eq 0 ]
then
   sleep 1
fi

echo ""
echo "========================================"
echo "Halting the event processor (2)..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT2_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,121," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT2_URL
savedStatus=$?
echo ""

#if [ $savedStatus -eq 0 ]
#then
#   sleep 1
#fi

echo ""
echo "========================================"
echo "Halting the event processor (3)..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT3_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,221," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT3_URL
savedStatus=$?
echo ""

if [ $savedStatus -eq 0 ]
then
   sleep 1
fi

echo ""
echo "========================================"
echo "Halting the event processor (4)..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT4_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,321," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT4_URL
savedStatus=$?
echo ""

#if [ $savedStatus -eq 0 ]
#then
#   sleep 1
#fi

echo ""
echo "========================================"
echo "Halting the event processor (5)..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT5_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,421," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT5_URL
savedStatus=$?
echo ""

if [ $savedStatus -eq 0 ]
then
   sleep 1
fi

echo ""
echo "========================================"
echo "Halting the event processor (6)..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT6_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,521," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT6_URL
savedStatus=$?
echo ""

#if [ $savedStatus -eq 0 ]
#then
#   sleep 1
#fi

echo ""
echo "========================================"
echo "Halting the event processor (7)..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT7_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,621," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT7_URL
savedStatus=$?
echo ""

if [ $savedStatus -eq 0 ]
then
   sleep 1
fi

echo ""
echo "========================================"
echo "Halting the event processor (8)..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$FILTER_UNIT8_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,721," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $FILTER_UNIT8_URL
echo ""

sleep 5

echo "========================================"
echo "Halting the storage manager..."
echo "========================================"
rm -f tmp.xml
rm -f tmp2.xml
cat action.xml | sed "s,APPLICATION_URL,$STORAGE_MANAGER_URL," > tmp.xml
cat tmp.xml | sed "s,XDAQ_LID,29," > tmp2.xml
rm -f tmp.xml
cat tmp2.xml | sed "s,REQUESTED_ACTION,Halt," > tmp.xml
curl --stderr /dev/null -H "SOAPAction: urn:xdaq-application:lid=4" \
  -d @tmp.xml $STORAGE_MANAGER_URL
echo ""

sleep 5

rm -f tmp.xml
rm -f tmp2.xml
