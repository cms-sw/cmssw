#! /bin/sh


if [ $HOSTNAME == "srv-C2D05-09" ]; then
CLIENTS="ee eb l1t l1temulator"
fi
if [ $HOSTNAME == "srv-C2D05-10" ]; then
CLIENTS="hcal csc dt rpc fed"
fi
if [ $HOSTNAME == "srv-C2D05-11" ]; then
CLIENTS="pixel hlt hlx sistrip"
fi
if [ $HOSTNAME == "srv-c2d05-11" ]; then
CLIENTS="pixel hlt hlx sistrip"
fi

MODE="playback"
#MODE="live"

for c in $CLIENTS;
do
  echo "starting $c ... "
  ./loop_generic.sh $c $MODE &
done
