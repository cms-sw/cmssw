#!/bin/bash
root -b make_up.C
root -b gaussianResolution.C
root -b silvia.C
mkdir ~/www/dt/default/
cp * ~/www/dt/default/ -rf
