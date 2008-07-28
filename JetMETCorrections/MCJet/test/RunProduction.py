#! /usr/bin/env python
import os

command = "ResponseFitter HistoSettings.dat"
os.system(command)
command = "L3Correction HistoSettings.dat"
os.system(command)
command = "L2Correction HistoSettings.dat"
os.system(command)
