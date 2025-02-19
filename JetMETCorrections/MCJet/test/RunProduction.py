#! /usr/bin/env python
import os

command = "ReadTree TreeSettings.conf"
os.system(command)
command = "ResponseFitter HistoSettings.conf"
os.system(command)
command = "L3Correction HistoSettings.conf"
os.system(command)
command = "L2Correction HistoSettings.conf"
os.system(command)
