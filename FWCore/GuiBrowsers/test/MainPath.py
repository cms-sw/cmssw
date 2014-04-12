import sys
import os

try:
    distBaseDirectory=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    if not os.path.exists(distBaseDirectory) or not "Vispa" in os.listdir(distBaseDirectory):
        distBaseDirectory=os.path.abspath(os.path.join(os.path.dirname(__file__),"../python"))
    if not os.path.exists(distBaseDirectory) or not "Vispa" in os.listdir(distBaseDirectory):
        distBaseDirectory=os.path.abspath(os.path.expandvars("$CMSSW_BASE/python/FWCore/GuiBrowsers"))
    if not os.path.exists(distBaseDirectory) or not "Vispa" in os.listdir(distBaseDirectory):
        distBaseDirectory=os.path.abspath(os.path.expandvars("$CMSSW_RELEASE_BASE/python/FWCore/GuiBrowsers"))
except Exception:
    distBaseDirectory=os.path.abspath(os.path.join(os.path.dirname(sys.argv[0]),".."))

sys.path.append(distBaseDirectory)
