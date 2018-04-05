import FWCore.ParameterSet.Config as cms
import os

dir = os.environ['CMSSW_RELEASE_BASE']+"/python"

def check(dir) :
    for root, dirs, files in os.walk(dir,followlinks=True):
        for f in files:
            if f[-1] != 'c' and f.find("_cfi.py") != -1:
#               print root[len(dir)+1:], f
                p = cms.Process("DUMMY1")
                try:
                    p.load(root[len(dir)+1:]+"/"+f[:-3])
                except ImportError as e:
                    print root[len(dir)+1:], f
                    print "ImportError:",e
                except RuntimeError as e:
                    print root[len(dir)+1:], f
                    print "RuntimeError:", e
                except ValueError as e:
                    print root[len(dir)+1:], f
                    print "ValueError:",e
                except NameError as e:
                    print root[len(dir)+1:], f
                    print "NameError:",e
                except TypeError as e:
                    print root[len(dir)+1:], f
                    print "NameError:",e

print "START"
check(dir)
print "STOP"
