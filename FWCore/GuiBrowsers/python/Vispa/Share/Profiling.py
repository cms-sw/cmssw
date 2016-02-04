import sys
import os
try:
    import cProfile
except Exception:
    import profile as cProfile
import pstats

from Vispa.Main.Directories import logDirectory

def analyze(function,filename,filter=None):
    profilename=os.path.join(logDirectory,os.path.splitext(os.path.basename(filename))[0]+"_profile")
    cProfile.run(function,profilename)
    p = pstats.Stats(profilename)
    p.strip_dirs()
    p.sort_stats("cumulative")
    f="^(?!unittest.py)"
    if filter!=None:
        f+=filter
    p.print_stats(f,30)
