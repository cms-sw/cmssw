import sys
import os
try:
    import cProfile
except Exception:
    import profile as cProfile
import pstats

def analyze(function,filename,filter=None):
    profilename=os.path.splitext(filename)[0]+"_profile"
    cProfile.run(function,profilename)
    p = pstats.Stats(profilename)
    p.strip_dirs()
    p.sort_stats("cumulative")
    f="^(?!unittest.py)"
    if filter!=None:
        f+=filter
    p.print_stats(f,30)
