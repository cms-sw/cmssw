Here there is a small suite of scripts and cfgs to test the tracker with bad modules,
apvs and strips.
- doIdealDB.sh generates a "dbfile.db" with an 'almost-ideal' detector.
- doBad----.sh generates a dbfile with bad ----; you can tune the amount of badness
  by looking at the mkBad---.pl perl script that creates a random list of bad components
NOTE: doIdealDB.sh deletes the databases created by doBad---.sh; sorry fot that.

