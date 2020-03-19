from DQMServices.Core.DQMStore_cfi import DQMStore

# This flag is turned on automatically when a job runs with more than one stream.
# However, it changes much more than just thread safety related things, so it is
# better to turn it on in *any* job that can be run multi-threaded, to get more
# reliable tests. Running single-threaded with this flag turned on is safe, but
# some modules (HARVESTING, legacy) will not work with this (and therefore can't
# run multi-threaded.
DQMStore.enableMultiThread = True
