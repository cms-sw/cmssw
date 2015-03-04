# do we need Digi2Raw and Raw2Digi?
from Configuration.StandardSequences.RawToDigi_cff import *
RawToDigi_noTk.remove(castorDigis)
# anything else to remove ?
RawToDigi = RawToDigi_noTk.copy()
