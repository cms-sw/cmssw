#!/usr/bin/env python
# Colin
# additional layer, on top of cmsBatch.py

import os

from castorBaseDir import getUserAndArea

def addToDatasets(sampleAndTier, user=os.getlogin()):
    user, _ = getUserAndArea(user) #for if we have a magic user name
    notThere = os.system( 'grep %s ~%s/public/DataSets.txt' % (sampleAndTier,user) )
    if notThere:
        os.system('echo %s >> ~%s/public/DataSets.txt' % (sampleAndTier,user) ) 
    return not notThere


if __name__ == '__main__':
    import sys
    addToDatasets(sys.argv[1], sys.argv[2])
