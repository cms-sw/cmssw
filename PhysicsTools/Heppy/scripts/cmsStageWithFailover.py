#!/usr/bin/env python

#this script runs cmsStage multiple times in the case where it failes for some reason

if __name__ == '__main__':

    import CMGTools.Production.eostools as eostools
    eostools.setCAFPath()

    from cmsIO import *
    from cmsStage import *

    import sys, time

    #this taken from the main of cmsStage
    argv = sys.argv[1:]
    (args, debug, force ) = parseOpts( argv )

    if not os.path.isfile(args[0]):
        print args[0], 'does not exist.'
        sys.exit(1)
    source = cmsFile( args[0], "rfio" )
    destination = cmsFile( args[1], "stageout" )
    checkArgs( source, destination, force )

    #find the destination LFN
    dest = args[1]
    if eostools.isDirectory(dest):
        dest = os.path.join(dest,os.path.basename(args[0]))
    
    sleep_lengths = [1,10,60,600,1800]
    return_code = 0
    for i in xrange(5):

        #sleep for a while before running
        time.sleep(sleep_lengths[i])

        try:
            #run cmsStage
            print 'cmsStage %s [%d/5]' % (' '.join(argv) , i+1)
            main(argv)

        except SystemExit, e:
            print "cmsStage exited with code '%s'. Retrying... [%d/5]" % ( str(e), i+1 )
            return_code = e.code
        
        #sleep again before checking
        time.sleep(3)

        if eostools.fileExists(dest) and eostools.isFile(dest):
            if source.size() == destination.size():
                return_code = 0
                break

    sys.exit(return_code)
