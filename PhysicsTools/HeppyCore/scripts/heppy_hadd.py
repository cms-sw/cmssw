#!/usr/bin/env python
# Copyright (C) 2014 Colin Bernet
# https://github.com/cbernet/heppy/blob/master/LICENSE

import os
import pprint
import pickle
import shutil

MAX_ARG_STRLEN = 131072

def haddPck(file, odir, idirs):
    '''add pck files in directories idirs to a directory outdir.
    All dirs in idirs must have the same subdirectory structure.
    Each pickle file will be opened, and the corresponding objects added to a destination pickle in odir.
    '''
    sum = None
    for dir in idirs:
        fileName = file.replace( idirs[0], dir )
        pckfile = open(fileName)
        obj = pickle.load(pckfile)
        if sum is None:
            sum = obj
        else:
            try:
                sum += obj
            except TypeError:
                # += not implemented, nevermind
                pass
                
    oFileName = file.replace( idirs[0], odir )
    pckfile = open(oFileName, 'w')
    pickle.dump(sum, pckfile)
    txtFileName = oFileName.replace('.pck','.txt')
    txtFile = open(txtFileName, 'w')
    txtFile.write( str(sum) )
    txtFile.write( '\n' )
    txtFile.close()
    

def hadd(file, odir, idirs, appx=''):
    if file.endswith('.pck'):
        try:
            haddPck( file, odir, idirs)
        except ImportError:
            pass
        return
    elif not file.endswith('.root'):
        return
    haddCmd = ['hadd']
    haddCmd.append( file.replace( idirs[0], odir ).replace('.root', appx+'.root') )
    for dir in idirs:
        haddCmd.append( file.replace( idirs[0], dir ) )
    # import pdb; pdb.set_trace()
    cmd = ' '.join(haddCmd)
    print cmd
    if len(cmd) > MAX_ARG_STRLEN:
        print 'Command longer than maximum unix string length; dividing into 2'
        hadd(file, odir, idirs[:len(idirs)/2], '1')
        hadd(file.replace(idirs[0], idirs[len(idirs)/2]), odir, idirs[len(idirs)/2:], '2')
        haddCmd = ['hadd']
        haddCmd.append( file.replace( idirs[0], odir ).replace('.root', appx+'.root') )
        haddCmd.append( file.replace( idirs[0], odir ).replace('.root', '1.root') )
        haddCmd.append( file.replace( idirs[0], odir ).replace('.root', '2.root') )
        cmd = ' '.join(haddCmd)
        print 'Running merge cmd:', cmd
        os.system(cmd)
    else:
        os.system(cmd)


def haddRec(odir, idirs):
    print 'adding', idirs
    print 'to', odir 

    cmd = ' '.join( ['mkdir', odir])
    # import pdb; pdb.set_trace()
    # os.system( cmd )
    try:
        os.mkdir( odir )
    except OSError:
        print 
        print 'ERROR: directory in the way. Maybe you ran hadd already in this directory? Remove it and try again'
        print 
        raise
    for root,dirs,files in os.walk( idirs[0] ):
        # print root, dirs, files
        for dir in dirs:
            dir = '/'.join([root, dir])
            dir = dir.replace(idirs[0], odir)
            cmd = 'mkdir ' + dir 
            # print cmd
            # os.system(cmd)
            os.mkdir(dir)
        for file in files:
            hadd('/'.join([root, file]), odir, idirs)

def haddChunks(idir, removeDestDir, cleanUp=False, odir_cmd='./'):
    chunks = {}
    for file in sorted(os.listdir(idir)):
        filepath = '/'.join( [idir, file] )
        # print filepath
        if os.path.isdir(filepath):
            compdir = file
            try:
                prefix,num = compdir.split('_Chunk')
            except ValueError:
                # ok, not a chunk
                continue
            # print prefix, num
            chunks.setdefault( prefix, list() ).append(filepath)
    if len(chunks)==0:
        print 'warning: no chunk found.'
        return
    for comp, cchunks in chunks.iteritems():
        odir = odir_cmd+'/'+'/'.join( [idir, comp] )
        print odir, cchunks
        if removeDestDir:
            if os.path.isdir( odir ):
                shutil.rmtree(odir)
        haddRec(odir, cchunks)
    if cleanUp:
        chunkDir = 'Chunks'
        if os.path.isdir('Chunks'):
            shutil.rmtree(chunkDir)
        os.mkdir(chunkDir)
        print chunks
        for comp, chunks in chunks.iteritems():
            for chunk in chunks:
                shutil.move(chunk, chunkDir)
        

if __name__ == '__main__':

    import os
    import sys
    from optparse import OptionParser

    parser = OptionParser()
    parser.usage = """
    %prog <dir>
    Find chunks in dir, and run recursive hadd to group all chunks.
    For example: 
    DYJets_Chunk0/, DYJets_Chunk1/ ... -> hadd -> DYJets/
    WJets_Chunk0/, WJets_Chunk1/ ... -> hadd -> WJets/
    """
    parser.add_option("-r","--remove", dest="remove",
                      default=False,action="store_true",
                      help="remove existing destination directories.")
    parser.add_option("-c","--clean", dest="clean",
                      default=False,action="store_true",
                      help="move chunks to Chunks/ after processing.")

    (options,args) = parser.parse_args()

    if len(args)>2:
        print 'provide at most 2 directory as arguments: first the source, then the destination (optional)'
        sys.exit(1)

    dir = args[0]
    if(len(args)>1):
      odir = args[1]
    else:
      odir='./'

    haddChunks(dir, options.remove, options.clean, odir)

