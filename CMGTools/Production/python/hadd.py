import os
import pprint
import pickle
import shutil

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
    

def hadd(file, odir, idirs):
    if file.endswith('.pck'):
        try:
            haddPck( file, odir, idirs)
        except ImportError:
            pass
        return
    elif not file.endswith('.root'):
        return
    haddCmd = ['hadd']
    haddCmd.append( file.replace( idirs[0], odir ) )
    for dir in idirs:
        haddCmd.append( file.replace( idirs[0], dir ) )
    # import pdb; pdb.set_trace()
    cmd = ' '.join(haddCmd)
    print cmd
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


def haddChunks(idir, removeDestDir, cleanUp=False, ignoreDirs=None):

    chunks = {}
    compsToSpare = set()
    if ignoreDirs == None: ignoreDirs = set()

    for file in sorted(os.listdir(idir)):
        filepath = '/'.join( [idir, file] )
        # print filepath
        if os.path.isdir(filepath):
            compdir = file
            skipDir = False
            if compdir in ignoreDirs:
              ignoreDirs.remove(compdir) 
              skipDir = True
            try:
                prefix,num = compdir.split('_Chunk')
            except ValueError:
                # ok, not a chunk
                continue
            #print prefix, num
            if skipDir: 
              compsToSpare.add(prefix)
              continue
            chunks.setdefault( prefix, list() ).append(filepath)
    if len(chunks)==0:
        print 'warning: no chunk found.'
        return
    for comp, cchunks in chunks.iteritems():
        odir = '/'.join( [idir, comp] )
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
            cleanIt = True
            if comp in compsToSpare :
              cleanIt = False
              compsToSpare.remove(comp)
            if cleanIt :
              for chunk in chunks:
                  shutil.move(chunk, chunkDir)




        
if __name__ == '__main__':
    import sys
    args = sys.argv
    # odir = args[1]
    # idirs = args[2:]
    # haddRec(odir, idirs)
    haddChunks(sys.argv[1])
