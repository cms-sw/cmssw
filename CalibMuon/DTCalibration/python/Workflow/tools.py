import os,sys,imp
import subprocess
import logging
import fnmatch
import select

log = logging.getLogger(__name__)

def replaceTemplate(template,**opts):
    result = open(template).read()
    for item in opts:
         old = '@@%s@@'%item
         new = str(opts[item])
         print("Replacing",old,"to",new)
         result = result.replace(old,new)

    return result

def getDatasetStr(datasetpath):
    datasetstr = datasetpath
    datasetstr.strip()
    if datasetstr[0] == '/': datasetstr = datasetstr[1:]
    datasetstr = datasetstr.replace('/','_')

    return datasetstr

def listFilesLocal(paths, extension = '.root'):
    file_paths = []
    for path in paths:
        if not os.path.exists( path ):
            log.error( "Specified input path '%s' does not exist!" % path )
            continue
        if path.endswith( extension ):
            file_paths.append( path )
        for root, dirnames, filenames in os.walk( path ):
            for filename in fnmatch.filter( filenames, '*' + extension ):
                file_paths.append( os.path.join( root, filename ) )
    return file_paths

def haddLocal(files,result_file,extension = 'root'):
    log.info("hadd command: {}".format(" ".join(['hadd','-f', result_file] + files)))
    process = subprocess.Popen( ['hadd','-f', result_file] + files,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
    stdout = process.communicate()[0]
    log.info(f"hadd output: {stdout}")
    return process.returncode

def loadCmsProcessFile(psetName):
    pset = imp.load_source("psetmodule",psetName)
    return pset.process

def loadCmsProcess(psetPath):
    module = __import__(psetPath)
    process = sys.modules[psetPath].process

    import copy
    #FIXME: clone process
    #processNew = copy.deepcopy(process)
    processNew = copy.copy(process)
    return processNew

def prependPaths(process,seqname):
    for path in process.paths:
        getattr(process,path)._seq = getattr(process,seqname)*getattr(process,path)._seq


def stdinWait(prompt_text, default, time, timeoutDisplay = None, **kwargs):
    print(prompt_text)
    i, o, e = select.select( [sys.stdin], [], [], 10 )

    if (i):
        inp = sys.stdin.readline().strip()
    else:
        inp = default
    return inp

def interrupt(signum, frame):
    raise Exception("")

def getTerminalSize():
    #taken from http://stackoverflow.com/a/566752
    # returns width, size of terminal
    env = os.environ
    def ioctl_GWINSZ(fd):
        try:
            import fcntl, termios, struct, os
            cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,
        '1234'))
        except:
            return
        return cr
    cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
    if not cr:
        try:
            fd = os.open(os.ctermid(), os.O_RDONLY)
            cr = ioctl_GWINSZ(fd)
            os.close(fd)
        except:
            pass
    if not cr:
        cr = (env.get('LINES', 25), env.get('COLUMNS', 80))

        ### Use get(key[, default]) instead of a try/catch
        #try:
        #    cr = (env['LINES'], env['COLUMNS'])
        #except:
        #    cr = (25, 80)
    return int(cr[1]), int(cr[0])
