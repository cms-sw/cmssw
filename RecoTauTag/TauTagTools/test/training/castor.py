# @file castor.py
# @brief A simple helper to handle simple tasks with CASTOR
#
#   - nsls     : lists CASTOR name server directory. Handles wildcards only
#                in the filename (no wildcard in path to file allowed)
#   - stager_get: takes a path-to-file-pattern and stages the matching files
#                per bunch of N (default=10) files
#   - rfcat    : todo
#   - rfcp     : done in a very naive way
#   - rfiod    : todo
#   - rfrename : todo
#   - rfstat   : done
#   - rfchmod  : todo
#   - rfdir    : done
#   - rfmkdir  : todo
#   - rfrm     : todo
#   - rftp     : todo
#
# date:   May 2006
# @author: Sebastien Binet <binet@cern.ch>

import datetime
import commands
import os
import fnmatch
import time

# Memoize calls to nslsl
_CACHE = {}

def group(iterator, count):
    """
    This function extracts items from a sequence or iterator 'count' at a time:
    >>> list(group([0, 1, 2, 3, 4, 5, 6], 2))
    [(0, 1), (2, 3), (4, 5)]
    Stolen from :
    http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/439095
    """
    itr = iter(iterator)
    while True:
        yield tuple([itr.next() for i in xrange(count)])

__author__  = "Sebastien Binet <binet@cern.ch>"
__version__ = "$Revision: 1.1 $"
__doc__ = """A set of simple helper methods to handle simple tasks with CASTOR.
"""

def hasWildcard(name) :
    """Return true if the name has a UNIX wildcard (*,?,[,])
    """
    if ( name.count('*') > 0 or name.count('?') > 0 or
         name.count('[') > 0 or name.count(']') > 0 ) :
        return True
    else :
        return False

def nslsl(path):
    '''
    List CASTOR directory/file contents, returning the information found
    in 'nsls -l'.  The output is a list of dictionaries with the following
    entries:
        permissions
        file
        modified
        size (in bytes)
    '''
    # Get path
    directory = os.path.dirname(path)
    basename = os.path.basename(path)

    # Memoize the results
    if path in _CACHE:
        yield _CACHE[path]
    else:
        status,output = commands.getstatusoutput('nsls -l ' + directory)

        if status == 256:
            raise IOError("Can't nsls -l on path: %s, it doesn't exist!" % path)
        for line in output.splitlines():
            fields = line.split()
            output = {
                'permissions' : fields[0],
                'size' : int(fields[4]),
                'file' : fields[8]
            }
            time_stamp = " ".join(fields[5:8])
            # If it doesn't have a year stamp, assume it was made this year.
            if time_stamp.find(':') != -1:
                output['time'] = time.strptime(
                    time_stamp + " " + str(datetime.datetime.now().year),
                    "%b %d %H:%M %Y")
            else:
                output['time'] = time.strptime(time_stamp, "%b %d %Y")

            # If the date is greater than the current date, it was made *last*
            # year.  nsls sucks.
            if output['time'] > datetime.datetime.now().timetuple():
                # Add a year
                current = datetime.datetime.fromtimestamp(
                    time.mktime(output['time']))
                current -= datetime.timedelta(days=365)
                output['time'] = current.timetuple()

            output['path'] = os.path.join(directory, output['file'])
            _CACHE[output['path']] = output
            #print path, output['file']
            # Check the filename matches our pattern
            if not basename or fnmatch.fnmatch(output['file'], basename):
                #print "match!"
                yield output

def nsls(path):
    for file_info in nslsl(path):
        yield file_info['file']

#def nsls(path):
    #"""
    #lists CASTOR name server directory/file entries.
    #If path is a directory, nsls lists the entries in the directory;
    #they are sorted alphabetically.

    #path specifies the CASTOR pathname. If path does not start  with  /,
    #it  is  prefixed  by  the content of the CASTOR_HOME environment
    #variable.

    #ex:
    #>>> nsls( '/castor/cern.ch/atlas/*' )
    #>>> nsls( 'mydata' )
    #"""

    #wildcards = False
    #tail = "*"
    #path = os.path.expandvars(path)

    #if path.endswith('/') :
        #path = path[0:len(path)-1]
    ## Do we detect a wildcard in the path we are given ?
    ## if so then we have to parse it to remove them because
    ## nsls does not understand them.
    ## The ouput of the command will be filtered afterwards
    #if hasWildcard(path) :
        #wildcards = True

        #wholepath = path.split(os.sep)

        ## Here we assume the wildcards are located *only* in the filename !!
        #tail      = wholepath[len(wholepath)-1]
        #if tail == '' :
            #if len(wholepath) >= 2 :
                #tail = wholepath[len(wholepath)-2]
            #else :
                #raise Exception, \
                      #"Malformed path to files: <"+path+">"

        ## Check that the wildcard is not in the path to files
        #if tail.count('/') > 0 :
            #if tail.endswith('/') :
                ## the / is sitting in last position. Can safely remove it
                #tail = tail[0:len(tail)-1]
            #else :
                #raise Exception, \
                      #"No wildcard allowed in the path to files: <"+path+">"


        #path      = path.split(tail)[0]
        #if hasWildcard(path) :
            #raise ValueError("No wildcard allowed in the path to files: <"+path+">")
        ##print path

    #status,output = commands.getstatusoutput('nsls '+path)

    #if status != 0 :
        #print output
        #return []

    #output = output.splitlines()

    #if wildcards :
        #output = fnmatch.filter(output,tail)

    #for i in xrange(0,len(output)) :
        #if output[i].count(path) < 1:
            #output[i] = path+"/"+output[i]
        #output[i] = output[i].replace('//','/')
    #return output

def pool_nsls( path ) :
    """
    lists CASTOR name server directory/file entries.
    Prepend the 'rfio:' prefix so the output list can be used as an input
    for an xmlfile_catalog file.
    """
    files = nsls(path)
    for i in xrange(len(files)) :
        files[i] = "rfio:"+files[i]
        pass

    return files

def getFileSize( pathToFile = None ) :
    """
    Use nsls -l function to read the file and decypher the output string to
    extract the size of the file.
    Returns the size in Mb
    """
    if hasWildcard(pathToFile) :
        raise Exception, \
              "No wildcard allowed in the path to files: <"+pathToFile+">"

    status,output = commands.getstatusoutput( 'nsls -l '+pathToFile )
    #'nsls -l $CASTOR_DIR/$FILE | awk -F ' ' '{print $5}'

    if status != 0 :
        print "** PyCastor ERROR **"
        print output
        return []

    output = output.splitlines()

    #print "output:",output
    #print "output size= ",len(output)

    if len(output) != 1 :
        raise Exception, \
              "Wrong status (didn't find only *1* file!!)"

    output = output[0]
    output = output.split( " " )

    result = []
    # Removes whitespaces
    for i in output :
        if i != '' :
            result.append( i )
            pass
        pass

    size = int(result[4])/(1024.*1024.) # size in Mb
    #print "size = ",size," Mb"

    return size

def stagein( fileListPattern = None, nSlices = 10, verbose = True ) :
    """
    Take a path to a file pattern and stages all the files corresponding
    to this pattern by bunchs of N (default=10) files.
    """
    files = nsls( fileListPattern )
    if ( type(files) != type([]) or len(files) < 1 ) :
        raise Exception, \
              "Error, no file to stagein !!"
        return

    slices = list(group(files,nSlices))

    for slice in slices :
        stageList = ' -M '.join( [''] + [ s for s in slice ] )
        cmd = 'stager_get %s' % stageList
        if verbose :
            print ">>> cmd= ",cmd
        status,output = commands.getstatusoutput(cmd)

        if status != 0 :
            print "** PyCastor ERROR **"
            print output
            pass
        else :
            if verbose :
                # print output
                pass
            pass
        pass

    return 0

def stager_qry(inFiles):
    """
    Find out the stage status of the inFiles
    returns dictionary of outStatus(files:status) status = 0|1
    """
    outStatus = dict()
    for inFile in inFiles:
        cmd = "stager_qry -M %s " % ( inFile, )
        sc,out = commands.getstatusoutput( cmd )
        if sc != 0:
            print "** PyCastor ERROR **"
            print "## Could not check status of this file [%s] !!" % inFile
            print "## status sc=", sc
            print "## output out=", out

        #for str in out.split():
        #   print "out_str=", str

        if out.split()[-1] == "STAGED":
            outStatus[inFile] = 1
        else:
            outStatus[inFile] = 0
        #print "-"*77

    return outStatus

def extract_rfio(inFile, outDir):
    """
    Extract the list of rfio:/castor/.. files from given input file_name
    - Finds out STAGED status of files using stager_qry -M ...
    - if STAGED: rfcp them into outDir
    - if NOT: stage them in using stager_get -M ...
    - returns status dictionary returned by stager_qry() above
    """
    allGood = True
    f = open(inFile, 'r')
    file_text = f.read()
    f.close()

    import re
    import urlparse
    def grep_path(schema, text):
        expr = "\"" + schema + "\:.+?\""
        lines = re.findall(expr, text)
        lines = [line.strip('"') for line in lines ]
        import urlparse
        paths = [urlparse.urlparse(line)[2] for line in lines]
        #results = [str[len(schema)+1:] for str in results if str.startswith(schema+':')]
        return paths

    path_list = grep_path("rfio", file_text)
    print "rfio_file list extracted from input file =", inFile
    print "-"*77; print path_list; print "-"*77

    def _print(str):
        print str

    status_dict = stager_qry(path_list)
    ready_files_list = [file for file in status_dict if status_dict[file] == 1]
    print "---STAGED (ready to be copied):";
    p = map(_print, ready_files_list); print "-"*77

    noready_files_list = [file for file in status_dict if status_dict[file] == 0]
    print "---NOT STAGED (not ready to be copied):";
    p = map(_print, noready_files_list); print "-"*77

    def _rfcp(file): #aux func. just for reporting purpose
        print "rfcp ", file
        return file
    rfcp( map(_rfcp, ready_files_list), #[file for file in ready_files_list],
          outDir  )

    def _stager_get(file): #aux func. just for reporting purpose
        print "stager_get -M ", file
        stager_get(file)
    map(_stager_get, noready_files_list) #[stager_get(file) for file in noready_files_list if 1 print "stager_get -M ", file]

    return status_dict #returned from stager_qry(),
    #not completely true since the outcome of rfcp is not checked here

def stager_get(inFile):
    """
    STAGE IN the inFile on castor
    """
    allGood = True
    cmd = "stager_get -M %s" % (inFile)
    sc,out = commands.getstatusoutput( cmd )
    if sc != 0:
        print "** PyCastor ERROR **"
        print "## Could not stager_get this file [%s] !!" % inFile
        allGood = False
        pass
    if allGood:
        return 0
    return 1

def rfcp( inFiles, outDir ):
    """
    Copy the inFiles into the outDir
    """
    allGood = True
    for inFile in inFiles:
        cmd = "rfcp %s %s" % ( inFile,
                               os.path.join( outDir,
                                             os.path.basename(inFile) ) )
        sc,out = commands.getstatusoutput( cmd )
        if sc != 0:
            print "** PyCastor ERROR **"
            print "## Could not copy this file [%s] !!" % inFile
            allGood = False
            pass
        pass
    if allGood:
        return 0
    return 1


def rfstat (pathname):
    """rfstat <file_path>
    Perform a stat system call on the given path
       @param `pathname` to a file or directory on a castor node
       @return a dictionary of entries built from rfstat's summary
    """
    cmd = 'rfstat %s' % pathname
    sc, out = commands.getstatusoutput (cmd)
    if sc != 0:
        print "** PyCastor ERROR **"
        print ":: command: [%s]" % cmd
        print ":: status:  [%s]" % sc
        print out
        raise RuntimeError (sc)

    stat = dict()
    for l in out.splitlines():
        l = l.strip()
        o = l.split(':')
        hdr = o[0].strip()
        tail= ''.join(l.split(o[0]+':')[1:]).strip()
        stat[hdr] = tail
    return stat

def rfdir (paths, recursive=False):
    """ rfdir file|directory
    """
    if isinstance(paths, str):
        paths = [paths]

    cmd = "rfdir %s %s" % ('-R' if recursive else '',
                           ' '.join(paths))
    sc, out = commands.getstatusoutput (cmd)
    return sc, out

def last_modified(path):
    ''' last_modified <file_path>
    Return the last modification time for a castor file
    '''

    #Castor time string
    castor_time = time.strptime(rfstat(path)['Last modify'],
                                "%a %b %d %H:%M:%S %Y")
    return time.mktime(castor_time)

