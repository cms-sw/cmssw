'''This module collects some frequently used helper functions
'''
import time,ast,re,json,coral,array
def flatten(obj):
    '''Given nested lists or tuples, returns a single flattened list'''
    result = []
    for piece in obj:
        if hasattr (piece, '__iter__') and not isinstance (piece, str):
            result.extend( flatten (piece) )
        else:
            result.append (piece)
    return result

def parseTime(iTime):
    '''
    input string of the ("^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$|^\d{6}$|^\d{4}$" format
    output (runnum,fillnum,timeStr)
    '''
    if not iTime: return (None,None,None)
    p=re.compile('^\d\d/\d\d/\d\d \d\d:\d\d:\d\d$')
    if re.match(p,iTime):
        return (None,None,iTime)
    p=re.compile('^\d{6}$')
    if re.match(p,iTime):
        return (int(iTime),None,None)
    p=re.compile('^\d{4}$')
    if re.match(p,iTime):
        return (None,int(iTime),None)
    
def lumiUnitForPrint(t):
    '''
    input : largest lumivalue
    output: (unitstring,denomitor)
    '''
    unitstring='/ub'
    denomitor=1.0
    if t>=1.0e3 and t<1.0e06:
        denomitor=1.0e3
        unitstring='/nb'
    elif t>=1.0e6 and t<1.0e9:
        denomitor=1.0e6
        unitstring='/pb'
    elif t>=1.0e9 and t<1.0e12:
        denomitor=1.0e9
        unitstring='/fb'
    elif t>=1.0e12 and t<1.0e15:
        denomitor=1.0e12
        unitstring='/ab'
    elif t<1.0 and t>=1.0e-3: #left direction
        denomitor=1.0e-03
        unitstring='/mb'
    elif t<1.0e-03 and t>=1.0e-06:
        denomitor=1.0e-06
        unitstring='/b'
    elif t<1.0e-06 and t>=1.0e-09:
        denomitor=1.0e-9
        unitstring='/kb'
    return (unitstring,denomitor)
def guessUnit(inverseubval):
    '''
    input:
        float value in 1/ub
    output:
        printable value (value(float),unit(str)) unit in [1/kb,1/b,1/mb,1/ub,1/nb,1/pb,1/fb]
    '''
    if inverseubval>=1.0e-09 and inverseubval<1.0e-06:
        denomitor=1.0e-09
        unitstring='/kb'
        return (float(inverseubval)/float(denomitor),unitstring)
    if inverseubval>=1.0e-06 and inverseubval<1.0e-03:
        denomitor=1.0e-06
        unitstring='/b'
        return (float(inverseubval)/float(denomitor),unitstring)
    if inverseubval>=1.0e-03 and inverseubval<1.0:
        denomitor=1.0e-03
        unitstring='/mb'
        return (float(inverseubval)/float(denomitor),unitstring)
    if inverseubval>=1.0 and inverseubval<1.0e3:
        unitstring='/ub'
        return (inverseubval,unitstring)
    if inverseubval>=1.0e3 and inverseubval<1.0e06:
        denomitor=1.0e3
        unitstring='/nb'
        return (float(inverseubval)/float(denomitor),unitstring)
    if inverseubval>=1.0e6 and inverseubval<1.0e9:
        denomitor=1.0e6
        unitstring='/pb'
        return (float(inverseubval)/float(denomitor),unitstring)
    if inverseubval>=1.0e9 and inverseubval<1.0e12:
        denomitor=1.0e9
        unitstring='/fb'
        return (float(inverseubval)/float(denomitor),unitstring)
    if inverseubval>=1.0e12 and inverseubval<1.0e15:
        denomitor=1.0e12
        unitstring='/ab'
        return (float(inverseubval)/float(denomitor),unitstring)
    return (float(inverseubval),'/ub')
def pairwise(lst):
    """
    yield item i and item i+1 in lst. e.g.
    (lst[0], lst[1]), (lst[1], lst[2]), ..., (lst[-1], None)
    
    from http://code.activestate.com/recipes/409825-look-ahead-one-item-during-iteration
    """
    if not len(lst): return
    #yield None, lst[0]
    for i in range(len(lst)-1):
        yield lst[i], lst[i+1]
    yield lst[-1], None
def findInList(mylist,element):
    """
    check if an element is in the list
    """
    pos=-1
    try:
        pos=mylist.index(element)
    except ValueError:
        pos=-1
    return pos!=-1
def is_intstr(s):
    """test if a string can be converted to a int
    """
    try:
        int(s)
        return True
    except ValueError:
        return False
def is_floatstr(s):
    """
    test if a string can be converted to a float
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
def count_dups(l):
    """
    report the number of duplicates in a python list
    """
    from collections import defaultdict
    tally=defaultdict(int)
    for x in l:
        tally[x]+=1
    return tally.items()
def transposed(lists, defaultval=None):
    """
    transposing list of lists
    from http://code.activestate.com/recipes/410687-transposing-a-list-of-lists-with-different-lengths/
    """
    if not lists: return []
    #return map(lambda *row: [elem or defaultval for elem in row], *lists)
    return map(lambda *row: [elem for elem in row or defaultval], *lists)
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)
def packToString(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long in string format
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    fmt="%u"
    return fmt%pack(high,low)
def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)
def unpackFromString(i):
    """unpack 64bit unsigned long long in string format into 2 32bit unsigned int, return tuple(high,low)
    """
    return unpack(int(i))
def timeStamptoDate(i):
    """convert 64bit timestamp to local date in string format
    """
    return time.ctime(unpack(i)[0])
def timeStamptoUTC(i):
    """convert 64bit timestamp to Universal Time in string format
    """
    t=unpack(i)[0]
    return time.strftime("%a, %d %b %Y %H:%M:%S +0000",time.gmtime(t))
def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}
def inclusiveRange(start,stop,step):
    """return range including the stop value
    """
    v=start
    while v<stop:
        yield v
        v+=step
    if v>=stop:
        yield stop
        
def tolegalJSON(inputstring):
   '''
   convert json like string to legal json string
   add double quote around json keys if they are not there, change single quote to double quote around keys
   '''
   strresult=inputstring.strip()
   strresult=re.sub("\s+","",strresult)
   try:
       mydict=ast.literal_eval(strresult)
   except SyntaxError:
       print 'error in converting string to dict'
       raise
   result={}
   for k,v in mydict.items():
       if not isinstance(k,str):
           result[str(k)]=v
       else:
           result[k]=v
   return re.sub("'",'"',str(result))

def packArraytoBlob(iarray):
    '''
    Inputs:
    inputarray: a python array
    '''
    result=coral.Blob()
    result.write(iarray.tostring())
    return result

def unpackBlobtoArray(iblob,itemtypecode):
    '''
    Inputs:
    iblob: coral.Blob
    itemtypecode: python array type code 
    '''
    if itemtypecode not in ['c','b','B','u','h','H','i','I','l','L','f','d']:
        raise RuntimeError('unsupported typecode '+itemtypecode)
    result=array.array(itemtypecode)
    blobstr=iblob.readline()
    if not blobstr :
        return None
    result.fromstring(blobstr)
    return result

def packListstrtoCLOB(iListstr,separator=','):
    '''
    pack list of string of comma separated large string CLOB
    '''
    return separator.join(iListstr)

def unpackCLOBtoListstr(iStr,separator=','):
    '''
    unpack a large string to list of string
    '''
    return [i.strip() for i in iStr.strip().split(separator)]

def splitlistToRangeString (inPut):
    result = []
    first = inPut[0]
    last = inPut[0]
    result.append ([inPut[0]])
    counter = 0
    for i in inPut[1:]:
        if i == last+1:
            result[counter].append (i)
        else:
            counter += 1
            result.append ([i])
        last = i
    return ', '.join (['['+str (min (x))+'-'+str (max (x))+']' for x in result])

def parselumicorrector(correctorStr):
    '''
    output: (functionname,parametersinuppercase[])
    '''
    cleancorrectorStr=correctorStr.replace(' ','')#in case of whitespace by mistake
    [correctorFunc,paramsStr]=cleancorrectorStr.split(':')
    params=paramsStr.split(',')
    return (correctorFunc,params)

if __name__=='__main__':
    nested=[[[1,2],[6,6,8]],[[3,4,5],[4,5]]]
    print 'flattened ',flatten(nested)
    a=[1,2,3,4,5]
    for i,j in pairwise(a):
        if j :
            print i,j
    lst = ['I1','I2','I1','I3','I4','I4','I7','I7','I7','I7','I7']
    print count_dups(lst)
    seqbag=[[1,2,3],[1,3,3],[1,4,6],[4,5,6,7],[8,9]]
    print 'before ',seqbag
    print 'after ',transposed(seqbag,None)
    print [i for i in inclusiveRange(1,3,1)]
    
    result=tolegalJSON('{1:[],2:[[1,3],[4,5]]}')
    print result
    pp=json.loads(result)
    print pp["2"]
    result=tolegalJSON("{'1':[],'2':[[1,3],[4,5]]}")
    print result
    pp=json.loads(result)
    print pp["2"]
    result=tolegalJSON('{"1":[],"2":[[1,3],[4,5]]}')
    print result
    pp=json.loads(result)
    print pp["2"]
    
    a=array.array('f')
    a.append(1.3)
    a.append(1.4)
    a.append(2.3)
    a.append(6.3)
    myblob=packArraytoBlob(a)
    print myblob.size()
    print unpackBlobtoArray(myblob,'f')
    b=array.array('f')
    myblob=packArraytoBlob(b)
    print myblob.size()
    a=['aa_f', 'bb', 'dfc']
    print packListstrtoCLOB(a)
