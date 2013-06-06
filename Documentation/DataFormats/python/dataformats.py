
import cStringIO,operator

def indent(rows, hasHeader=False, headerChar='-', delim=' | ', justify='left',
           separateRows=False, prefix='', postfix='', wrapfunc=lambda x:x):
    """Indents a table by column.
       - rows: A sequence of sequences of items, one sequence per row.
       - hasHeader: True if the first row consists of the columns' names.
       - headerChar: Character to be used for the row separator line
         (if hasHeader==True or separateRows==True).
       - delim: The column delimiter.
       - justify: Determines how are data justified in their column. 
         Valid values are 'left','right' and 'center'.
       - separateRows: True if rows are to be separated by a line
         of 'headerChar's.
       - prefix: A string prepended to each printed row.
       - postfix: A string appended to each printed row.
       - wrapfunc: A function f(text) for wrapping text; each element in
         the table is first wrapped by this function."""
    # closure for breaking logical rows to physical, using wrapfunc
    def rowWrapper(row):
        newRows = [wrapfunc(item).split('\n') for item in row]
        return [[substr or '' for substr in item] for item in map(None,*newRows)]
    # break each logical row into one or more physical ones
    logicalRows = [rowWrapper(row) for row in rows]
    # columns of physical rows
    columns = map(None,*reduce(operator.add,logicalRows))
    # get the maximum of each column by the string length of its items
    maxWidths = [max([len(str(item)) for item in column]) for column in columns]
    rowSeparator = headerChar * (len(prefix) + len(postfix) + sum(maxWidths) + \
                                 len(delim)*(len(maxWidths)-1))
    # select the appropriate justify method
    justify = {'center':str.center, 'right':str.rjust, 'left':str.ljust}[justify.lower()]
    output=cStringIO.StringIO()
    if separateRows: print >> output, rowSeparator
    for physicalRows in logicalRows:
        for row in physicalRows:
            print >> output, \
                prefix \
                + delim.join([justify(str(item),width) for (item,width) in zip(row,maxWidths)]) \
                + postfix
        if separateRows or hasHeader: print >> output, rowSeparator; hasHeader=False
    return output.getvalue()

# written by Mike Brown
# http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/148061
def wrap_onspace(text, width):
    """
    A word-wrap function that preserves existing line breaks
    and most spaces in the text. Expects that existing line
    breaks are posix newlines (\n).
    """
    return reduce(lambda line, word, width=width: '%s%s%s' %
                  (line,
                   ' \n'[(len(line[line.rfind('\n')+1:])
                         + len(word.split('\n',1)[0]
                              ) >= width)],
                   word),
                  text.split(' ')
                 )

import re
def wrap_onspace_strict(text, width):
    """Similar to wrap_onspace, but enforces the width constraint:
       words longer than width are split."""
    wordRegex = re.compile(r'\S{'+str(width)+r',}')
    return wrap_onspace(wordRegex.sub(lambda m: wrap_always(m.group(),width),text),width)

import math
def wrap_always(text, width):
    """A simple word-wrap function that wraps text on exactly width characters.
       It doesn't split the text in words."""
    return '\n'.join([ text[width*i:width*(i+1)] \
                       for i in xrange(int(math.ceil(1.*len(text)/width))) ])



# END OF TABLE FORMATING

# START of import
import sys, pprint
imported_modules = []

def importDF(path):

    modules_to_import = "RecoTracker RecoLocalTracker RecoLocalCalo RecoEcal RecoEgamma RecoLocalMuon RecoMuon RecoJets RecoMET RecoBTag RecoTauTag RecoVertex RecoPixelVertexing HLTrigger RecoParticleFlow".split()
    modules_to_import = "RecoLocalTracker RecoLocalMuon RecoLocalCalo RecoEcal TrackingTools RecoTracker RecoJets RecoMET RecoMuon RecoBTau RecoBTag RecoTauTag RecoVertex RecoPixelVertexing RecoEgamma RecoParticleFlow L1Trigger".split()
  

    for m in modules_to_import:
        m = m + "_dataformats"
        try:
            sys.path.append(path+"/src/Documentation/DataFormats/python/")
            globals()[m] = __import__(m)
            imported_modules.append(m)
	    print m
        except ImportError:
            print "skipping", m
        
# END of import            
        
def search(query):
    labels = ('Where(Package)', 'Instance', 'Container', 'Description')
    width = 20
    
    data = ""
    for module in imported_modules:
        for dict_name in ["full", "reco", "aod"]:         # going through all dictionaries
            #print module
            dict = vars(globals()[module])[dict_name] 
              
            for key in sorted(dict.keys()):                        # going though all elements in dictionary
                # TODO: IMPROVE
                element = dict[key]
                if query.lower() in element.__str__().lower():    # searching for query
                    if not (("No documentation".lower()) in element.__str__().lower()):
                        data+= module.replace("_dataformats", "")+" ("+ dict_name.replace("full", "FEVT") + ")||" + "||".join(element)+"\n"
                    
    if (data != ""):
        rows = [row.strip().split('||')  for row in data.splitlines()]
        print indent([labels]+rows, hasHeader=True, separateRows=True, prefix='| ', postfix=' |',  wrapfunc=lambda x: wrap_always(x,width))
    else:
        print "No documentation found"   

def help():
    print "usage: dataformats pattern_to_search"
    print "example: dataformats muon"
    print "Note! multiple patterns separated by space are not supported"

if __name__ == "__main__":

    if ("help" in sys.argv):
        help()
        sys.exit(0)	

    print sys.argv[1]
    print sys.argv[2]

    if (len(sys.argv) > 2):
	importDF(sys.argv[1])
        print "\nSearching for: "+sys.argv[2]+"\n" 
        search(sys.argv[2]) 
    else:
        help()
 
 
