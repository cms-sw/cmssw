import sys

args = sys.argv

if len(args) == 1:
    print("file names must be passed as arguments")
    exit(-1)
if args[1] == '-h' or args[1] == '--help':
    print(
"""python edm_modernize_messagelogger.py [-h/--help] filename [...]
    
   Converts explicit constructions of cms.Service("MessageLogger") from old MessageLogger
   configration syntax to new the new syntax.
   The script expects a list of files to be modified in place.
    
   NOTE: The script is known to miss some corner-cases in the conversion so always check
   the results of the transformation.
"""
    )
    exit(0)

for arg in args[1:]:
    execfile(arg)
    
    ml = process.MessageLogger.clone()
    if hasattr(process.MessageLogger, "statistics"):
        stat = process.MessageLogger.statistics
        for s in stat:
            dest = getattr(ml, s.value())
            dest.enableStatistics = cms.untracked.bool(True)
        del ml.statistics
    dest = process.MessageLogger.destinations
    files = cms.untracked.PSet()
    if 'cerr' not in dest:
        ml.cerr = cms.untracked.PSet(enable = cms.untracked.bool(False))
    for d in dest:
        if 'cout' == d:
            continue
        if 'cerr' == d:
            continue
        setattr(files, d, getattr(ml,d.value()))
        delattr(ml, d)
            
    if hasattr(ml,'categories'):
        del ml.categories
    del ml.destinations
    
    f = open(arg)
    newF = open(arg+"new", "w")
    
    processingML = False
    parenthesis = 0
    for l in f.readlines():
        if not processingML:
            if 'process.MessageLogger' == l[0:21]:
                processingML = True
                parenthesis = l.count('(')
                parenthesis -= l.count(')')
                if 0 == parenthesis:
                    processingML = False
                continue
            newF.write(l)
        else:
            parenthesis += l.count('(')
            parenthesis -= l.count(')')
            if 0 == parenthesis:
                processingML = False
                newF.write('process.MessageLogger = '+ml.dumpPython())
        
    