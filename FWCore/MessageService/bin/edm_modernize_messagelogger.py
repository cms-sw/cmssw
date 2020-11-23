import sys

args = sys.argv

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
        
    