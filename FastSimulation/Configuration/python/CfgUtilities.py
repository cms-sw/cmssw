def getObjName(myobj,_locals):
    for name,obj in _locals.iteritems():
        if obj == myobj:
            return name
    return "unknown"
            
def getSeqEntryNames(seq,_locals):
    myobjs = seq._seq._collection
    mynames = [getObjName(seq,_locals)]
    for myobj in myobjs:
        try:
            mynames.extend(getSeqEntryNames(myobj,_locals))
        except:
            mynames.append(getObjName(myobj,_locals))
            pass
    return mynames

def removeSeqEntriesEveryWhere(seq,_locals):
    myobjs = seq.expandAndClone()._seq._collection
    for name,obj in _locals.iteritems():
        for myobj in myobjs:
            try:
                obj.remove(myobj)
            except:
                pass
