def plotSequences(seq,filename):
    from sys import stderr, argv
    from os import popen
    from os.path import basename
    from re import sub;
    import FWCore.ParameterSet.Config as cms
    stderr.write("Writing plot to %s\n" % (filename,))
    dot = popen("dot -Tpng > %s" % (filename,), "w")
    dot.write("digraph G { \n rankdir=\"LR\" \n")
    class visitor(object):
        def __init__(self,seq,dot):
            self._dot = dot
            self._stack = []
            self._seq = seq.label()
            self._dot.write( "%s [  shape=rect style=filled fillcolor=%s label=\"%s\" ]" % (self._seq,'orange',self._seq) + "\n" )
        def seq(self, seq):
            self._stack.append(self._seq)
            self._seq = seq.label()
        def enter(self,v):
            if isinstance(v, cms.Sequence):
                self._dot.write( "%s [  shape=rect style=filled fillcolor=%s label=\"%s\" ]" % (v.label(),'orange',v.label()) + "\n" )
                self.dep(v)
                self.seq(v)
            if isinstance(v, (cms.EDProducer, cms.EDFilter, cms.EDAnalyzer)):
                self._dot.write( "%s [  shape=rect style=filled fillcolor=%s label=\"%s\" ]" % (v.label(),'green',v.label()) + "\n" ) 
                self.dep(v)
        def leave(self,v):
            if isinstance(v, cms.Sequence):
                self._seq = self._stack.pop()
        def dep(self,v):
            self._dot.write("%s -> %s" %(v.label(), self._seq) +"\n")
    seq.visit(visitor(seq,dot))
    dot.write("}\n")
    dot.close()

def plotModuleInputs(seq,filename,printOuter=True,printLinkNames=True):
    from sys import stderr, argv
    from os import popen
    from os.path import basename
    from re import sub;
    import FWCore.ParameterSet.Config as cms
    stderr.write("Writing plot to %s\n" % (filename,))
    dot = popen("dot -Tpng > %s" % (filename,), "w")
    #dot = open("%s.dot" % (filename,), "w")
    dot.write("digraph G { \n rankdir=\"LR\" \n")
    deps = {}; alls = {}
    modules = []
    class visitor(object):
        def enter(self,v):
            if isinstance(v, (cms.EDProducer, cms.EDFilter, cms.EDAnalyzer)):
                modules.append(v)
        def leave(self,v):
            pass
    def greptags(ps,basename=""):
        ret = []
        for pn, pv in ps.parameters_().items():
            type = pv.configTypeName()
            if type == 'InputTag'    : ret.append( (basename+pn, pv.configValue()) )
            elif type == 'VInputTag' : ret += [ ("%s%s[%d]"%(basename,pn,i+1),v.configValue()) for i,v in enumerate(pv.value()) ]
            elif type == 'PSet'      : ret += greptags(pv, basename+pn+'.')
            elif type == 'VPset'     : 
                for r1 in [greptags(pvi, basename+pn+'.') for pvi in pv.value()]: ret += r1
        return ret
    def escapeParValue(name): return sub(r":.*","", name)
    seq.visit(visitor())
    for m in modules:
        dot.write( "%s [  shape=rect style=filled fillcolor=%s label=\"%s\" ]" % (m.label(),'green',m.label()) + "\n")
        tags = greptags(m)
        #stderr.write("Tags for %s: %s\n" % (m.label(), tags))
        deps[m.label()] = tags;
        if m.label() not in alls: alls[m.label()]=True
        for (tn,tv) in tags:
            tve = escapeParValue(tv)
            if tve not in alls: alls[tve]=True
    names = deps.keys();
    if printOuter: names = alls.keys()
    done = {}
    for n in names:
        ne = escapeParValue(n)
        if ne not in deps:
            dot.write( "%s [  shape=rect style=filled fillcolor=%s label=\"%s\" ]" % (ne,'yellow',ne) + "\n")
        else:
            for tn,tv in deps[ne]:
                tve = escapeParValue(tv)
                if printOuter or tve in deps:
                    style = ""
                    if printLinkNames: style = " [label=\"%s\" ]" %(tn,)
                    dot.write(  "%s -> %s%s\n"%(tve,ne,style))
    dot.write("}\n")
    dot.close()


