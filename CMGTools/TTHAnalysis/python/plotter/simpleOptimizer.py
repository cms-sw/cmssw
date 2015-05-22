#!/usr/bin/env python
#from mcPlots import *
from CMGTools.TTHAnalysis.plotter.mcPlots import *


class Param:
    def __init__(self,name,value):
        self.name  = name
        self.value = value
    def getPoints(self):
        return [self.value]

class DiscreteParam(Param):
    def __init__(self,name,value,values):
        Param.__init__(self,name,value)
        self.values = values
    def getPoints(self):
        return self.values

class ParamSampler:
    def __init__(self,params):
        self._params = params
        self._cache = {}
    def _addAllPoints(self, param, points):
        ## trivial case
        if len(points) == 0:
            return [ [ (param.name,p) ] for p in param.getPoints() ]
        ## generic case, recursive
        ret = []
        for p in param.getPoints():
            for rest in points:
                pfull = [ (param.name,p) ] + rest
                ret.append(pfull)
        return ret
    def getAllPoints(self):
        allpoints = []
        for p in self._params:
            allpoints = self._addAllPoints(p, allpoints)
        return [ dict(x) for x in allpoints ]
    def optimize(self,mca,cut,fom,algo):
        if algo == "scan": return self.scan(mca,cut,fom,algo)
        if algo == "walk": return self.walk(mca,cut,fom,algo)
        if "cloud:" in algo: return self.cloud(mca,cut,fom,algo)
        if "scipy:" in algo: return self.scipyOpt(mca,cut,fom,algo)
    def onepoint(self,mca,cut,fom,point):
        if self._cache != None:
            key = repr(point)
            if key in self._cache:
                return self._cache[key]
        mycut = CutsFile([('all',cut.format(**point))])   ## need the '**' to unpack a map
        yields = mca.getYields(mycut,nodata=True,noEntryLine=True)
        ret = fom(yields)
        if self._cache != None:
            self._cache[repr(point)] = ret
        return ret
    def scan(self,mca,cut,fom,algo):
        best, bval = None, None
        allpoints = self.getAllPoints()
        print "I have %d points to test... please sit back and relax." % len(allpoints)
        for i,point in enumerate(allpoints):
            print "trying point ",i,": ",point,
            fomval = self.onepoint(mca,cut,fom,point)
            print "     ---> ",fomval
            if best == None or bval[0] < fomval[0]:
                best = point
                bval = fomval
        return (best,bval)
    def scan1(self,mca,cut,fom,algo,param,cursor,value):
        print "Scanning %s values [ %s ] starting from %s" % (param.name, param.getPoints(), cursor )
        best, bval = cursor[param.name], value
        for pv in param.getPoints():
            if pv == best: continue
            cursor[param.name] = pv
            fomval = self.onepoint(mca,cut,fom,cursor)
            if fomval[0] > bval[0]:
                best = pv; bval = fomval
        cursor[param.name] = best;
        return (cursor, best)
    def expand(self,mca,cut,fom,cursor,value,size=1):
        neighbours = [ [] ] # list of points, each of which is a lists of pairs (name,value)
        for param in self._params:
            values = param.getPoints()
            index = values.index(cursor[param.name])
            newneigh = []
            for i in xrange(max(0,index-size),min(index+size,len(values)-1)+1):
                for prefix in neighbours:
                    newneigh.append( prefix + [ (param.name, values[i]) ] )
            neighbours = newneigh
        neighbours = [ dict(x) for x in neighbours ]
        print "Scanning all the %d neighbours within %d: " % (len(neighbours), size)
        best, bval = cursor, value
        for i,point in enumerate(neighbours):
            fomval = self.onepoint(mca,cut,fom,point)
            if fomval[0] > bval[0]:
                print "  %6d ---> %s (improvement)" % (i,fomval) 
                best = point; bval = fomval
        if bval[0] > value[0]:
            print "   found new best point %s fom value %s" % (best, bval)
        return (best, bval, bval[0] > value[0])
    def step1(self,mca,cut,fom,algo,param,cursor,value):
        print "Stepping %s values [ %s ] starting from %s at %s" % (param.name, param.getPoints(), cursor, value )
        best, bval = cursor[param.name], value
        values  = param.getPoints()
        index = values.index(cursor[param.name])
        ileft = index - 1; moved = False
        while ileft >= 0:
            cursor[param.name] = values[ileft]
            fomval = self.onepoint(mca,cut,fom,cursor)
            if fomval[0] > bval[0]:
                best = values[ileft]; bval = fomval; moved = True
                ileft -= 1
            else:
                break
        if moved: 
            cursor[param.name] = best; 
            print "   moved %s to %s, fom value %s" % (param.name, best, bval)
            return (cursor, bval, True)
        iright = index + 1
        while iright < len(values):
            cursor[param.name] = values[iright]
            fomval = self.onepoint(mca,cut,fom,cursor)
            if fomval[0] > bval[0]:
                best = values[iright]; bval = fomval; moved = True
                iright += 1
            else:
                break
        if moved:
            cursor[param.name] = best;
            print "   moved %s to %s, fom value %s" % (param.name, best, bval)
            return (cursor, bval, True)
        cursor[param.name] = best;
        print "   did not move %s from %s at  %s" % (param.name, best, bval)
        return (cursor, bval, False)
    def walk(self,mca,cut,fom,algo):
        cursor = dict( [ (p.name,p.value) for p in self._params ] )
        fomval = self.onepoint(mca,cut,fom,cursor)
        print "Starting point: %s (fomval = %s)" % (cursor, fomval)
        active  = dict([ (p.name,True) for p in self._params ])
        constant = dict([ (p.name,len(p.getPoints()) == 1) for p in self._params ])
        for iter in xrange(1000):
            walked = False
            lastactive = {}
            for p in self._params:
                if constant[p.name]: continue
                if active[p.name]:
                    lastactive[p.name] = True
                    (cursor, fomval, act) = self.step1(mca,cut,fom,algo,p,cursor,fomval)
                    active[p.name] = act
                    if act: walked = True
            if walked: continue
            for p in self._params:
                if constant[p.name]: continue
                if p.name in lastactive: continue # these are already at the maximum
                (cursor, fomval, act) = self.step1(mca,cut,fom,algo,p,cursor,fomval)
                active[p.name] = act
                if act: walked = True
            if walked:
                for pn in active.keys(): active[pn] = True
            else:
                break
        return cursor, fomval
    def cloud(self,mca,cut,fom,algo):
        cursor = dict( [ (p.name,p.value) for p in self._params ] )
        fomval = self.onepoint(mca,cut,fom,cursor)
        print "Starting point: %s (fomval = %s)" % (cursor, fomval)
        for iter in xrange(1000):
            (cursor, fomval, moved) = self.expand(mca,cut,fom,cursor,fomval,size=int(algo.replace("cloud:","")))
            if not moved:
                break
        return (cursor,fomval)
    def scipyOpt(self,mca,cut,fom,algo):
        cursor = dict( [ (p.name,p.value) for p in self._params ] )
        fomval = self.onepoint(mca,cut,fom,cursor)
        print "Starting point: %s (fomval = %s)" % (cursor, fomval)
        import scipy.optimize
        minimizer = getattr(scipy.optimize,algo.replace("scipy:",""))
        fun = ParamSampelFunctor(self,mca,cut,fom)
        ret = minimizer(fun, fun.x0)
        cursor = fun.int2ext(ret)
        fomval = self.onepoint(mca,cut,fom,cursor)
        print "Ending point: %s (fomval = %s)" % (cursor, fomval)
        return (cursor,fomval)
class ParamSampelFunctor:
    def __init__(self,paramSampler,mca,cut,fom):
        self._sampler = paramSampler
        self._mca = mca
        self._cut = cut
        self._fom = fom
        self._i = 0
        self.xnames = []
        self.xmin   = []
        self.xmax   = []
        self.x0     = []
        self.consts = []
        for p in self._sampler._params:
            points = p.getPoints()
            if len(points) > 1:
                self.xnames.append(p.name)
                self.xmax.append(max(points))
                self.xmin.append(min(points))
                self.x0.append(0.5)
            else:
                self.consts.append((p.name,p.value))
    def int2ext(self,x):
        point = dict(self.consts)
        for (n,xmin,xmax,xraw) in zip(self.xnames,self.xmin,self.xmax,x):
            point[n] = max(0,0.5*((xraw-0.5)*10+1))*(xmax-xmin) + xmin
        return point
    def __call__(self,x):
        point = self.int2ext(x)
        fomval = self._sampler.onepoint(self._mca,self._cut,self._fom, point)
        self._i += 1
        print "  %6d: %s --> %s " % (self._i, point,fomval)
        return -fomval[0]
        
class ParamFile:
    def __init__(self,params,options):
        self._params = []
        for line in open(params,'r'):
            if re.match("\s*#.*", line): continue
            line = re.sub("#.*","",line)
            extra = {}
            if ";" in line:
                (line,more) = line.split(";")[:2]
                for setting in [f.strip() for f in more.split(',')]:
                    if "=" in setting: 
                        (key,val) = [f.strip() for f in setting.split("=")]
                        extra[key] = eval(val)
                    else: extra[setting] = True
            field = [f.strip() for f in line.split(':')]
            if len(field) < 2: continue
            if len(field) == 2:
                self._params.append( Param(field[0], float(field[1])) )
            else:
                self._params.append( DiscreteParam(field[0], float(field[1]), [float(x) for x in field[2].split(",")]) )
    def allParams(self):
        return self._params[:]

class SimpleFOM:
    def __init__(self,mca,syst=0.0,includeS=False):
        self._signals     = mca.listSignals()
        self._backgrounds = [ (b,syst) for b in mca.listBackgrounds() ]
        self._includeS = includeS
    def __call__(self,report):
        tots, totb, totbsyst = 0.,0.,0.
        for s in  self._signals:
            if s not in report: continue
            tots += report[s][-1][1][0]
        for b,syst in self._backgrounds:
            if b not in report: continue
            totb     += report[b][-1][1][0]
            totbsyst += (syst*report[b][-1][1][0])**2
        if self._includeS: totb += tots
        return (tots/sqrt(totb + totbsyst**2), "S=%.2f, B=%.2f" % (tots, totb))
class PunziFOM:
    def __init__(self,mca,syst=0.0,a=3):
        self._signals     = mca.listSignals()
        self._backgrounds = [ (b,syst) for b in mca.listBackgrounds() ]
        self._a = a
    def __call__(self,report):
        tots, totb, totbsyst = 0.,0.,0.
        for s in  self._signals:
            if s not in report: continue
            tots += report[s][-1][1][0]
        for b,syst in self._backgrounds:
            if b not in report: continue
            totb     += report[b][-1][1][0]
            totbsyst += (syst*report[b][-1][1][0])**2
        return (tots/(0.5*self._a + sqrt(totb + totbsyst**2)), "S=%.2f, B=%.2f" % (tots, totb))

if __name__ == "__main__":
    from optparse import OptionParser
    parser = OptionParser(usage="%prog [options] mc.txt cuts.txt algo params.txt")
    addMCAnalysisOptions(parser)
    (options, args) = parser.parse_args()
    mca  = MCAnalysis(args[0],options)
    cuts = CutsFile(args[1],options)
    #fom = SimpleFOM(mca)
    fom = PunziFOM(mca,a=2)
    algo = args[2]
    params = ParamFile(args[3],options)
    sampler = ParamSampler(params.allParams())
    (point,val) = sampler.optimize(mca,cuts.allCuts(),fom,algo)
    print "The best set of parameters is "
    for k,v in point.iteritems(): print "   %-20s: %8.3f" % (k,v)
    print "The corresponding figure of merit is ",val
    cuts.setParams(point)
    report = mca.getYields(cuts,nodata=True)
    mca.prettyPrint(report)
